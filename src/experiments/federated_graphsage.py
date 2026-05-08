import argparse
import random
from collections import OrderedDict
from collections.abc import Callable

import dgl
import numpy as np
import torch

from src.datasets.loader import load_graph_dataset
from src.federated import FedAvgServer, FederatedClient
from src.models import GraphSAGEClassifier
from src.training import (
    compute_class_weights,
    evaluate_federated,
    is_better_metric,
    label_counts,
    split_local,
)
from src.utilities.partitioning import (
    dirichlet_partition,
    kmeans_partition,
    metis_partition,
    random_partition,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_partitions(
    graph: dgl.DGLGraph,
    labels: torch.Tensor,
    features: torch.Tensor,
    n_clients: int,
    method: str,
    dirichlet_alpha: float,
    seed: int,
) -> dict[int, dgl.DGLGraph]:
    if method == "metis":
        return metis_partition(graph, n_clients)
    if method == "random":
        return random_partition(graph, n_clients, seed=seed)
    if method == "dirichlet":
        return dirichlet_partition(
            labels,
            graph,
            n_clients,
            alpha=dirichlet_alpha,
            seed=seed,
        )
    if method == "kmeans":
        return kmeans_partition(graph, n_clients, features=features, seed=seed)
    raise ValueError(f"Unsupported partition method: {method}")


def add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=str,
        default="amazon",
        choices=["amazon", "yelp"],
    )
    parser.add_argument("--n-clients", type=int, default=10)
    parser.add_argument("--global-rounds", type=int, default=1000)
    parser.add_argument("--local-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--partition-method",
        type=str,
        default="metis",
        choices=["metis", "random", "dirichlet", "kmeans"],
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument(
        "--class-weighting",
        type=str,
        default="local",
        choices=["none", "local"],
        help="Class weights used only for local training loss.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="macro_f1",
        choices=[
            "loss",
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ],
        help="Validation metric used to select/tune the best run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dont-aggregate",
        action="store_true",
        help="If set, evaluates local client models without serveraggregation (for baseline with blind clients).",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Federated GraphSAGE with FedAvg on DGL fraud graph datasets."
    )
    add_experiment_args(parser)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_clients <= 0:
        raise ValueError("--n-clients must be > 0.")
    if args.global_rounds <= 0:
        raise ValueError("--global-rounds must be > 0.")
    if args.local_epochs <= 0:
        raise ValueError("--local-epochs must be > 0.")
    if args.dirichlet_alpha <= 0:
        raise ValueError("--dirichlet-alpha must be > 0.")
    if (
        args.train_ratio <= 0
        or args.val_ratio < 0
        or args.train_ratio + args.val_ratio >= 1
    ):
        raise ValueError("--train-ratio must be > 0 and train+val must be < 1.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")


def run_experiment(
    args: argparse.Namespace,
    *,
    verbose: bool = True,
    evaluate_test: bool = True,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, object]:
    def log(message: str) -> None:
        if verbose:
            print(message)

    validate_args(args)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_graph_dataset(args.dataset)
    graph = data.graph
    num_classes = data.num_classes
    features = data.features
    labels = data.labels

    log(f"Device: {device}")
    log(f"Partition method: {args.partition_method}")

    partitions = build_partitions(
        graph=graph,
        labels=labels,
        features=features,
        n_clients=args.n_clients,
        method=args.partition_method,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
    )

    in_dim = features.shape[1]
    log(f"Input feature dim: {in_dim}")

    model_fn = lambda: GraphSAGEClassifier(
        in_dim,
        args.hidden_dim,
        num_classes,
        dropout=args.dropout,
    )

    server = FedAvgServer(model_fn=model_fn, device=device)
    clients = _build_clients(args, partitions, num_classes, device, log)

    if not clients:
        raise RuntimeError("No client has train samples. Check partitioning/split.")

    return _run_training_loop(
        args=args,
        server=server,
        clients=clients,
        model_fn=model_fn,
        evaluate_test=evaluate_test,
        progress_callback=progress_callback,
        log=log,
    )


def _build_clients(
    args: argparse.Namespace,
    partitions: dict[int, dgl.DGLGraph],
    num_classes: int,
    device: str,
    log: Callable[[str], None],
) -> list[FederatedClient]:
    clients: list[FederatedClient] = []

    for client_id in range(args.n_clients):
        subg = partitions[client_id]
        train_idx, val_idx, test_idx = split_local(
            subg.ndata["label"],
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed + client_id,
        )
        train_mask = torch.zeros(subg.num_nodes(), dtype=torch.bool)
        val_mask = torch.zeros(subg.num_nodes(), dtype=torch.bool)
        test_mask = torch.zeros(subg.num_nodes(), dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        client_class_weights = None
        if args.class_weighting == "local":
            client_class_weights = compute_class_weights(
                subg.ndata["label"][train_mask],
                num_classes,
            )
            log(f"Client {client_id} class weights: {client_class_weights}")

        client = FederatedClient(
            client_id=client_id,
            subgraph=subg,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            lr=args.lr,
            local_epochs=args.local_epochs,
            weight_decay=args.weight_decay,
            class_weights=client_class_weights,
            device=device,
        )

        if client.num_train_samples == 0:
            log(f"Client {client_id}: skipped (no train samples)")
            continue

        clients.append(client)
        log(
            _format_client_summary(
                client_id,
                subg,
                train_mask,
                val_mask,
                test_mask,
                num_classes,
                client,
            )
        )

    return clients


def _format_client_summary(
    client_id: int,
    subg: dgl.DGLGraph,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    num_classes: int,
    client: FederatedClient,
) -> str:
    return (
        f"Client {client_id}: "
        f"nodes={subg.num_nodes()}, "
        f"train={client.num_train_samples}, "
        f"val={int(val_mask.sum().item())}, "
        f"test={int(test_mask.sum().item())}, "
        f"train_labels={label_counts(subg.ndata['label'][train_mask], num_classes)}, "
        f"val_labels={label_counts(subg.ndata['label'][val_mask], num_classes)}, "
        f"test_labels={label_counts(subg.ndata['label'][test_mask], num_classes)}"
    )


def _run_training_loop(
    args: argparse.Namespace,
    server: FedAvgServer,
    clients: list[FederatedClient],
    model_fn: Callable[[], torch.nn.Module],
    evaluate_test: bool,
    progress_callback: Callable[[int, dict[str, float]], None] | None,
    log: Callable[[str], None],
) -> dict[str, object]:
    ''' Every round, trains all clients and aggregates updates on the server, evaluates on val/test every eval_every rounds, tracks best round by selection_metric. Returns a dict with best round info and history. '''
    history: list[dict[str, object]] = []
    best_round = 0
    best_val_metrics: dict[str, float] = {}
    best_test_metrics: dict[str, float] | None = None
    final_val_metrics: dict[str, float] = {}
    final_test_metrics: dict[str, float] | None = None
    best_score = float("inf") if args.selection_metric == "loss" else -float("inf")

    for rnd in range(1, args.global_rounds + 1):
        train_loss = _train_round(server, clients, model_fn)

        should_eval = (
            rnd == 1
            or rnd == args.global_rounds
            or (args.eval_every > 0 and rnd % args.eval_every == 0)
        )
        if not should_eval:
            log(f"Round {rnd:04d} | train_loss={train_loss:.4f}")
            continue

        val_metrics = evaluate_federated(server, clients, model_fn, "val", args.dont_aggregate)
        test_metrics = (
            evaluate_federated(server, clients, model_fn, "test", args.dont_aggregate)
            if evaluate_test
            else None
        )
        final_val_metrics = val_metrics
        final_test_metrics = test_metrics

        val_score = val_metrics[args.selection_metric]
        if is_better_metric(args.selection_metric, val_score, best_score):
            best_score = val_score
            best_round = rnd
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics

        history.append(_build_history_entry(rnd, train_loss, val_metrics, test_metrics))

        if progress_callback is not None:
            progress_callback(rnd, val_metrics)

        log(_format_round_summary(rnd, train_loss, val_metrics, test_metrics))

    return {
        "best_round": best_round,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "history": history,
    }


def _train_round(
    server: FedAvgServer,
    clients: list[FederatedClient],
    model_fn: Callable[[], torch.nn.Module],
) -> float:
    global_state = server.get_global_state()
    updates: list[tuple[OrderedDict[str, torch.Tensor], int]] = [] # list of (client_state_dict, num_samples)
    weighted_round_loss = 0.0
    round_samples = 0

    for client in clients:
        client_state, n_samples, client_loss = client.train_one_round(
            global_state,
            model_fn,
        )
        updates.append((client.client_id, client_state, n_samples))
        weighted_round_loss += client_loss * n_samples
        round_samples += n_samples

    server.aggregate(updates)
    return weighted_round_loss / max(round_samples, 1)


def _build_history_entry(
    rnd: int,
    train_loss: float,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float] | None,
) -> dict[str, object]:
    history_entry: dict[str, object] = {
        "round": rnd,
        "train_loss": train_loss,
        "val": val_metrics,
    }
    if test_metrics is not None:
        history_entry["test"] = test_metrics
    return history_entry


def _format_round_summary(
    rnd: int,
    train_loss: float,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float] | None,
) -> str:
    message = (
        f"Round {rnd:04d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} "
        f"val_acc={val_metrics['accuracy']:.4f} "
        f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
        f"val_macro_f1={val_metrics['macro_f1']:.4f}"
    )
    if test_metrics is not None:
        message += (
            f" | test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} "
            f"test_bal_acc={test_metrics['balanced_accuracy']:.4f} "
            f"test_macro_f1={test_metrics['macro_f1']:.4f}"
        )
    return message
