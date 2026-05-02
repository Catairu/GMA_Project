import argparse
import random
from collections import OrderedDict
from collections.abc import Callable

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn

from src.datasets.loader import load_graph_dataset
from src.federated import FedAvgServer, FederatedClient
from src.utilities.partitioning import (
    dirichlet_partition,
    kmeans_partition,
    metis_partition,
    random_partition,
)


class GraphSAGEClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_dim, hidden_dim, aggregator_type="mean")
        self.conv2 = dglnn.SAGEConv(hidden_dim, out_dim, aggregator_type="mean")
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(graph, x)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(graph, h)
        return h


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_local(
    labels: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stratified local split, preserving each client's label distribution."""
    g = torch.Generator().manual_seed(seed)
    labels = labels.detach().cpu().long()

    train_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []

    for class_id in torch.unique(labels).tolist():
        class_idx = torch.where(labels == class_id)[0]
        class_idx = class_idx[torch.randperm(class_idx.numel(), generator=g)]

        n_class = class_idx.numel()
        n_train = int(train_ratio * n_class)
        if n_train == 0 and n_class > 0 and train_ratio > 0:
            n_train = 1

        n_val = int(val_ratio * n_class)
        n_val = min(n_val, n_class - n_train)

        train_parts.append(class_idx[:n_train])
        val_parts.append(class_idx[n_train:n_train + n_val])
        test_parts.append(class_idx[n_train + n_val:])

    def concat_and_shuffle(parts: list[torch.Tensor]) -> torch.Tensor:
        non_empty = [part for part in parts if part.numel() > 0]
        if not non_empty:
            return torch.empty(0, dtype=torch.long)
        idx = torch.cat(non_empty)
        return idx[torch.randperm(idx.numel(), generator=g)]

    return (
        concat_and_shuffle(train_parts),
        concat_and_shuffle(val_parts),
        concat_and_shuffle(test_parts),
    )


def compute_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = labels.detach().cpu().long().bincount(minlength=num_classes).float()
    total = counts.sum()
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    weights = total / (num_classes * counts.clamp(min=1.0))
    weights[counts == 0] = 0.0
    return weights


def label_counts(labels: torch.Tensor, num_classes: int) -> list[int]:
    return labels.detach().cpu().long().bincount(minlength=num_classes).tolist()


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
        return dirichlet_partition(labels, graph, n_clients, alpha=dirichlet_alpha, seed=seed)
    if method == "kmeans":
        return kmeans_partition(graph, n_clients, features=features, seed=seed)
    raise ValueError(f"Unsupported partition method: {method}")


def evaluate_federated(server, clients, model_fn, mask_type="test"):
    """Evaluate the global model on all clients with sample-weighted metrics."""
    global_state = server.get_global_state()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_confmat = None

    for client in clients:
        loss, correct, n_samples, confmat = client.evaluate(
            global_state, model_fn, mask_type=mask_type
        )
        if n_samples == 0:
            continue

        total_loss += loss * n_samples
        total_correct += correct
        total_samples += n_samples
        total_confmat = confmat if total_confmat is None else total_confmat + confmat

    if total_samples == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
        }

    support = total_confmat.sum(dim=1)
    pred_count = total_confmat.sum(dim=0)
    tp = total_confmat.diag()
    eps = 1e-12
    per_class_recall = tp / (support + eps)
    per_class_precision = tp / (pred_count + eps)
    per_class_f1 = 2.0 * per_class_precision * per_class_recall / (
        per_class_precision + per_class_recall + eps
    )
    valid_classes = support > 0

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "balanced_accuracy": per_class_recall[valid_classes].mean().item(),
        "macro_precision": per_class_precision[valid_classes].mean().item(),
        "macro_recall": per_class_recall[valid_classes].mean().item(),
        "macro_f1": per_class_f1[valid_classes].mean().item(),
    }


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
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio must be > 0 and train+val must be < 1.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")


def is_better_metric(metric: str, score: float, best_score: float) -> bool:
    if metric == "loss":
        return score < best_score
    return score > best_score


def run_experiment(
    args: argparse.Namespace,
    *,
    verbose: bool = True,
    evaluate_test: bool = True,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, object]:
    validate_args(args)

    def log(message: str) -> None:
        if verbose:
            print(message)

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_clients = args.n_clients
    global_rounds = args.global_rounds
    local_epochs = args.local_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    hidden_dim = args.hidden_dim

    # Questa parte è ridondante e smaltibile utilizzando solo dglgraph invece di creare quell'oggetto
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
        n_clients=n_clients,
        method=args.partition_method,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
    )

    in_dim = features.shape[1]
    log(f"Input feature dim: {in_dim}")

    model_fn = lambda: GraphSAGEClassifier(
        in_dim,
        hidden_dim,
        num_classes,
        dropout=args.dropout,
    )

    server = FedAvgServer(model_fn=model_fn, device=device)
    clients: list[FederatedClient] = []

    for client_id in range(n_clients):
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
            lr=lr,
            local_epochs=local_epochs,
            weight_decay=weight_decay,
            class_weights=client_class_weights,
            device=device,
        )

        if client.num_train_samples == 0:
            log(f"Client {client_id}: skipped (no train samples)")
            continue

        clients.append(client)
        log(
            f"Client {client_id}: "
            f"nodes={subg.num_nodes()}, "
            f"train={client.num_train_samples}, "
            f"val={int(val_mask.sum().item())}, "
            f"test={int(test_mask.sum().item())}, "
            f"train_labels={label_counts(subg.ndata['label'][train_mask], num_classes)}, "
            f"val_labels={label_counts(subg.ndata['label'][val_mask], num_classes)}, "
            f"test_labels={label_counts(subg.ndata['label'][test_mask], num_classes)}"
        )

    if not clients:
        raise RuntimeError("No client has train samples. Check partitioning/split.")

    history: list[dict[str, object]] = []
    best_round = 0
    best_val_metrics: dict[str, float] = {}
    best_test_metrics: dict[str, float] | None = None
    final_val_metrics: dict[str, float] = {}
    final_test_metrics: dict[str, float] | None = None
    best_score = float("inf") if args.selection_metric == "loss" else -float("inf")

    for rnd in range(1, global_rounds + 1):
        global_state = server.get_global_state()
        updates: list[tuple[OrderedDict[str, torch.Tensor], int]] = []
        weighted_round_loss = 0.0
        round_samples = 0

        for client in clients:
            client_state, n_samples, c_loss = client.train_one_round(global_state, model_fn)
            updates.append((client_state, n_samples))
            weighted_round_loss += c_loss * n_samples
            round_samples += n_samples

        server.aggregate(updates)
        train_loss = weighted_round_loss / max(round_samples, 1)

        should_eval = (
            rnd == 1
            or rnd == global_rounds
            or (args.eval_every > 0 and rnd % args.eval_every == 0)
        )
        if should_eval:
            val_metrics = evaluate_federated(server, clients, model_fn, "val")
            test_metrics = (
                evaluate_federated(server, clients, model_fn, "test")
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

            history_entry: dict[str, object] = {
                "round": rnd,
                "train_loss": train_loss,
                "val": val_metrics,
            }
            if test_metrics is not None:
                history_entry["test"] = test_metrics
            history.append(history_entry)

            if progress_callback is not None:
                progress_callback(rnd, val_metrics)

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
            log(message)
        else:
            log(f"Round {rnd:04d} | train_loss={train_loss:.4f}")

    return {
        "best_round": best_round,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "history": history,
    }


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
