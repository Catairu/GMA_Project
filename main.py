from __future__ import annotations

import argparse
from collections import OrderedDict

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
from dgl.data import FraudDataset

from src.federated import FedAvgServer, FederatedClient


class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_dim, hidden_dim, aggregator_type="mean")
        self.conv2 = dglnn.SAGEConv(hidden_dim, out_dim, aggregator_type="mean")
        self.dropout = nn.Dropout(0.2)

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(graph, x)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(graph, h)
        return h


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        del graph
        return self.net(x)


def build_global_split(
    n_nodes: int, train_ratio: float = 0.8, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_nodes, generator=g)
    cut = int(train_ratio * n_nodes)
    return perm[:cut], perm[cut:]


def evaluate(
    model: nn.Module,
    graph: dgl.DGLGraph,
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    device: str,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        g = graph.to(device)
        logits = model(g, x.to(device))
        pred = logits.argmax(dim=1).cpu()[idx]
        y_true = y[idx]
        n_classes = int(y.max().item()) + 1

        # Confusion matrix on selected split.
        confmat = torch.zeros(n_classes, n_classes, dtype=torch.float32)
        for t, p in zip(y_true, pred):
            confmat[int(t.item()), int(p.item())] += 1.0

        support = confmat.sum(dim=1)
        pred_count = confmat.sum(dim=0)
        tp = confmat.diag()

        eps = 1e-12
        per_class_recall = tp / (support + eps)
        per_class_precision = tp / (pred_count + eps)
        per_class_f1 = 2.0 * per_class_precision * per_class_recall / (
            per_class_precision + per_class_recall + eps
        )
        valid_class_mask = support > 0

        acc = (pred == y_true).float().mean().item()
        balanced_acc = per_class_recall[valid_class_mask].mean().item()
        macro_precision = per_class_precision[valid_class_mask].mean().item()
        macro_recall = per_class_recall[valid_class_mask].mean().item()
        macro_f1 = per_class_f1[valid_class_mask].mean().item()

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Federated GraphSAGE on DGL FraudDataset with imbalance-aware metrics."
    )
    parser.add_argument("--n-clients", type=int, default=4)
    parser.add_argument("--global-rounds", type=int, default=1000)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--model",
        type=str,
        default="graphsage",
        choices=["graphsage", "mlp"],
        help="Backbone model used by each federated client.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Use inverse-frequency class weights in CrossEntropy loss.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = FraudDataset("amazon", train_size=1.0, val_size=0.0)
    g_hetero = dataset[0]
    num_classes = dataset.num_classes

    features = g_hetero.ndata["feature"]
    labels = g_hetero.ndata["label"].long()
    g_homo = dgl.to_homogeneous(g_hetero)

    n_clients = args.n_clients
    global_rounds = args.global_rounds
    local_epochs = args.local_epochs
    lr = args.lr
    weight_decay = args.weight_decay
    hidden_dim = args.hidden_dim

    class_counts = labels.bincount(minlength=num_classes).float()
    print(f"Classes: {labels.unique().tolist()} | Counts: {class_counts.int().tolist()}")

    class_weights = None
    if args.use_class_weights:
        # Inverse-frequency weights to reduce majority class dominance.
        class_weights = class_counts.sum() / (num_classes * class_counts.clamp(min=1.0))
        print(f"Using class weights: {[round(v, 4) for v in class_weights.tolist()]}")

    partitions = dgl.metis_partition(g_homo, k=n_clients)

    train_idx, test_idx = build_global_split(
        g_homo.num_nodes(), train_ratio=args.train_ratio, seed=args.seed
    )
    train_mask = torch.zeros(g_homo.num_nodes(), dtype=torch.bool)
    train_mask[train_idx] = True

    in_dim = features.shape[1]
    if args.model == "mlp":
        model_fn = lambda: MLPClassifier(in_dim, hidden_dim, num_classes)
    else:
        model_fn = lambda: GraphSAGEClassifier(in_dim, hidden_dim, num_classes)

    server = FedAvgServer(model_fn=model_fn, device=device)
    clients: list[FederatedClient] = []

    for client_id in range(n_clients):
        subg = partitions[client_id]
        global_ids = subg.ndata["_ID"].to(torch.long)
        subg.ndata["feature"] = features[global_ids]
        subg.ndata["label"] = labels[global_ids]

        inner_mask = subg.ndata["inner_node"].bool()
        local_train_mask = inner_mask & train_mask[global_ids]
        if local_train_mask.sum().item() == 0:
            continue

        client = FederatedClient(
            client_id=client_id,
            subgraph=subg,
            train_mask=local_train_mask,
            lr=lr,
            local_epochs=local_epochs,
            weight_decay=weight_decay,
            class_weights=class_weights,
            device=device,
        )
        clients.append(client)
        print(f"Client {client_id}: train_samples={client.num_train_samples}")

    if not clients:
        raise RuntimeError("No client has train samples. Check partitioning/split.")

    for rnd in range(1, global_rounds + 1):
        global_state = server.get_global_state()
        updates: list[tuple[OrderedDict[str, torch.Tensor], int]] = []
        round_loss = 0.0

        for client in clients:
            client_state, n_samples, c_loss = client.train_one_round(global_state, model_fn)
            updates.append((client_state, n_samples))
            round_loss += c_loss

        server.aggregate(updates)
        avg_client_loss = round_loss / len(clients)
        metrics = evaluate(server.global_model, g_homo, features, labels, test_idx, device)
        print(
            f"Round {rnd:02d} | avg_client_loss={avg_client_loss:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"macro_p={metrics['macro_precision']:.4f} | "
            f"macro_r={metrics['macro_recall']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()