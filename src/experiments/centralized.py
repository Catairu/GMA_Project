import argparse

import torch
import torch.nn as nn

from src.datasets.loader import load_graph_dataset
from src.experiments.federated_graphsage import set_seed
from src.models import GraphSAGEClassifier
from src.training import compute_class_weights, split_local


def run_centralized(args: argparse.Namespace) -> dict[str, float]:
    """Ceiling baseline: trains GraphSAGE on the full graph without federation.

    Compute budget equivalent to federated: global_rounds * local_epochs total epochs.
    """
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_graph_dataset(args.dataset)
    graph, features, labels = data.graph, data.features, data.labels
    num_classes = data.num_classes

    train_idx, val_idx, test_idx = split_local(
        labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    in_dim = features.shape[1]
    model = GraphSAGEClassifier(in_dim, args.hidden_dim, num_classes, dropout=args.dropout).to(device)

    class_weights = compute_class_weights(labels[train_idx], num_classes).to(device)
    train_criterion = nn.CrossEntropyLoss(weight=class_weights)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    g = graph.to(device)
    x = features.to(device)
    y = labels.to(device)

    n = graph.num_nodes()
    train_mask = torch.zeros(n, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n, dtype=torch.bool, device=device)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    total_epochs = args.global_rounds * args.local_epochs
    eval_every = args.eval_every * args.local_epochs

    best_val_f1 = -1.0
    best_test_metrics: dict[str, float] = {}

    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(g, x)
        loss = train_criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        should_eval = epoch == 1 or epoch == total_epochs or epoch % eval_every == 0
        if not should_eval:
            continue

        model.eval()
        with torch.no_grad():
            logits = model(g, x)
            val_m = _compute_metrics(logits, y, val_mask, num_classes, eval_criterion)
            test_m = _compute_metrics(logits, y, test_mask, num_classes, eval_criterion)

        if val_m["macro_f1"] > best_val_f1:
            best_val_f1 = val_m["macro_f1"]
            best_test_metrics = test_m

    return best_test_metrics


def _compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
    criterion: nn.Module,
) -> dict[str, float]:
    n = int(mask.sum().item())
    if n == 0:
        return {k: 0.0 for k in ["loss", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "fraud_f1"]}

    loss = criterion(logits[mask], labels[mask]).item()
    preds = logits[mask].argmax(dim=1)
    y_true = labels[mask]
    correct = int((preds == y_true).sum().item())

    flat = torch.bincount(y_true * num_classes + preds, minlength=num_classes * num_classes)
    confmat = flat.reshape(num_classes, num_classes).float().cpu()

    support = confmat.sum(dim=1)
    pred_count = confmat.sum(dim=0)
    tp = confmat.diag()
    eps = 1e-12
    recall = tp / (support + eps)
    precision = tp / (pred_count + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    valid = support > 0

    fraud_f1 = f1[1].item() if num_classes > 1 and support[1] > 0 else 0.0

    return {
        "loss": loss,
        "accuracy": correct / n,
        "balanced_accuracy": recall[valid].mean().item(),
        "macro_precision": precision[valid].mean().item(),
        "macro_recall": recall[valid].mean().item(),
        "macro_f1": f1[valid].mean().item(),
        "fraud_f1": fraud_f1,
    }
