from collections.abc import Callable

import torch

from src.federated import FedAvgServer, FederatedClient


ModelFactory = Callable[[], torch.nn.Module]


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


def evaluate_federated(
    server: FedAvgServer,
    clients: list[FederatedClient],
    model_fn: ModelFactory,
    mask_type: str = "test",
) -> dict[str, float]:
    """Evaluate the global model on all clients with sample-weighted metrics."""
    global_state = server.get_global_state()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_confmat = None

    for client in clients:
        loss, correct, n_samples, confmat = client.evaluate(
            global_state,
            model_fn,
            mask_type=mask_type,
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

    if total_confmat is None:
        raise RuntimeError("Confusion matrix was not computed despite non-empty samples.")

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


def is_better_metric(metric: str, score: float, best_score: float) -> bool:
    if metric == "loss":
        return score < best_score
    return score > best_score
