import torch


def split_local(
    labels: torch.Tensor,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a stratified local split preserving each client's label distribution."""
    generator = torch.Generator().manual_seed(seed)
    labels = labels.detach().cpu().long()

    train_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []

    for class_id in torch.unique(labels).tolist():
        class_idx = torch.where(labels == class_id)[0]
        class_idx = class_idx[torch.randperm(class_idx.numel(), generator=generator)]

        n_class = class_idx.numel()
        n_train = int(train_ratio * n_class)
        if n_train == 0 and n_class > 0 and train_ratio > 0:
            n_train = 1

        n_val = int(val_ratio * n_class)
        n_val = min(n_val, n_class - n_train)

        train_parts.append(class_idx[:n_train])
        val_parts.append(class_idx[n_train : n_train + n_val])
        test_parts.append(class_idx[n_train + n_val :])

    return (
        _concat_and_shuffle(train_parts, generator),
        _concat_and_shuffle(val_parts, generator),
        _concat_and_shuffle(test_parts, generator),
    )


def _concat_and_shuffle(
    parts: list[torch.Tensor],
    generator: torch.Generator,
) -> torch.Tensor:
    non_empty = [part for part in parts if part.numel() > 0]
    if not non_empty:
        return torch.empty(0, dtype=torch.long)

    idx = torch.cat(non_empty)
    return idx[torch.randperm(idx.numel(), generator=generator)]
