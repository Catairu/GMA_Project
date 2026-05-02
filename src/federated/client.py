from collections import OrderedDict
import dgl
import torch
import torch.nn as nn


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        subgraph: dgl.DGLGraph,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
        test_mask: torch.Tensor,
        lr: float = 1e-3,
        local_epochs: int = 1,
        weight_decay: float = 0.0,
        class_weights: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        self.subgraph = subgraph
        self.train_mask = train_mask.bool()
        self.val_mask = val_mask.bool()
        self.test_mask = test_mask.bool()
        self.local_epochs = local_epochs
        self.device = torch.device(device)
        train_weights = class_weights.to(self.device) if class_weights is not None else None
        self.train_criterion = nn.CrossEntropyLoss(weight=train_weights)
        self.eval_criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_train_samples = int(self.train_mask.sum().item())

    def train_one_round(
        self, global_state_dict: OrderedDict[str, torch.Tensor], model_fn
    ) -> tuple[OrderedDict[str, torch.Tensor], int, float]:
        if self.num_train_samples == 0:
            return global_state_dict, 0, 0.0

        model = model_fn().to(self.device)
        model.load_state_dict(global_state_dict)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        total_loss = 0.0

        g_local = self.subgraph.to(self.device)
        x = g_local.ndata["feature"]
        y = g_local.ndata["label"].long()
        m = self.train_mask.to(self.device)

        for _ in range(self.local_epochs):
            optimizer.zero_grad()
            logits = model(g_local, x)
            loss = self.train_criterion(logits[m], y[m])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(self.local_epochs, 1)
        updated_state = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
        )
        return updated_state, self.num_train_samples, avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        state_dict: OrderedDict[str, torch.Tensor],
        model_fn,
        mask_type: str = "test",
    ) -> tuple[float, int, int, torch.Tensor]:
        model = model_fn().to(self.device)
        model.load_state_dict(state_dict)
        model.eval()

        mask = getattr(self, f"{mask_type}_mask").to(self.device)
        n_samples = int(mask.sum().item())
        if n_samples == 0:
            return 0.0, 0, 0, torch.zeros(0, 0, dtype=torch.float32)

        g_local = self.subgraph.to(self.device)
        x = g_local.ndata["feature"]
        y = g_local.ndata["label"].long()

        logits = model(g_local, x)
        loss = self.eval_criterion(logits[mask], y[mask]).item()

        preds = logits[mask].argmax(dim=1)
        y_true = y[mask]
        correct = int((preds == y_true).sum().item())
        n_classes = logits.shape[1]
        flat_confmat = torch.bincount(
            y_true * n_classes + preds,
            minlength=n_classes * n_classes,
        )
        confmat = flat_confmat.reshape(n_classes, n_classes).float().cpu()

        return loss, correct, n_samples, confmat
