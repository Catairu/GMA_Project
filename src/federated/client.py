from __future__ import annotations

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
        lr: float = 1e-3,
        local_epochs: int = 1,
        weight_decay: float = 0.0,
        class_weights: torch.Tensor | None = None,
        device: str = "cpu",
    ) -> None:
        self.client_id = client_id
        self.subgraph = subgraph
        self.train_mask = train_mask.bool()
        self.local_epochs = local_epochs
        self.device = torch.device(device)
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

        self.num_train_samples = int(self.train_mask.sum().item())

    def train_one_round(
        self, global_state_dict: OrderedDict[str, torch.Tensor], model_fn
    ) -> tuple[OrderedDict[str, torch.Tensor], int, float]:
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

        if m.sum().item() == 0:
            updated_state = OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
            )
            return updated_state, 0, 0.0

        for _ in range(self.local_epochs):
            optimizer.zero_grad()
            logits = model(g_local, x)
            loss = self.criterion(logits[m], y[m])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(self.local_epochs, 1)
        updated_state = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
        )
        return updated_state, self.num_train_samples, avg_loss
