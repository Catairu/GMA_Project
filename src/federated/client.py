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

        self.lr = lr
        self.local_epochs = local_epochs
        self.weight_decay = weight_decay
        self.device = torch.device(device)

        self.num_train_samples = int(self.train_mask.sum().item())

        train_weights = (
            class_weights.to(self.device)
            if class_weights is not None
            else None
        )

        self.train_criterion = nn.CrossEntropyLoss(weight=train_weights)
        self.eval_criterion = nn.CrossEntropyLoss()

    def _build_train_subgraph(self) -> dgl.DGLGraph:
        """
        Costruisce un sottografo contenente solo i nodi di training.

        Questo evita che, durante il training, GraphSAGE faccia message passing
        usando nodi di validation/test presenti nel subgrafo locale.
        """
        train_nodes = torch.where(self.train_mask)[0]

        if train_nodes.numel() == 0:
            raise RuntimeError(
                f"Client {self.client_id} has no training nodes."
            )

        g_train = dgl.node_subgraph(self.subgraph, train_nodes)

        return g_train

    def train_one_round(
        self,
        global_state_dict: OrderedDict[str, torch.Tensor],
        model_fn,
    ) -> tuple[OrderedDict[str, torch.Tensor], int, float]:
        """
        Esegue un round locale di training.

        Il modello riceve i pesi globali dal server, si allena solo sui nodi
        train del client e restituisce i pesi aggiornati.
        """
        if self.num_train_samples == 0:
            return global_state_dict, 0, 0.0

        model = model_fn().to(self.device)
        model.load_state_dict(global_state_dict)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        g_train = self._build_train_subgraph().to(self.device)

        x = g_train.ndata["feature"]
        y = g_train.ndata["label"].long()

        total_loss = 0.0

        for _ in range(self.local_epochs):
            optimizer.zero_grad()

            logits = model(g_train, x)
            loss = self.train_criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(self.local_epochs, 1)

        updated_state = OrderedDict(
            (key, value.detach().cpu().clone())
            for key, value in model.state_dict().items()
        )

        return updated_state, self.num_train_samples, avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        state_dict: OrderedDict[str, torch.Tensor],
        model_fn,
        mask_type: str = "test",
    ) -> tuple[float, int, int, torch.Tensor]:
        """
        Valuta il modello sul subgrafo locale completo usando la maschera scelta.

        mask_type può essere:
        - "train"
        - "val"
        - "test"
        """
        if mask_type not in {"train", "val", "test"}:
            raise ValueError(
                f"mask_type must be one of 'train', 'val', 'test'. Got: {mask_type}"
            )

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