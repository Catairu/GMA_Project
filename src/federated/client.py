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
        early_stopping_patience: int = 4,
        early_stopping_delta: float = 1e-4,
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
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta

        train_weights = (
            class_weights.to(self.device)
            if class_weights is not None
            else None
        )

        self.train_criterion = nn.CrossEntropyLoss(weight=train_weights)
        self.eval_criterion = nn.CrossEntropyLoss()

    def _build_train_subgraph(self) -> dgl.DGLGraph:
        """
        Build a subgraph containing only training nodes.

        This prevents GraphSAGE from doing message passing through
        val/test nodes present in the local subgraph during training.
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
        Run one local training round.

        The model receives the global weights from the server, trains on the
        client's train nodes only, and returns the updated weights.
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
        
        # Early stopping variables
        best_val_loss = float("inf")
        best_state_dict = None
        patience_counter = 0
        
        epochs_run = 0
        for epoch in range(self.local_epochs):
            optimizer.zero_grad()

            logits = model(g_train, x)
            loss = self.train_criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epochs_run += 1

            # Evaluate on validation set every epoch to detect plateau precisely
            val_loss, _, n_val_samples, _ = self.evaluate(
                model.state_dict(), 
                model_fn,
                mask_type="val"
            )

            # If there are no validation samples, skip early-stopping update
            if n_val_samples == 0:
                # do not update best_val_loss or patience counter
                continue

            # Early stopping logic
            if val_loss < best_val_loss - self.early_stopping_delta:
                best_val_loss = val_loss
                best_state_dict = OrderedDict(
                    (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if self.early_stopping_patience > 0 and patience_counter >= self.early_stopping_patience:
                print(f"Client {self.client_id}: Early stopping at epoch {epoch+1} with best val loss {best_val_loss:.4f}")
                break

        # Return best state if found, otherwise final state
        if best_state_dict is not None:
            updated_state = best_state_dict
        else:
            updated_state = OrderedDict(
                (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
            )
        
        avg_loss = total_loss / max(epochs_run, 1)
        return updated_state, self.num_train_samples, avg_loss

    @torch.no_grad()
    def evaluate(
        self,
        state_dict: OrderedDict[str, torch.Tensor],
        model_fn,
        mask_type: str = "test",
    ) -> tuple[float, int, int, torch.Tensor]:
        """
        Evaluate the model on the full local subgraph using the chosen mask.

        mask_type must be one of: "train", "val", "test".
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