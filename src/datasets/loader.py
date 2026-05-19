from pathlib import Path

import dgl
import torch
from dgl.data import FraudDataset


class GraphFraudDataset:
    def __init__(
        self,
        name: str = "amazon",
        train_size: float = 1.0,
        val_size: float = 0.0,
        root: str = "./data",
    ) -> None:
        name = name.lower().strip()
        supported_datasets = ("amazon", "yelp", "cora")
        if name not in supported_datasets:
            raise ValueError(
                f"Dataset '{name}' is not supported. "
                f"Choose one of: {', '.join(supported_datasets)}."
            )

        print(f"--- Loading dataset: {name} ---")
        self.name = name

        if name == "cora":
            self._load_cora_from_pyg(root=root)
        else:
            self._load_fraud_dataset(
                name=name,
                train_size=train_size,
                val_size=val_size,
            )

        print(
            f"Graph: {self.graph.num_nodes()} nodes, "
            f"{self.graph.num_edges()} edges, "
            f"{self.features.shape[1]} feature dimensions, "
            f"{self.num_classes} classes."
        )

    def _load_fraud_dataset(
        self,
        name: str,
        train_size: float,
        val_size: float,
    ) -> None:
        self.raw = FraudDataset(name, train_size=train_size, val_size=val_size)
        self.num_classes = int(self.raw.num_classes)

        g_hetero = self.raw[0]
        self.graph = dgl.to_simple(
            dgl.to_homogeneous(g_hetero, ndata=["feature", "label"]),
            return_counts="weight",
        )
        self.features = self.graph.ndata["feature"]
        self.labels = self.graph.ndata["label"].long()

    def _load_cora_from_pyg(self, root: str) -> None:
        try:
            from torch_geometric.datasets import Planetoid
        except ImportError as exc:
            raise ImportError(
                "To load Cora from PyG, install torch_geometric first. "
                "Example: pip install torch_geometric"
            ) from exc

        pyg_root = Path(root) / "pyg"
        self.raw = Planetoid(root=str(pyg_root), name="Cora")
        data = self.raw[0]

        src, dst = data.edge_index
        self.graph = dgl.graph(
            (src, dst),
            num_nodes=data.num_nodes,
            idtype=torch.int64,
        )

        # Keep the same node-data names used by the Fraud datasets.
        self.graph.ndata["feature"] = data.x.float()
        self.graph.ndata["label"] = data.y.long()

        # Optional but useful if later code needs the standard Planetoid splits.
        if hasattr(data, "train_mask") and data.train_mask is not None:
            self.graph.ndata["train_mask"] = data.train_mask
        if hasattr(data, "val_mask") and data.val_mask is not None:
            self.graph.ndata["val_mask"] = data.val_mask
        if hasattr(data, "test_mask") and data.test_mask is not None:
            self.graph.ndata["test_mask"] = data.test_mask

        # Match the edge-data convention created by dgl.to_simple(..., return_counts="weight").
        self.graph.edata["weight"] = torch.ones(
            self.graph.num_edges(),
            dtype=torch.float32,
        )

        self.features = self.graph.ndata["feature"]
        self.labels = self.graph.ndata["label"]
        self.num_classes = int(self.raw.num_classes)


def load_graph_dataset(name: str = "amazon", root: str = "./data") -> GraphFraudDataset:
    return GraphFraudDataset(name=name, root=root)


if __name__ == "__main__":
    ds = load_graph_dataset("cora")
    print(ds.graph.edata["weight"])
    print(ds.graph.ndata["feature"][0])
    print(ds.graph.ndata["label"][0])
    print(ds.num_classes)