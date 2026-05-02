import dgl
import torch
from dgl.data import FraudDataset


class GraphFraudDataset:
    def __init__(
        self,
        name: str = "amazon",
        train_size: float = 1.0,
        val_size: float = 0.0,
    ) -> None:
        name = name.lower().strip()
        if name not in ("amazon", "yelp"):
            raise ValueError(
                f"Dataset '{name}' is not supported. Choose 'amazon' or 'yelp'."
            )

        print(f"--- Loading dataset: {name} ---")
        self.name = name
        self.raw = FraudDataset(name, train_size=train_size, val_size=val_size)
        self.num_classes = int(self.raw.num_classes)
        g_hetero = self.raw[0]
        self.graph = dgl.to_simple(dgl.to_homogeneous(g_hetero, ndata=["feature", "label"]), return_counts="weight")
        self.features = self.graph.ndata["feature"]
        self.labels = self.graph.ndata["label"].long()

        print(
            f"Graph: {self.graph.num_nodes()} nodes, "
            f"{self.graph.num_edges()} edges, "
            f"{self.features.shape[1]} feature dimensions, "
            f"{self.num_classes} classes."
        )


def load_graph_dataset(name: str = "amazon") -> tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor]:
    ds = GraphFraudDataset(name=name)
    return ds


if __name__ == "__main__":
    ds = load_graph_dataset("yelp")
    print(ds.graph.edata["weight"])
    print(ds.graph.ndata["feature"][0])
    print(ds.graph.ndata["label"][0])
    print(ds.num_classes)
