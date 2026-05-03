import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn


class GraphSAGEClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_dim, hidden_dim, aggregator_type="mean")
        self.conv2 = dglnn.SAGEConv(hidden_dim, out_dim, aggregator_type="mean")
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(graph, x)
        h = torch.relu(h)
        h = self.dropout(h)
        return self.conv2(graph, h)
