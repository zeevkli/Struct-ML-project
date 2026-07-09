"""
models.py  —  GraphSAGE (mean aggregation) for the Oversmoothing ablation (Aspect 4).

A per-type linear encoder first maps every node type to a common `hidden_dim`; this
keeps the width fixed across depth so (a) residual/skip connections can add layer
inputs to outputs, and (b) the oversmoothing metrics compare like-for-like reps.

Then L × SAGEConv(hidden → hidden, aggr='mean'), each lifted to the heterogeneous
graph with a per-layer `to_hetero`. Two mitigation switches:

  mitigation = "none"      plain deep stack (expected to oversmooth as L grows)
  mitigation = "residual"  h ← h + SAGE(h)   (skip connection)
  mitigation = "dropedge"  randomly drop a fraction of edges each training step

`forward_embeddings` returns the pre-head target-node representations so the training
script can measure oversmoothing (MAD / cosine similarity) at any depth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero


def drop_edges(edge_index_dict, p: float):
    """DropEdge: keep each edge with prob (1-p), independently per edge type."""
    if p <= 0.0:
        return edge_index_dict
    out = {}
    for et, ei in edge_index_dict.items():
        if ei.size(1) == 0:
            out[et] = ei
            continue
        keep = torch.rand(ei.size(1), device=ei.device) >= p
        out[et] = ei[:, keep]
    return out


class HomoSAGE(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = SAGEConv(hidden_dim, hidden_dim, aggr="mean")

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class SAGEDepthModel(nn.Module):
    def __init__(self, metadata, target_node_type, feat_dims, hidden_dim,
                 num_layers, mitigation="none", dropout=0.3, dropedge_p=0.2):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.target_node_type = target_node_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mitigation = mitigation
        self.dropout = dropout
        self.dropedge_p = dropedge_p

        # per-type input encoder → uniform hidden width
        self.encoder = nn.ModuleDict({
            nt: nn.Linear(int(feat_dims.get(nt, 1)), hidden_dim) for nt in self.node_types
        })

        active_nodes = list({n for e in self.edge_types for n in (e[0], e[2])})
        active_meta = (active_nodes, [tuple(e) for e in self.edge_types])
        self.convs = nn.ModuleList([
            to_hetero(HomoSAGE(hidden_dim), active_meta) for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 1)

    def _encode(self, x_dict):
        return {nt: self.encoder[nt](x) for nt, x in x_dict.items() if nt in self.encoder}

    def _propagate(self, h, edge_index_dict):
        """Run the L conv layers, returning the final pre-head representation dict."""
        if self.mitigation == "dropedge" and self.training:
            edge_index_dict = drop_edges(edge_index_dict, self.dropedge_p)

        for i, conv in enumerate(self.convs):
            out = conv(h, edge_index_dict)
            new_h = {}
            last = (i == self.num_layers - 1)
            for nt, prev in h.items():
                o = out.get(nt)
                if o is None:                       # source-only type this layer
                    new_h[nt] = prev
                    continue
                if not last:
                    o = F.relu(o)
                if self.mitigation == "residual":   # skip connection
                    o = o + prev
                if not last:
                    o = F.dropout(o, p=self.dropout, training=self.training)
                new_h[nt] = o
            h = new_h
        return h

    def forward(self, x_dict, edge_index_dict):
        h = self._propagate(self._encode(x_dict), edge_index_dict)
        out = F.dropout(h[self.target_node_type], p=self.dropout, training=self.training)
        return torch.sigmoid(self.head(out)).squeeze(-1)

    @torch.no_grad()
    def forward_embeddings(self, x_dict, edge_index_dict):
        """Pre-head representation of the target node type (for oversmoothing metrics)."""
        was_training = self.training
        self.eval()
        h = self._propagate(self._encode(x_dict), edge_index_dict)
        if was_training:
            self.train()
        return h[self.target_node_type]


def build_model(metadata, target_node_type, feat_dims, num_layers,
                mitigation="none", hidden_dim=64, dropout=0.3, dropedge_p=0.2):
    return SAGEDepthModel(metadata, target_node_type, feat_dims, hidden_dim,
                          num_layers, mitigation, dropout, dropedge_p)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
