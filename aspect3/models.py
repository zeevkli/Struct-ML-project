"""
models.py  —  HGT model for the Node-Features ablation (Aspect 3).

A single HGT backbone is shared across the three feature strategies; only the
INPUT ENCODER that produces the initial per-node hidden vector changes. This
isolates the variable under study (initial node representation) — same backbone,
same hidden_dim, same #layers/heads, so the only differences are (a) downstream
performance, (b) parameter count (which the encoder dominates), and (c) usability.

  feat_mode = "id"      IdEncoder      — Embedding[num_global_nt, hidden] per type,
                                          indexed by node.gid. Ignores tuple content.
  feat_mode = "column"  ColumnEncoder  — Linear[feat_dim_nt → hidden] per type,
                                          on node.x (column-wise encoded features).
  feat_mode = "llm"     LlmEncoder     — Linear[llm_dim → hidden] per type,
                                          on node.x_llm (sentence-transformer vecs).

Backbone: L × HGTConv(hidden, hidden, metadata, heads) with ReLU + dropout between
layers, then a shared Linear(hidden → 1) + Sigmoid binary head on the target type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


# ──────────────────────────────────────────────────────────────────────────────
# Input encoders  (one per feature strategy)
# ──────────────────────────────────────────────────────────────────────────────

class IdEncoder(nn.Module):
    """No features: a learnable embedding per node, keyed on the stable global id.

    Parameter count scales with the number of nodes (sum of table sizes) — this is
    the whole point of the `id` baseline and is reported as the model-complexity axis.
    """

    def __init__(self, node_types, num_global: dict, hidden_dim: int):
        super().__init__()
        self.emb = nn.ModuleDict({
            nt: nn.Embedding(max(int(num_global.get(nt, 1)), 1), hidden_dim)
            for nt in node_types
        })

    def forward(self, gid_dict, x_dict, x_llm_dict):
        out = {}
        for nt, emb in self.emb.items():
            if nt in gid_dict:
                # clamp guards against any id ≥ table size (e.g. unseen at build time);
                # non-in-place so the loader's batch tensor is not mutated.
                idx = gid_dict[nt].clamp(0, emb.num_embeddings - 1)
                out[nt] = emb(idx)
        return out


class ColumnEncoder(nn.Module):
    """Column-wise features → hidden, via a per-table linear encoder (ModuleDict of
    Linear(table_in_dim → hidden), exactly the homogenizing encoder from the spec)."""

    def __init__(self, node_types, feat_dims: dict, hidden_dim: int):
        super().__init__()
        self.lin = nn.ModuleDict({
            nt: nn.Linear(int(feat_dims.get(nt, 1)), hidden_dim) for nt in node_types
        })

    def forward(self, gid_dict, x_dict, x_llm_dict):
        return {nt: lin(x_dict[nt]) for nt, lin in self.lin.items() if nt in x_dict}


class LlmEncoder(nn.Module):
    """Frozen sentence-transformer vectors → hidden, via a per-table linear layer.
    llm_dim is identical across tables, so parameter count is small and uniform."""

    def __init__(self, node_types, llm_dim: int, hidden_dim: int):
        super().__init__()
        self.lin = nn.ModuleDict({
            nt: nn.Linear(llm_dim, hidden_dim) for nt in node_types
        })

    def forward(self, gid_dict, x_dict, x_llm_dict):
        return {nt: lin(x_llm_dict[nt]) for nt, lin in self.lin.items() if nt in x_llm_dict}


def build_encoder(feat_mode, node_types, feat_dims, num_global, llm_dim, hidden_dim):
    if feat_mode == "id":
        return IdEncoder(node_types, num_global, hidden_dim)
    if feat_mode == "column":
        return ColumnEncoder(node_types, feat_dims, hidden_dim)
    if feat_mode == "llm":
        return LlmEncoder(node_types, llm_dim, hidden_dim)
    raise ValueError(f"unknown feat_mode: {feat_mode}")


# ──────────────────────────────────────────────────────────────────────────────
# HGT model
# ──────────────────────────────────────────────────────────────────────────────

class HGTModel(nn.Module):
    def __init__(
        self,
        metadata,                 # (node_types, edge_types) from HeteroData.metadata()
        target_node_type: str,
        feat_mode: str,
        feat_dims: dict,
        num_global: dict,
        llm_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.target_node_type = target_node_type
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.feat_mode = feat_mode

        self.encoder = build_encoder(
            feat_mode, self.node_types, feat_dims, num_global, llm_dim, hidden_dim
        )
        self.convs = nn.ModuleList([
            HGTConv(hidden_dim, hidden_dim, metadata, heads=heads)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 1)

    def _complete_x(self, x_dict, device):
        """HGTConv is built from the (fixed) train metadata, but a mini-batch may not
        contain every node type. Pad the missing ones with empty [0, hidden] tensors
        so HGTConv's per-type parameters all receive an input."""
        for nt in self.node_types:
            if nt not in x_dict or x_dict[nt] is None:
                x_dict[nt] = torch.zeros(0, self.hidden_dim, device=device)
        return x_dict

    def _complete_edges(self, edge_index_dict, device):
        for et in self.edge_types:
            et = tuple(et)
            if et not in edge_index_dict:
                edge_index_dict[et] = torch.empty(2, 0, dtype=torch.long, device=device)
        return edge_index_dict

    def forward(self, gid_dict, x_dict, x_llm_dict, edge_index_dict):
        device = self.head.weight.device
        h = self.encoder(gid_dict, x_dict, x_llm_dict)
        h = self._complete_x(h, device)
        edge_index_dict = self._complete_edges(dict(edge_index_dict), device)

        for i, conv in enumerate(self.convs):
            new_h = conv(h, edge_index_dict)
            last = (i == len(self.convs) - 1)
            # HGTConv may omit source-only types; keep their previous representation.
            h = {
                nt: (new_h[nt] if last else
                     F.dropout(F.relu(new_h[nt]), p=self.dropout, training=self.training))
                if new_h.get(nt) is not None else prev
                for nt, prev in h.items()
            }

        out = h[self.target_node_type]
        out = F.dropout(out, p=self.dropout, training=self.training)
        return torch.sigmoid(self.head(out)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Factory / utils
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    metadata,
    target_node_type: str,
    feat_mode: str,
    feat_dims: dict,
    num_global: dict,
    llm_dim: int,
    num_layers: int,
    hidden_dim: int = 64,
    heads: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    return HGTModel(
        metadata=metadata, target_node_type=target_node_type, feat_mode=feat_mode,
        feat_dims=feat_dims, num_global=num_global, llm_dim=llm_dim,
        hidden_dim=hidden_dim, num_layers=num_layers, heads=heads, dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_encoder_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
