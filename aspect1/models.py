"""
models.py  —  GNN model classes for the structural directionality experiment.

Six model variants = 2 archs (sage, gat) × 3 modes (mpnn_u, mpnn_d, dir_gnn):

  MPNN-U   : SAGEConv/GATConv on all edges (forward FK→PK + reversed PK→FK).
             Wrapped with to_hetero; effectively undirected.
  MPNN-D   : Same but uses only FK→PK (forward) edges.
             to_hetero applied; messages flow child→parent only.
  Dir-GNN  : Two separate to_hetero sub-models per layer — one for forward
             edges, one for reversed edges — each with hidden_dim//2 output.
             Concatenated per-layer then projected back to hidden_dim.

All models:
  - hidden_dim = 64 (configurable)
  - Dropout = 0.3 between layers
  - ReLU between layers
  - Per-node-type lazy init via (-1,-1) in SAGEConv/GATConv handles varying
    raw feature dims across node types without explicit input projection layers.
  - Binary output: Linear(hidden_dim→1) + Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, to_hetero


# ──────────────────────────────────────────────────────────────────────────────
# Single-layer homogeneous building blocks (wrapped per-layer with to_hetero)
# ──────────────────────────────────────────────────────────────────────────────

class HomoSAGE1(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv = SAGEConv((-1, -1), hidden_dim)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class HomoGAT1(nn.Module):
    def __init__(self, out_per_head: int, heads: int = 2):
        super().__init__()
        self.conv = GATConv((-1, -1), out_per_head, heads=heads,
                            add_self_loops=False, concat=True)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


# ──────────────────────────────────────────────────────────────────────────────
# Dir-GNN layer
# ──────────────────────────────────────────────────────────────────────────────

class DirGNNLayer(nn.Module):
    """
    One layer of Dir-GNN:
      1. fwd_conv: to_hetero(SingleConv(dir_dim), fwd_meta) on FK→PK edges
      2. rev_conv: to_hetero(SingleConv(dir_dim), rev_meta) on PK→FK edges
      3. cat([fwd_out, rev_out], dim=-1) → [N, hidden_dim]
      4. per-node-type linear projection → [N, hidden_dim] + ReLU + dropout

    dir_dim = hidden_dim // 2  (SAGE)
             = hidden_dim // 4 per head × 2 heads = hidden_dim // 2  (GAT)
    """

    def __init__(
        self,
        node_types,
        fwd_edge_types,
        rev_edge_types,
        hidden_dim: int,
        arch: str,
        dropout: float,
    ):
        super().__init__()
        dir_dim = hidden_dim // 2

        if arch == "sage":
            fwd_homo = HomoSAGE1(dir_dim)
            rev_homo = HomoSAGE1(dir_dim)
        else:  # gat: hidden//4 per head × 2 heads = hidden//2 per direction
            per_head = hidden_dim // 4
            fwd_homo = HomoGAT1(per_head, heads=2)
            rev_homo = HomoGAT1(per_head, heads=2)

        # Only include node types that participate in each edge set.
        # to_hetero fails if a node type is in metadata but never a src or dst.
        fwd_nodes = list({n for e in fwd_edge_types for n in (e[0], e[2])})
        rev_nodes = list({n for e in rev_edge_types for n in (e[0], e[2])})
        self.fwd_conv = to_hetero(fwd_homo, (fwd_nodes, fwd_edge_types))
        self.rev_conv = to_hetero(rev_homo, (rev_nodes, rev_edge_types))

        # Per-node-type projection: cat(hidden//2, hidden//2) → hidden
        self.proj = nn.ModuleDict(
            {nt: nn.Linear(hidden_dim, hidden_dim) for nt in node_types}
        )
        self.dir_dim = dir_dim
        self.dropout = dropout

    def forward(self, x_dict, fwd_dict, rev_dict):
        fwd_out = self.fwd_conv(x_dict, fwd_dict)
        rev_out = self.rev_conv(x_dict, rev_dict)

        out = {}
        for nt, x in x_dict.items():
            n, dev = x.size(0), x.device
            zeros = torch.zeros(n, self.dir_dim, device=dev)
            # to_hetero stores None for node types that never appear as dst;
            # .get(nt, default) returns None (not default) when key exists with None value.
            f = fwd_out.get(nt); f = zeros if f is None else f
            r = rev_out.get(nt); r = zeros if r is None else r
            cat = torch.cat([f, r], dim=-1)
            out[nt] = F.relu(
                F.dropout(self.proj[nt](cat), p=self.dropout, training=self.training)
            )
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Top-level model classes
# ──────────────────────────────────────────────────────────────────────────────

class MPNNModel(nn.Module):
    """
    MPNN-U (mode='mpnn_u'): all edges (FK→PK + rev PK→FK), undirected.
    MPNN-D (mode='mpnn_d'): FK→PK edges only, directed child→parent.

    Per-layer to_hetero (same pattern as DirGNNLayer).  Wrapping the whole
    multi-layer stack in a single to_hetero fails when any node type is only
    a source (never a dst) in the active edge set — to_hetero requires every
    node type to be updated after every layer.  Per-layer avoids that constraint:
    nodes that receive no messages simply keep their previous representation.
    """

    def __init__(
        self,
        metadata,
        target_node_type: str,
        hidden_dim: int,
        num_layers: int,
        arch: str,
        mode: str,
        dropout: float,
    ):
        super().__init__()
        node_types, edge_types = metadata
        fwd_edge_types = [e for e in edge_types if not e[1].startswith("rev_")]
        active_edge_types = edge_types if mode == "mpnn_u" else fwd_edge_types
        active_nodes = list({n for e in active_edge_types for n in (e[0], e[2])})
        active_meta = (active_nodes, active_edge_types)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            if arch == "sage":
                homo = HomoSAGE1(hidden_dim)
            else:  # gat: hidden//2 per head × 2 heads = hidden_dim output
                homo = HomoGAT1(hidden_dim // 2, heads=2)
            self.convs.append(to_hetero(homo, active_meta))

        self.head = nn.Linear(hidden_dim, 1)
        self.target_node_type = target_node_type
        self.mode    = mode
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        if self.mode == "mpnn_d":
            edge_index_dict = {
                k: v for k, v in edge_index_dict.items()
                if not k[1].startswith("rev_")
            }

        for i, conv in enumerate(self.convs):
            new_x = conv(x_dict, edge_index_dict)
            last = (i == len(self.convs) - 1)
            # to_hetero sets None (not absent) for source-only nodes; check explicitly.
            x_dict = {
                k: (new_x[k] if last else
                    F.dropout(F.relu(new_x[k]), p=self.dropout, training=self.training))
                if new_x.get(k) is not None else v
                for k, v in x_dict.items()
            }

        out = x_dict[self.target_node_type]
        out = F.dropout(out, p=self.dropout, training=self.training)
        return torch.sigmoid(self.head(out)).squeeze(-1)


class DirGNNModel(nn.Module):
    """
    Dir-GNN: num_layers of DirGNNLayer (one fwd + one rev conv each layer),
    followed by a shared binary classification head.
    """

    def __init__(
        self,
        metadata,
        target_node_type: str,
        hidden_dim: int,
        num_layers: int,
        arch: str,
        dropout: float,
    ):
        super().__init__()
        node_types, edge_types = metadata
        fwd_edge_types = [e for e in edge_types if not e[1].startswith("rev_")]
        rev_edge_types = [e for e in edge_types if e[1].startswith("rev_")]

        self.layers = nn.ModuleList([
            DirGNNLayer(node_types, fwd_edge_types, rev_edge_types,
                        hidden_dim, arch, dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, 1)
        self.target_node_type = target_node_type
        self.dropout = dropout

    def forward(self, x_dict, edge_index_dict):
        fwd_dict = {k: v for k, v in edge_index_dict.items() if not k[1].startswith("rev_")}
        rev_dict = {k: v for k, v in edge_index_dict.items() if k[1].startswith("rev_")}

        for layer in self.layers:
            x_dict = layer(x_dict, fwd_dict, rev_dict)

        out = x_dict[self.target_node_type]
        out = F.dropout(out, p=self.dropout, training=self.training)
        return torch.sigmoid(self.head(out)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    metadata,
    target_node_type: str,
    arch: str,
    mode: str,
    num_layers: int,
    hidden_dim: int = 64,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Args:
        metadata:          (node_types, edge_types) from HeteroData.metadata()
        target_node_type:  node type to predict labels for
        arch:              'sage' or 'gat'
        mode:              'mpnn_u', 'mpnn_d', or 'dir_gnn'
        num_layers:        number of message-passing layers
        hidden_dim:        latent dimension (default 64)
        dropout:           dropout probability (default 0.3)
    """
    if mode in ("mpnn_u", "mpnn_d"):
        return MPNNModel(metadata, target_node_type, hidden_dim,
                         num_layers, arch, mode, dropout)
    else:
        return DirGNNModel(metadata, target_node_type, hidden_dim,
                           num_layers, arch, dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
