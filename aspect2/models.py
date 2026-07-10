"""
models.py  —  GNN models for the homogeneous vs heterogeneous experiment.

Four settings × three architectures = twelve model variants:

  homo       : per-type Linear encoders → collapse → shared GNN weights
  homo_noenc : zero-pad raw features to max_feat_dim → shared GNN (no per-type encoders)
  hybrid     : per-type encoders → hetero first layer → shared GNN for remaining layers
  hetero     : type-specific GNN weights throughout

Homo baseline follows the HGB (KDD 2021) standard: per-type input projections to bring
all node types to the same hidden_dim, then shared-weight message passing.

Homo_noenc ablation: removes per-type encoders to isolate whether they (not the shared GNN)
drive homo performance.

Hybrid ablation: hetero only for the first message-passing layer, then collapses to homo.
Tests whether type-specificity is only needed at the input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HGTConv, SAGEConv, to_hetero


# ──────────────────────────────────────────────────────────────────────────────
# Single-layer building blocks for to_hetero wrapping
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
# Shared helper: collapse hetero graph to homogeneous
# ──────────────────────────────────────────────────────────────────────────────

def _build_homo_graph(node_types, x_dict, edge_index_dict, pad_to=None):
    """
    Concatenate all node feature tensors into one and remap edge indices to
    global offsets.  If pad_to is given, raw features are zero-padded (or
    truncated) to that width before concatenation.

    Returns: x_homo, edge_index, offset_map, ordered, sizes
    """
    ordered = [nt for nt in node_types if nt in x_dict]
    sizes   = [x_dict[nt].size(0) for nt in ordered]
    cumsum  = [0]
    for s in sizes:
        cumsum.append(cumsum[-1] + s)
    offset_map = {nt: cumsum[i] for i, nt in enumerate(ordered)}

    if pad_to is not None:
        tensors = []
        for nt in ordered:
            x = x_dict[nt]
            if x.size(1) < pad_to:
                pad = torch.zeros(x.size(0), pad_to - x.size(1),
                                  device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            elif x.size(1) > pad_to:
                x = x[:, :pad_to]
            tensors.append(x)
        x_homo = torch.cat(tensors, dim=0)
    else:
        x_homo = torch.cat([x_dict[nt] for nt in ordered], dim=0)

    edge_list = []
    for et, ei in edge_index_dict.items():
        src_type, _, dst_type = et
        if src_type in offset_map and dst_type in offset_map:
            shifted = ei + torch.tensor(
                [[offset_map[src_type]], [offset_map[dst_type]]],
                dtype=torch.long, device=ei.device,
            )
            edge_list.append(shifted)
    edge_index = (torch.cat(edge_list, dim=1) if edge_list
                  else torch.zeros(2, 0, dtype=torch.long, device=x_homo.device))

    return x_homo, edge_index, offset_map, ordered, sizes


def _run_homo_convs(convs, arch, x_homo, edge_index, dropout, training):
    """Run a list of homogeneous conv layers with ReLU + dropout between them."""
    if arch == "hgt":
        for i, conv in enumerate(convs):
            out = conv({"node": x_homo},
                       {("node", "edge", "node"): edge_index})["node"]
            x_homo = (F.dropout(F.relu(out), p=dropout, training=training)
                      if i < len(convs) - 1 else out)
    else:
        for i, conv in enumerate(convs):
            out = conv(x_homo, edge_index)
            x_homo = (F.dropout(F.relu(out), p=dropout, training=training)
                      if i < len(convs) - 1 else out)
    return x_homo


# ──────────────────────────────────────────────────────────────────────────────
# HomoModel  (standard homo baseline — HGB KDD 2021)
# ──────────────────────────────────────────────────────────────────────────────

class HomoModel(nn.Module):
    """
    Per-type nn.Linear encoders project all node features to hidden_dim, then
    all nodes are concatenated into a single tensor and processed by a
    shared-weight GNN (SAGEConv/GATConv) or by HGTConv with a single node/edge
    type (which degenerates to standard attention).
    """

    def __init__(self, metadata, target_node_type, feat_dims,
                 hidden_dim, num_layers, arch, dropout, num_classes=1):
        super().__init__()
        node_types, _ = metadata
        self.target_node_type = target_node_type
        self.hidden_dim = hidden_dim
        self.arch = arch
        self.dropout = dropout
        self.node_types = node_types
        self.num_classes = num_classes

        self.encoders = nn.ModuleDict({
            nt: nn.Linear(feat_dims[nt], hidden_dim) for nt in node_types
        })

        single_meta = (["node"], [("node", "edge", "node")])
        self.convs = nn.ModuleList()
        if arch == "sage":
            for _ in range(num_layers):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        elif arch == "gat":
            for _ in range(num_layers):
                self.convs.append(GATConv(hidden_dim, hidden_dim // 2, heads=2,
                                          add_self_loops=False, concat=True))
        else:  # hgt
            for _ in range(num_layers):
                self.convs.append(HGTConv(hidden_dim, hidden_dim, single_meta, heads=4))

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        enc_dict = {
            nt: F.relu(self.encoders[nt](x))
            for nt, x in x_dict.items() if nt in self.encoders
        }
        x_homo, edge_index, offset_map, ordered, sizes = _build_homo_graph(
            self.node_types, enc_dict, edge_index_dict
        )
        x_homo = _run_homo_convs(self.convs, self.arch, x_homo, edge_index,
                                  self.dropout, self.training)
        t_off  = offset_map[self.target_node_type]
        t_size = sizes[ordered.index(self.target_node_type)]
        out = F.dropout(x_homo[t_off:t_off + t_size], p=self.dropout, training=self.training)
        logits = self.head(out)
        if self.num_classes == 1:
            return torch.sigmoid(logits).squeeze(-1)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# HomoModelNoEnc  (ablation: shared GNN with no per-type input encoders)
# ──────────────────────────────────────────────────────────────────────────────

class HomoModelNoEnc(nn.Module):
    """
    Homo WITHOUT per-type encoders.

    Raw node features are zero-padded to max_feat_dim and fed directly into a
    shared GNN (no type-specific projection).  Isolates whether the per-type
    Linear encoders in HomoModel — not the shared message-passing weights — are
    the primary driver of homo performance.
    """

    def __init__(self, metadata, target_node_type, feat_dims,
                 hidden_dim, num_layers, arch, dropout, num_classes=1):
        super().__init__()
        node_types, _ = metadata
        self.target_node_type = target_node_type
        self.hidden_dim = hidden_dim
        self.arch = arch
        self.dropout = dropout
        self.node_types = node_types
        self.max_feat_dim = max(feat_dims.values())
        self.num_classes = num_classes

        single_meta = (["node"], [("node", "edge", "node")])
        self.convs = nn.ModuleList()
        in_dim = self.max_feat_dim
        if arch == "sage":
            for _ in range(num_layers):
                self.convs.append(SAGEConv(in_dim, hidden_dim))
                in_dim = hidden_dim
        elif arch == "gat":
            for _ in range(num_layers):
                self.convs.append(GATConv(in_dim, hidden_dim // 2, heads=2,
                                          add_self_loops=False, concat=True))
                in_dim = hidden_dim
        else:  # hgt
            for _ in range(num_layers):
                self.convs.append(HGTConv(in_dim, hidden_dim, single_meta, heads=4))
                in_dim = hidden_dim

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        x_homo, edge_index, offset_map, ordered, sizes = _build_homo_graph(
            self.node_types, x_dict, edge_index_dict, pad_to=self.max_feat_dim
        )
        x_homo = _run_homo_convs(self.convs, self.arch, x_homo, edge_index,
                                  self.dropout, self.training)
        t_off  = offset_map[self.target_node_type]
        t_size = sizes[ordered.index(self.target_node_type)]
        out = F.dropout(x_homo[t_off:t_off + t_size], p=self.dropout, training=self.training)
        logits = self.head(out)
        if self.num_classes == 1:
            return torch.sigmoid(logits).squeeze(-1)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# HybridModel  (ablation: hetero first layer → homo remaining layers)
# ──────────────────────────────────────────────────────────────────────────────

class HybridModel(nn.Module):
    """
    Layer 1: heterogeneous message passing (type-specific via to_hetero / HGTConv).
    Layers 2+: homogeneous message passing (shared GNN on collapsed graph).

    Per-type encoders project raw features to hidden_dim before the hetero layer,
    ensuring all nodes have the same dimension when collapsed for homo layers.

    Tests whether type-specificity needs to persist through all layers or only
    matters at the first hop (closest to the raw input features).
    """

    def __init__(self, metadata, target_node_type, feat_dims,
                 hidden_dim, num_layers, arch, dropout, num_classes=1):
        super().__init__()
        node_types, _ = metadata
        self.target_node_type = target_node_type
        self.hidden_dim = hidden_dim
        self.arch = arch
        self.dropout = dropout
        self.node_types = node_types
        self.num_classes = num_classes

        # Per-type encoders (same as HomoModel — give uniform dim before hetero layer)
        self.encoders = nn.ModuleDict({
            nt: nn.Linear(feat_dims[nt], hidden_dim) for nt in node_types
        })

        # Layer 1: type-specific
        if arch == "sage":
            self.hetero_conv = to_hetero(HomoSAGE1(hidden_dim), metadata)
        elif arch == "gat":
            self.hetero_conv = to_hetero(HomoGAT1(hidden_dim // 2, heads=2), metadata)
        else:  # hgt
            self.hetero_conv = HGTConv(hidden_dim, hidden_dim, metadata, heads=4)

        # Layers 2+: shared (homo)
        single_meta = (["node"], [("node", "edge", "node")])
        self.homo_convs = nn.ModuleList()
        for _ in range(num_layers - 1):
            if arch == "sage":
                self.homo_convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif arch == "gat":
                self.homo_convs.append(GATConv(hidden_dim, hidden_dim // 2, heads=2,
                                               add_self_loops=False, concat=True))
            else:
                self.homo_convs.append(HGTConv(hidden_dim, hidden_dim, single_meta, heads=4))

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        # Encode all types to hidden_dim
        x_dict = {
            nt: F.relu(self.encoders[nt](x))
            for nt, x in x_dict.items() if nt in self.encoders
        }

        # Layer 1: hetero
        new_x = self.hetero_conv(x_dict, edge_index_dict)
        x_dict = {
            k: (F.dropout(F.relu(new_x[k]), p=self.dropout, training=self.training)
                if new_x.get(k) is not None else v)
            for k, v in x_dict.items()
        }

        if not self.homo_convs:
            out = F.dropout(x_dict[self.target_node_type], p=self.dropout, training=self.training)
            logits = self.head(out)
            if self.num_classes == 1:
                return torch.sigmoid(logits).squeeze(-1)
            return logits

        # Collapse to homo and run remaining layers
        x_homo, edge_index, offset_map, ordered, sizes = _build_homo_graph(
            self.node_types, x_dict, edge_index_dict
        )
        x_homo = _run_homo_convs(self.homo_convs, self.arch, x_homo, edge_index,
                                  self.dropout, self.training)
        t_off  = offset_map[self.target_node_type]
        t_size = sizes[ordered.index(self.target_node_type)]
        out = F.dropout(x_homo[t_off:t_off + t_size], p=self.dropout, training=self.training)
        logits = self.head(out)
        if self.num_classes == 1:
            return torch.sigmoid(logits).squeeze(-1)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# HeteroModel
# ──────────────────────────────────────────────────────────────────────────────

class HeteroModel(nn.Module):
    """
    SAGE/GAT: per-layer to_hetero(conv, metadata) — separate weight matrix for
              each (src_type, edge_type, dst_type) triple.  Lazy init (-1,-1)
              handles varying raw feature dims without explicit input projections.

    HGT:      per-type Linear encoders → HGTConv per layer.  HGTConv requires
              fixed input dims (no lazy support), so we project first.
    """

    def __init__(self, metadata, target_node_type, feat_dims,
                 hidden_dim, num_layers, arch, dropout, num_classes=1):
        super().__init__()
        node_types, _ = metadata
        self.target_node_type = target_node_type
        self.arch = arch
        self.dropout = dropout
        self.num_classes = num_classes

        if arch in ("sage", "gat"):
            self.encoders = None
            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                if arch == "sage":
                    homo = HomoSAGE1(hidden_dim)
                else:
                    homo = HomoGAT1(hidden_dim // 2, heads=2)
                self.convs.append(to_hetero(homo, metadata))
        else:  # hgt
            self.encoders = nn.ModuleDict({
                nt: nn.Linear(feat_dims[nt], hidden_dim) for nt in node_types
            })
            self.convs = nn.ModuleList([
                HGTConv(hidden_dim, hidden_dim, metadata, heads=4)
                for _ in range(num_layers)
            ])

        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        if self.arch == "hgt":
            x_dict = {
                nt: F.relu(self.encoders[nt](x)) if nt in self.encoders else x
                for nt, x in x_dict.items()
            }
            for i, conv in enumerate(self.convs):
                new_x = conv(x_dict, edge_index_dict)
                last = (i == len(self.convs) - 1)
                x_dict = {
                    nt: new_x[nt] if last
                    else F.dropout(F.relu(new_x[nt]), p=self.dropout, training=self.training)
                    for nt in x_dict
                }
        else:
            for i, conv in enumerate(self.convs):
                new_x = conv(x_dict, edge_index_dict)
                last = (i == len(self.convs) - 1)
                x_dict = {
                    k: (new_x[k] if last else
                        F.dropout(F.relu(new_x[k]), p=self.dropout, training=self.training))
                    if new_x.get(k) is not None else v
                    for k, v in x_dict.items()
                }

        out = x_dict[self.target_node_type]
        out = F.dropout(out, p=self.dropout, training=self.training)
        logits = self.head(out)
        if self.num_classes == 1:
            return torch.sigmoid(logits).squeeze(-1)
        return logits


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_model(metadata, target_node_type, feat_dims, arch, setting,
                num_layers, hidden_dim=64, dropout=0.3, num_classes=1):
    kwargs = dict(metadata=metadata, target_node_type=target_node_type,
                  feat_dims=feat_dims, hidden_dim=hidden_dim,
                  num_layers=num_layers, arch=arch, dropout=dropout,
                  num_classes=num_classes)
    if setting == "homo":
        return HomoModel(**kwargs)
    elif setting == "homo_noenc":
        return HomoModelNoEnc(**kwargs)
    elif setting == "hybrid":
        return HybridModel(**kwargs)
    elif setting == "hetero":
        return HeteroModel(**kwargs)
    else:
        raise ValueError(f"Unknown setting: {setting!r}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
