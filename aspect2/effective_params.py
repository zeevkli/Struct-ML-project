#!/home/zeev.kliot/miniconda3/envs/structml1/bin/python
"""
effective_params.py  —  Visualize effective vs nominal parameters in homo / hetero GNNs.

An encoder or conv weight is "effective" only if its corresponding node / edge type
lies close enough to the target node that its output can propagate to the target
within the available number of GNN layers.

  homo  : encoder for node type X is effective iff dist(X, target) <= num_layers
  hetero: weight matrix for edge (A,e,B) at GNN layer l is effective iff
            dist(B, target) <= num_layers - l   (remaining budget covers B→target)
            dist(A, target) <= num_layers - l + 1  (A can reach B, then target)

Run directly:  python aspect2/effective_params.py
"""

import json
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models import build_model   # noqa: E402 (aspect2/models.py)

# ── Config ────────────────────────────────────────────────────────────────────
TASKS = [
    ("rel-stack",  "user-engagement"),
    ("rel-avito",  "user-visits"),
    ("rel-arxiv",  "author-category"),
]
SETTINGS        = ["homo", "hetero"]
ARCHS           = ["sage", "gat", "hgt"]
HIDDEN_DIM      = 64
ALL_NUM_LAYERS  = [2, 3]   # show both configs
DROPOUT         = 0.0
PROCESSED       = ROOT / "processed"
OUT_DIR         = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Schema graph utils ─────────────────────────────────────────────────────────

def bfs_distance(target, edge_types, max_hops):
    """BFS on the undirected schema graph.  Returns {node_type: distance}."""
    dist  = {target: 0}
    queue = deque([target])
    adj   = {}
    for (src, e, dst) in edge_types:
        adj.setdefault(src, set()).add(dst)
        adj.setdefault(dst, set()).add(src)
    while queue:
        node = queue.popleft()
        d    = dist[node]
        if d >= max_hops:
            continue
        for nbr in adj.get(node, []):
            if nbr not in dist:
                dist[nbr] = d + 1
                queue.append(nbr)
    return dist


def effective_node_types(meta, num_layers):
    """Node types that can influence the target within num_layers hops."""
    et  = [tuple(e) for e in meta["edge_types"]]
    d   = bfs_distance(meta["target_node"], et, num_layers)
    return {nt: d[nt] for nt in meta["node_types"] if nt in d}


def active_edge_budget(meta, num_layers):
    """
    Returns {(src, e, dst): layers_active} — the set of GNN layers (1-indexed)
    at which each edge type is "active" (i.e., its weights contribute to prediction).

    Edge (A,e,B) is active at layer l iff:
      dist(B, target) <= num_layers - l     (B within remaining budget)
      dist(A, target) <= num_layers - l + 1 (A one step further)
    """
    et     = [tuple(e) for e in meta["edge_types"]]
    dist   = bfs_distance(meta["target_node"], et, num_layers + 1)
    budget = {}
    for edge in et:
        src, e, dst = edge
        active_layers = []
        for l in range(1, num_layers + 1):
            remaining = num_layers - l
            if (dist.get(dst, 9999) <= remaining and
                    dist.get(src, 9999) <= remaining + 1):
                active_layers.append(l)
        if active_layers:
            budget[edge] = active_layers
    return budget


# ── Dummy forward-pass data ────────────────────────────────────────────────────

def make_inputs(meta):
    """Minimal input dicts for a single forward pass (10 nodes, 5 edges each)."""
    x_dict = {nt: torch.zeros(10, fd)
               for nt, fd in meta["node_feat_dims"].items()}
    src_idx = torch.arange(5, dtype=torch.long)
    dst_idx = torch.arange(5, dtype=torch.long)
    ei      = torch.stack([src_idx, dst_idx])
    ei_dict = {tuple(e): ei for e in meta["edge_types"]}
    return x_dict, ei_dict


# ── Parameter counting ─────────────────────────────────────────────────────────

def _edge_key(src, e, dst):
    """PyG to_hetero uses double-underscore encoding for edge keys."""
    return f"{src}__{e}__{dst}"


def count_effective(meta, setting, arch, num_layers):
    node_types     = meta["node_types"]
    edge_types_raw = [tuple(e) for e in meta["edge_types"]]
    feat_dims      = meta["node_feat_dims"]
    target         = meta["target_node"]
    metadata       = (node_types, edge_types_raw)

    model = build_model(
        metadata=metadata, target_node_type=target, feat_dims=feat_dims,
        arch=arch, setting=setting, num_layers=num_layers,
        hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
    )

    x_dict, ei_dict = make_inputs(meta)
    try:
        with torch.no_grad():
            model(x_dict, ei_dict)
    except Exception:
        pass  # lazy init may still have fired

    named = {n: p.numel() for n, p in model.named_parameters()}

    eff_nodes    = effective_node_types(meta, num_layers)   # {nt: dist}
    edge_budget  = active_edge_budget(meta, num_layers)     # {edge: [layers]}
    n_total_et   = len(edge_types_raw)
    n_active_et  = len(edge_budget)
    active_frac  = n_active_et / max(n_total_et, 1)

    # Active edge keys for SAGEConv/GATConv (to_hetero naming: src__e__dst)
    active_edge_keys = {_edge_key(*e) for e in edge_budget}

    # Active node type keys for HGTConv node-specific submodules
    eff_node_set = set(eff_nodes.keys())

    totals = {"encoder_eff": 0, "encoder_wasted": 0,
              "conv_eff": 0,    "conv_wasted": 0,
              "head": 0}

    for pname, n in named.items():
        # Head
        if pname.startswith("head."):
            totals["head"] += n
            continue

        # Encoders (homo, hybrid, and HGT hetero)
        if "encoders." in pname:
            after = pname.split("encoders.")[1]
            nt    = after.split(".")[0]
            if nt in eff_node_set:
                totals["encoder_eff"] += n
            else:
                totals["encoder_wasted"] += n
            continue

        # Conv layers
        if "convs." in pname or "hetero_conv." in pname or "homo_convs." in pname:
            if setting == "homo":
                totals["conv_eff"] += n

            elif setting == "hetero":
                if arch in ("sage", "gat"):
                    # to_hetero naming: src__edge__dst
                    hit = any(key in pname for key in active_edge_keys)
                    if hit:
                        totals["conv_eff"] += n
                    else:
                        totals["conv_wasted"] += n
                else:
                    # HGT hetero: mixed per-node and per-edge params
                    # Per-node-type submodules: kqv_lin.lins.{nt}, out_lin.lins.{nt}, skip.{nt}
                    is_node_param = (
                        ".lins." in pname or
                        ".skip." in pname
                    )
                    if is_node_param:
                        # Extract node type name (after last '.')
                        parts = pname.split(".")
                        # skip.{nt} → parts[-1] is the nt
                        # lins.{nt}.weight → parts[-2] is the nt
                        nt = parts[-1] if "skip" in pname else parts[-2]
                        if nt in eff_node_set:
                            totals["conv_eff"] += n
                        else:
                            totals["conv_wasted"] += n
                    elif "p_rel." in pname:
                        # p_rel.{src}__{e}__{dst}: edge-type specific prior
                        hit = any(key in pname for key in active_edge_keys)
                        if hit:
                            totals["conv_eff"] += n
                        else:
                            totals["conv_wasted"] += n
                    else:
                        # k_rel, v_rel: bulk tensors indexed by edge type.
                        # Scale by fraction of active edge types.
                        eff_n   = round(n * active_frac)
                        wasted_n = n - eff_n
                        totals["conv_eff"]    += eff_n
                        totals["conv_wasted"] += wasted_n

            else:
                totals["conv_eff"] += n

    total     = sum(named.values())
    effective = totals["encoder_eff"] + totals["conv_eff"] + totals["head"]
    return total, effective, totals, eff_nodes, edge_budget


# ── Schema graph diagram ───────────────────────────────────────────────────────

def draw_schema(ax, meta, num_layers, title):
    """
    Draw the relational schema as a node-link diagram.
    Nodes are colored by hop distance from target (or grey if unreachable).
    """
    import math

    et      = [tuple(e) for e in meta["edge_types"]]
    target  = meta["target_node"]
    dist    = bfs_distance(target, et, num_layers + 2)
    nodes   = meta["node_types"]
    n       = len(nodes)

    # Circular layout
    angles  = {nd: 2 * math.pi * i / n for i, nd in enumerate(nodes)}
    pos     = {nd: (math.cos(a), math.sin(a)) for nd, a in angles.items()}

    # Color by distance
    cmap    = {0: "#2196F3", 1: "#4CAF50", 2: "#FF9800", 3: "#F44336", 99: "#BDBDBD"}
    palette = {nd: cmap.get(min(dist.get(nd, 99), 3), "#BDBDBD") for nd in nodes}
    alpha_f = {nd: 1.0 if dist.get(nd, 99) <= num_layers else 0.35 for nd in nodes}

    # Draw edges (fwd only to avoid double-drawing)
    fwd = {tuple(e) for e in meta.get("fwd_edge_types", [])}
    for (src, e, dst) in et:
        if (src, e, dst) not in fwd:
            continue
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        a = min(alpha_f[src], alpha_f[dst])
        ax.plot([x0, x1], [y0, y1], color="#9E9E9E", lw=0.8, alpha=a, zorder=1)

    # Draw nodes
    for nd in nodes:
        x, y  = pos[nd]
        a     = alpha_f[nd]
        ax.scatter(x, y, s=300, color=palette[nd], edgecolors="white",
                   linewidths=1.5, zorder=3, alpha=a)
        short = nd[:10] + ".." if len(nd) > 12 else nd
        d     = dist.get(nd, "∞")
        lbl   = f"{short}\n(d={d})" if d != "∞" else f"{short}\n(∞)"
        ax.text(x, y + 0.16, lbl, ha="center", va="bottom",
                fontsize=6, zorder=4, alpha=a)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.6)
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold")

    # Legend
    for hop, col, lbl in [(0, "#2196F3", "target (d=0)"),
                           (1, "#4CAF50", "1-hop"),
                           (2, "#FF9800", "2-hop"),
                           (3, "#F44336", "3-hop (unreachable@L=2)"),
                           (99, "#BDBDBD", "beyond")]:
        ax.scatter([], [], color=col, s=60, label=lbl)
    ax.legend(fontsize=5.5, loc="lower right", framealpha=0.7)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Gather data ────────────────────────────────────────────────────────────
    # results[num_layers][(ds, task)][setting][arch] = {...}
    results = {L: {} for L in ALL_NUM_LAYERS}

    for L in ALL_NUM_LAYERS:
        print(f"\n── L={L} ─────────────────────────────────────────────────────────")
        for ds, task in TASKS:
            meta_path = PROCESSED / ds / task / "meta.json"
            with open(meta_path) as f:
                meta = json.load(f)
            results[L][(ds, task)] = {}
            for setting in SETTINGS:
                results[L][(ds, task)][setting] = {}
                for arch in ARCHS:
                    total, eff, breakdown, eff_nodes, edge_budget = \
                        count_effective(meta, setting, arch, L)
                    results[L][(ds, task)][setting][arch] = {
                        "total": total, "effective": eff, "breakdown": breakdown,
                        "eff_nodes": eff_nodes, "edge_budget": edge_budget,
                    }
                    frac = eff / total * 100 if total else 0
                    print(f"  {ds}/{task:20s} {setting:8s} {arch:4s}  "
                          f"total={total:6,d}  eff={eff:6,d}  ({frac:.0f}%)")

    # Convenience alias: L=2 results for the single-depth figures
    res2 = results[2]

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 1 — Stacked bar: effective vs wasted params per task × arch
    # ═══════════════════════════════════════════════════════════════════════════

    task_labels = [f"{ds}\n{task}" for ds, task in TASKS]
    colors = {
        "encoder_eff":    "#4CAF50",
        "conv_eff":       "#2196F3",
        "head":           "#9C27B0",
        "encoder_wasted": "#FF9800",
        "conv_wasted":    "#F44336",
    }
    comp_order = ["encoder_eff", "conv_eff", "head", "encoder_wasted", "conv_wasted"]
    comp_labels = {
        "encoder_eff":    "Encoder (effective)",
        "conv_eff":       "Conv (effective)",
        "head":           "Head",
        "encoder_wasted": "Encoder (wasted)",
        "conv_wasted":    "Conv (wasted)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    fig.suptitle(
        "Effective vs. Nominal Parameters\n"
        "(Wasted = weights for nodes/edges unreachable within L=2 hops of target)",
        fontsize=13, fontweight="bold",
    )

    for row_i, setting in enumerate(SETTINGS):
        for col_i, (ds, task) in enumerate(TASKS):
            ax    = axes[row_i][col_i]
            rdata = res2[(ds, task)][setting]
            x     = np.arange(len(ARCHS))
            bar_w = 0.55

            for xi, arch in enumerate(ARCHS):
                bd = rdata[arch]["breakdown"]
                bot = 0
                for comp in comp_order:
                    val = bd.get(comp, 0)
                    if val > 0:
                        ax.bar(xi, val, bottom=bot, width=bar_w,
                               color=colors[comp], edgecolor="white", linewidth=0.3)
                        bot += val

                total = rdata[arch]["total"]
                eff   = rdata[arch]["effective"]
                frac  = eff / total * 100 if total else 0
                ax.text(xi, total + total * 0.01, f"{frac:.0f}%",
                        ha="center", va="bottom", fontsize=7.5, color="#333")

            ax.set_xticks(x)
            ax.set_xticklabels([a.upper() for a in ARCHS], fontsize=9)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v/1000)}k"))
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)

            if col_i == 0:
                ax.set_ylabel(f"{setting.upper()}\nParams", fontsize=9, fontweight="bold")
            if row_i == 0:
                ax.set_title(f"{ds} / {task}", fontsize=9)

    # Shared legend
    handles = [mpatches.Patch(color=colors[c], label=comp_labels[c])
               for c in comp_order]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out1 = OUT_DIR / "effective_params_stacked.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out1}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 2 — Effective % heatmap: setting × arch × task
    # ═══════════════════════════════════════════════════════════════════════════

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 3.5))
    fig2.suptitle("% Parameters That Are Effective (L=2 hops)", fontsize=12, fontweight="bold")

    for ax2, setting in zip(axes2, SETTINGS):
        mat = np.zeros((len(ARCHS), len(TASKS)))
        for ci, (ds, task) in enumerate(TASKS):
            for ri, arch in enumerate(ARCHS):
                d = res2[(ds, task)][setting][arch]
                mat[ri, ci] = d["effective"] / d["total"] * 100

        im = ax2.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax2.set_xticks(range(len(TASKS)))
        ax2.set_xticklabels([f"{ds}\n{t}" for ds, t in TASKS], fontsize=8)
        ax2.set_yticks(range(len(ARCHS)))
        ax2.set_yticklabels([a.upper() for a in ARCHS], fontsize=9)
        ax2.set_title(setting.upper(), fontsize=10, fontweight="bold")

        for ri in range(len(ARCHS)):
            for ci in range(len(TASKS)):
                ax2.text(ci, ri, f"{mat[ri, ci]:.0f}%",
                         ha="center", va="center", fontsize=10,
                         color="white" if mat[ri, ci] < 50 else "black",
                         fontweight="bold")

        plt.colorbar(im, ax=ax2, shrink=0.8, label="% effective")

    plt.tight_layout()
    out2 = OUT_DIR / "effective_params_heatmap.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 3 — Schema graphs colored by hop distance (one per task)
    # ═══════════════════════════════════════════════════════════════════════════

    fig3, axes3 = plt.subplots(1, 3, figsize=(17, 6))
    fig3.suptitle("Relational Schema: Node Reachability within L=2 GNN Layers",
                  fontsize=12, fontweight="bold")

    for ax3, (ds, task) in zip(axes3, TASKS):
        meta_path = PROCESSED / ds / task / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        draw_schema(ax3, meta, 2, f"{ds} / {task}")

    plt.tight_layout()
    out3 = OUT_DIR / "schema_reachability.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 4 — Effective param count: homo vs hetero at fixed L, colored by
    #            "how much is wasted" (as wasted fraction bubble or scatter)
    # ═══════════════════════════════════════════════════════════════════════════

    fig4, axes4 = plt.subplots(1, len(TASKS), figsize=(14, 5))
    fig4.suptitle("Total vs. Effective Parameter Count (L=2, hidden=64)",
                  fontsize=12, fontweight="bold")

    for ax4, (ds, task) in zip(axes4, TASKS):
        for arch_i, arch in enumerate(ARCHS):
            for set_i, setting in enumerate(SETTINGS):
                d      = res2[(ds, task)][setting][arch]
                total  = d["total"]
                eff    = d["effective"]
                wasted = total - eff
                frac_w = wasted / total if total else 0

                marker = {"sage": "o", "gat": "s", "hgt": "^"}[arch]
                color  = {"homo": "#2196F3", "hetero": "#FF5722"}[setting]

                ax4.scatter(total, eff, s=120 + frac_w * 400,
                            c=color, marker=marker, alpha=0.85,
                            edgecolors="white", linewidths=1.2, zorder=3)
                ax4.annotate(f"{arch.upper()}\n{setting[:4]}",
                             (total, eff), fontsize=5.5,
                             ha="left", va="bottom",
                             xytext=(4, 2), textcoords="offset points")

        # Diagonal = 100% effective reference
        lim = ax4.get_xlim()
        lo  = min(lim)
        hi  = max(lim)
        ax4.plot([lo, hi], [lo, hi], "--", color="#9E9E9E", lw=1, label="100% effective")
        ax4.set_xlabel("Total params", fontsize=9)
        ax4.set_ylabel("Effective params", fontsize=9)
        ax4.set_title(f"{ds} / {task}", fontsize=9, fontweight="bold")
        ax4.grid(alpha=0.25)
        ax4.spines[["top", "right"]].set_visible(False)

    # Legend
    arch_handles = [
        plt.scatter([], [], marker="o", color="#777", s=70, label="SAGE"),
        plt.scatter([], [], marker="s", color="#777", s=70, label="GAT"),
        plt.scatter([], [], marker="^", color="#777", s=70, label="HGT"),
    ]
    set_handles = [
        mpatches.Patch(color="#2196F3", label="homo"),
        mpatches.Patch(color="#FF5722", label="hetero"),
    ]
    fig4.legend(handles=arch_handles + set_handles,
                loc="lower center", ncol=5, fontsize=8,
                framealpha=0.9, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    out4 = OUT_DIR / "effective_params_scatter.png"
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out4}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 5 — L=2 vs L=3: how depth changes the effective parameter fraction
    # ═══════════════════════════════════════════════════════════════════════════

    fig5, axes5 = plt.subplots(len(SETTINGS), len(TASKS),
                               figsize=(5 * len(TASKS), 4 * len(SETTINGS)),
                               sharey=False)
    fig5.suptitle("Effective Parameter % by GNN Depth  (L=2 vs L=3)\n"
                  "Deeper models reach more node types → fewer wasted parameters",
                  fontsize=12, fontweight="bold")

    layer_colors = {2: "#2196F3", 3: "#FF9800"}

    for row_i, setting in enumerate(SETTINGS):
        for col_i, (ds, task) in enumerate(TASKS):
            ax5   = axes5[row_i][col_i]
            x5    = np.arange(len(ARCHS))
            width5 = 0.32

            for li, L in enumerate(ALL_NUM_LAYERS):
                fracs = []
                for arch in ARCHS:
                    d = results[L][(ds, task)][setting][arch]
                    fracs.append(d["effective"] / d["total"] * 100 if d["total"] else 0)
                bars5 = ax5.bar(x5 + (li - 0.5) * width5, fracs, width5,
                                label=f"L={L}", color=layer_colors[L], alpha=0.85)
                for bar5, v5 in zip(bars5, fracs):
                    ax5.text(bar5.get_x() + bar5.get_width() / 2,
                             bar5.get_height() + 0.5,
                             f"{v5:.0f}%", ha="center", va="bottom", fontsize=8)

            ax5.set_xticks(x5)
            ax5.set_xticklabels([a.upper() for a in ARCHS])
            ax5.set_ylim(0, 115)
            ax5.set_ylabel("% Effective parameters")
            ax5.axhline(100, color="black", lw=0.7, linestyle="--", alpha=0.4)
            ax5.spines[["top", "right"]].set_visible(False)
            ax5.grid(axis="y", alpha=0.25)

            if row_i == 0:
                ax5.set_title(f"{ds} / {task}", fontsize=9)
            if col_i == 0:
                ax5.set_ylabel(f"{setting.upper()}\n% Effective", fontsize=9, fontweight="bold")
            if row_i == 0 and col_i == 0:
                ax5.legend(fontsize=9)

    plt.tight_layout()
    out5 = OUT_DIR / "effective_params_depth_comparison.png"
    plt.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out5}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figure 6 — Test AUPRC vs Effective Parameter Count
    #             Same layout as the Parameter Efficiency scatter but x-axis =
    #             effective params (total × reachable fraction), not total params.
    # ═══════════════════════════════════════════════════════════════════════════

    import pandas as pd
    from matplotlib.lines import Line2D

    metrics_path = ROOT / "results" / "metrics.csv"
    if metrics_path.exists():
        SETTINGS_ALL = ["homo", "hetero", "homo_noenc", "hybrid"]

        # Pre-compute effective fractions for every (dataset, task, setting, arch, L).
        # The fraction is hidden_dim-invariant: only schema structure matters.
        eff_frac_map = {}
        for L in ALL_NUM_LAYERS:
            for ds, task in TASKS:
                meta_path = PROCESSED / ds / task / "meta.json"
                with open(meta_path) as f:
                    meta_s = json.load(f)
                for setting in SETTINGS_ALL:
                    for arch in ARCHS:
                        tot_s, eff_s, *_ = count_effective(meta_s, setting, arch, L)
                        eff_frac_map[(ds, task, setting, arch, L)] = (
                            eff_s / tot_s if tot_s else 1.0
                        )

        df = pd.read_csv(metrics_path)
        df = df.drop_duplicates(
            subset=["dataset", "task", "setting", "arch", "num_layers", "hidden_dim", "seed"],
            keep="last",
        ).reset_index(drop=True)

        df["eff_frac"] = df.apply(
            lambda r: eff_frac_map.get(
                (r["dataset"], r["task"], r["setting"], r["arch"], r["num_layers"]), 1.0
            ),
            axis=1,
        )
        df["eff_params"] = (df["num_params"] * df["eff_frac"]).round().astype(int)

        setting_colors = {
            "homo":       "#4C72B0",
            "hetero":     "#DD8452",
            "homo_noenc": "#55A868",
            "hybrid":     "#C44E52",
        }
        arch_markers = {"sage": "o", "gat": "^", "hgt": "s"}

        fig6, axes6 = plt.subplots(1, len(TASKS), figsize=(6 * len(TASKS), 6), sharey=False)
        fig6.suptitle(
            "Parameter Efficiency: Test AUPRC vs Effective Parameter Count\n"
            "(effective = params whose node/edge types lie within GNN reach of target)",
            fontsize=12, fontweight="bold",
        )

        for ax6, (ds, task) in zip(axes6, TASKS):
            sub = df[(df["dataset"] == ds) & (df["task"] == task)]
            for setting in SETTINGS_ALL:
                for arch in ARCHS:
                    pts = sub[(sub["setting"] == setting) & (sub["arch"] == arch)]
                    if pts.empty:
                        continue
                    ax6.scatter(
                        pts["eff_params"] / 1000,
                        pts["test_auprc"],
                        c=setting_colors[setting],
                        marker=arch_markers[arch],
                        s=100, alpha=0.85,
                        edgecolors="white", linewidths=0.8, zorder=3,
                    )
                    for _, row in pts.iterrows():
                        ax6.annotate(
                            f"h={int(row['hidden_dim'])}",
                            (row["eff_params"] / 1000, row["test_auprc"]),
                            textcoords="offset points", xytext=(4, 2),
                            fontsize=7, alpha=0.85,
                        )

            ax6.set_xlabel("Effective Parameter Count (k)", fontsize=11)
            ax6.set_ylabel("Test AUPRC", fontsize=11)
            ax6.set_title(f"{ds}\n({task})", fontsize=11, fontweight="bold")
            ax6.spines[["top", "right"]].set_visible(False)
            ax6.grid(alpha=0.2)

        legend_elements = [
            mpatches.Patch(color=c, label=s.replace("_", " ").title())
            for s, c in setting_colors.items()
            if s in df["setting"].values
        ] + [
            Line2D([0], [0], marker=arch_markers[a], color="w",
                   markerfacecolor="#555", markersize=9, label=a.upper())
            for a in ARCHS
        ]
        axes6[-1].legend(handles=legend_elements, fontsize=9,
                         title="Setting / Arch", loc="lower right", framealpha=0.8)

        plt.tight_layout()
        out6 = OUT_DIR / "auprc_vs_effective_params.png"
        plt.savefig(out6, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out6}")
    else:
        print("No metrics.csv found — skipping AUPRC vs effective params figure.")

    print("\nAll figures saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
