"""
oversmoothing.py  —  Similarity measures for quantifying oversmoothing.

Given a matrix of node representations H ∈ R[N, d], oversmoothing is the collapse of
rows toward each other. We report two standard, complementary measures (cf. tutorial 7):

  MAD (Mean Average Distance) — mean pairwise cosine *distance* (1 - cos). HIGH = diverse
      representations; → 0 means every node looks alike (oversmoothed).
  mean pairwise cosine similarity — the dual view; → 1 means oversmoothed.

Both are computed on a capped random sample of nodes (pairwise cost is O(N²)); the same
sample size is used across depths so the numbers are comparable.
"""

import numpy as np
import torch


@torch.no_grad()
def _sample_rows(H: torch.Tensor, max_nodes: int, seed: int = 0) -> torch.Tensor:
    n = H.size(0)
    if n <= max_nodes:
        return H
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:max_nodes]
    return H[idx]


@torch.no_grad()
def oversmoothing_metrics(H: torch.Tensor, max_nodes: int = 1000, seed: int = 0) -> dict:
    """Return {'mad', 'cos_sim'} for representation matrix H [N, d]."""
    H = _sample_rows(H.detach().float().cpu(), max_nodes, seed)
    if H.size(0) < 2:
        return dict(mad=0.0, cos_sim=1.0)

    Hn = torch.nn.functional.normalize(H, p=2, dim=1, eps=1e-8)
    sim = Hn @ Hn.t()                       # [n, n] cosine similarities
    n = sim.size(0)
    off = ~torch.eye(n, dtype=torch.bool)   # exclude self-pairs
    cos_sim = float(sim[off].mean())
    mad     = float((1.0 - sim)[off].mean())  # mean average (cosine) distance
    return dict(mad=mad, cos_sim=cos_sim)
