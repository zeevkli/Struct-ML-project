# Struct-ML Project

Experiments on **heterogeneous relational GNNs** using [RelBench](https://relbench.stanford.edu/) datasets.

---

## Environment Setup

```bash
conda env create -f environment.yml
conda activate structml1
```

If `torch-scatter` / `torch-sparse` fail to install:
```bash
pip install torch-scatter torch-sparse \
    -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
```

> Targets PyTorch 2.4.1 + CUDA 12.1. Replace `+cu121` in `environment.yml` for other CUDA versions.

---

## Project Structure

```
structML_project/
├── environment.yml      ← shared conda environment
├── aspect1/             ← GNN directionality experiment
│   └── README.md
├── aspect2/             ← Homo vs Hetero GNN experiment
│   └── README.md
├── aspect3/             ← Node features experiment
│   └── README.md
└── aspect4/             ← Oversmoothing experiment
    └── README.md
```

Each aspect is a **self-contained folder** (its own preprocessing, models, training,
runner, and notebook), so aspects can be developed and run independently.

---

## Aspect 1 — Message Direction

**Question:** Does message-passing direction affect node classification on relational graphs?

Three modes tested: MPNN-U (undirected), MPNN-D (FK→PK only), Dir-GNN (bidirectional, separate params).
Across two architectures (SAGE, GAT), three depths, and four tasks spanning three RelBench datasets — including rel-arxiv/author-category (multiclass, 53 ArXiv categories).

See [aspect1/README.md](aspect1/README.md) for full details.

---

## Aspect 2 — Homogeneous vs Heterogeneous GNNs

**Question:** Does preserving node/edge type information during message passing help, or can a simpler homogeneous GNN close the gap?

Four settings compared across three architectures and three RelBench datasets:

| Setting | Description |
|---|---|
| **Homo** | Per-type input encoders + shared GNN weights |
| **Homo (no enc)** | Zero-padded raw features + shared GNN (no per-type encoders) |
| **Hybrid** | Per-type encoders + hetero first layer + shared GNN for remaining layers |
| **Hetero** | Type-specific GNN weights (to_hetero / HGTConv) |

Datasets: rel-stack/user-engagement, rel-avito/user-visits, rel-arxiv/author-category (multiclass, 53 classes).

See [aspect2/README.md](aspect2/README.md) for full details.

---

## Aspect 3 — Node Features (Initial Node Representation)

**Question:** How does the initial node representation affect a heterogeneous GNN?

Three input strategies compared with a fixed HGT backbone:

| Strategy | Description |
|---|---|
| **id** | Learnable embedding per node (structure-only baseline) |
| **column** | Column-wise linear encoder over raw features |
| **llm** | Frozen sentence-transformer embedding of stringified tuples |

See [aspect3/README.md](aspect3/README.md) for full details.

---

## Aspect 4 — Oversmoothing

**Question:** Do deeper GNNs oversmooth, and does a mitigation restore depth-robustness?

Depth sweep (L ∈ {1,2,4,8,16}) with MAD / cosine similarity as oversmoothing measures, plus residual connections and DropEdge as mitigations.

See [aspect4/README.md](aspect4/README.md) for full details.
