# Aspect 2 — Homogeneous vs Heterogeneous GNNs

**54 runs** — 3 datasets × 3 archs × 4 settings × 2 depths × seed 0  
**+18 ablation runs** — homo at h=128 to match hetero parameter count

---

## Prerequisites

```bash
conda env create -f ../environment.yml
conda activate structml1
```

---

## Settings

| Setting | Description | Key question |
|---|---|---|
| `homo` | Per-type Linear encoders → shared GNN | Baseline |
| `homo_noenc` | Zero-pad raw features → shared GNN (no encoders) | Do per-type encoders drive homo performance? |
| `hybrid` | Per-type encoders → hetero L1 → shared GNN L2+ | Does type-specificity only matter at the input? |
| `hetero` | Type-specific GNN weights throughout | Full type-aware baseline |

**Parameter-matching ablation:** homo at h=128 ≈ hetero at h=64 in param count.  
Answers: *is hetero better because of type awareness, or just more parameters?*

---

## Datasets

| Dataset | Task | Node types | Metric |
|---|---|---|---|
| **rel-stack** | user-engagement | 7 | AUPRC |
| **rel-avito** | user-visits | 8 | AUPRC |
| **rel-arxiv** | author-category | 6 | Macro-F1 (53 classes) |

---

## Option A — Notebook

Open `notebook.ipynb` and run all cells.

---

## Option B — Scripts

```bash
# Preprocess all datasets
python preprocess.py

# Run all experiments (skips existing checkpoints)
python run_experiments.py

# Run only new ablation settings
python run_experiments.py --main

# Run parameter-matching ablation only
python run_experiments.py --ablation

# Dry run — show what would run
python run_experiments.py --dry_run

# On SLURM
sbatch run_all.sh
```

---

## Results

`results/metrics.csv` — one row per run  
`checkpoints/{dataset}/{task}/{arch}_{setting}_h{hidden_dim}_s{seed}.pt`
