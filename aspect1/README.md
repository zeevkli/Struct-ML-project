# Aspect 1 — GNN Directionality on Relational Data

Tests whether the **direction of message passing** (FK→PK vs PK→FK) affects
node classification performance on heterogeneous relational graphs.

---

## Experiment Design

| Axis | Values |
|---|---|
| Dataset / Task | rel-stack / user-engagement · rel-avito / user-visits |
| Architecture | GraphSAGE · GAT |
| Mode | MPNN-U (undirected) · MPNN-D (FK→PK only) · Dir-GNN (both, separate params) |
| Layers | 1 · 2 · 3 |
| Seeds | 0 · 1 · 2 |
| **Total runs** | **108** |

**Hypothesis:** Both tasks are user-level — the target node is a parent in the
FK hierarchy and signal flows naturally up via FK→PK edges. The comparison
across modes (MPNN-U, MPNN-D, Dir-GNN) measures whether reverse edges add
signal in user-level relational prediction across two distinct domains.

---

## Step 0 — Prerequisites

Install the environment (once, from the project root):
```bash
conda env create -f ../environment.yml
conda activate structml1
wandb login
```

---

## Step 1 — Preprocessing

Builds `train.pt`, `val.pt`, `test.pt`, and `meta.json` for all three tasks.
Downloads datasets from relbench on first run (~a few GB, cached automatically).

```bash
cd aspect1/
python preprocess.py
```

To preprocess a single combination:
```bash
python preprocess.py --dataset rel-amazon --task user-churn
```

Output layout:
```
processed/
  rel-stack/user-engagement/   train.pt  val.pt  test.pt  meta.json
  rel-avito/user-visits/       train.pt  val.pt  test.pt  meta.json
```

---

## Step 2 — Training via wandb Sweep (recommended)

### 2a. Create the sweep

```bash
cd aspect1/
wandb sweep sweep.yaml
# Output: "wandb: Created sweep with ID: <SWEEP_ID>"
export WANDB_SWEEP_ID=<SWEEP_ID>
```

### 2b. Launch via SLURM (162 parallel agents)

```bash
mkdir -p logs
sbatch run_sweep.sh
```

Each SLURM job runs one `wandb agent --count 1`, which picks exactly one
configuration from the sweep and runs it to completion. Results are streamed
live to wandb and also appended to `results/metrics.csv`.

> **SLURM partition:** Edit `#SBATCH --partition=gpu` in `run_sweep.sh` to
> match your cluster's GPU partition name.

### 2c. Run a single agent locally (for debugging)

```bash
export WANDB_SWEEP_ID=<SWEEP_ID>
wandb agent --count 1 $WANDB_SWEEP_ID
```

---

## Step 2 (alternative) — Direct CLI

No sweep required. Runs one configuration directly:

```bash
python train.py \
  --dataset rel-stack --task user-engagement \
  --arch sage --mode mpnn_u --num_layers 2 --seed 0
```

Add `--no_wandb` to skip wandb logging.

---

## Results

All 162 runs append to `results/metrics.csv` (file-locked for concurrent SLURM
jobs). Columns:

```
dataset, task, mode, arch, num_layers, hidden_dim, seed,
val_auc, val_auprc, val_precision, val_recall,
test_auc, test_auprc, test_precision, test_recall,
train_time_sec, num_params
```

---

## Key Implementation Notes

- **Edges:** Both FK→PK and reversed PK→FK edges stored in every `.pt` file
  (reversed edges are prefixed `rev_`). Each model mode selects which to use.
- **MPNN-U / MPNN-D:** Homogeneous SAGEConv / GATConv stack wrapped with
  `to_hetero(model, data.metadata())`.
- **Dir-GNN:** Per layer — two separate `to_hetero` single-conv models (one
  forward, one reversed). Outputs are concatenated then projected back to
  `hidden_dim`.
- **Feature encoding:** Fitted on train split only (no leakage). Datetime
  columns → integer days offset from split cutoff, clipped to ≥ 0.
- **Early stopping:** Patience 10 on val AUPRC; max 100 epochs.
