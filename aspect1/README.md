# Aspect 1 — GNN Directionality on Relational Data

**72 runs** — 4 datasets × 2 archs × 3 modes × 3 layers × seed 0

---

## Prerequisites

```bash
conda env create -f ../environment.yml
conda activate structml1
```

---

## Option A — Notebook

Open `notebook.ipynb` and run all cells. Preprocessing and training run automatically;
completed checkpoints are skipped.

---

## Option B — Scripts

```bash
# Preprocess (downloads datasets on first run)
python preprocess.py

# Run all 72 experiments (skips existing checkpoints)
python run_experiments.py

# Preview what would run without executing
python run_experiments.py --dry_run

# Run a single config
python train.py --dataset rel-stack --task user-engagement \
                --arch sage --mode mpnn_u --num_layers 2 --seed 0

# On SLURM
sbatch run_all.sh
```

---

## Datasets

| Dataset / Task | Target | Type | Expected |
|---|---|---|---|
| rel-stack / user-engagement | PK (users) | binary | MPNN-D or Dir-GNN — signal flows FK→PK |
| rel-avito / user-visits | PK (users) | binary | MPNN-D or Dir-GNN |
| rel-stack / post-votes | FK (posts) | binary | MPNN-U or Dir-GNN — useful signal flows PK→FK |
| rel-arxiv / author-category | authors | multiclass (53) | Dir-GNN — directed citation edges carry distinct signals |

The post-votes task reverses the FK/PK target, providing a direct test that the directionality effect is not an artifact of dataset choice.
rel-arxiv/author-category adds a naturally directed citation graph with 53-class output (ArXiv subject categories).

---

## Results

`results/metrics.csv` — one row per run  
`checkpoints/{dataset}/{task}/{arch}_{mode}_L{layers}_s{seed}.pt` — one checkpoint per run
