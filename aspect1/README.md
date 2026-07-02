# Aspect 1 — GNN Directionality on Relational Data

**36 runs** — 2 datasets × 2 archs × 3 modes × 3 layers × seed 0

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

# Run all 36 experiments (skips existing checkpoints)
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

## Results

`results/metrics.csv` — one row per run  
`checkpoints/{dataset}/{task}/{arch}_{mode}_L{layers}_s{seed}.pt` — one file per run
