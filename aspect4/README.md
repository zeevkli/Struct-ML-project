# Aspect 4 — Limitations of Deeper Models (Oversmoothing)

Deep message-passing GNNs suffer **oversmoothing**: as layers stack, node
representations converge and lose their distinctiveness. We measure this directly and
test whether a mitigation restores depth-robustness.

- **Model:** GraphSAGE with **mean aggregation** (per-type linear encoder → `hidden`,
  then L × `SAGEConv(mean)`, lifted to the heterogeneous graph via `to_hetero`).
- **Depth sweep:** L ∈ {1, 2, 4, 8, 16}.
- **Oversmoothing measures** (on target-node representations, cf. tutorial 7):
  - **MAD** (Mean Average Distance = mean pairwise cosine distance) — → 0 when oversmoothed.
  - **Mean pairwise cosine similarity** — → 1 when oversmoothed.
- **Downstream performance:** ROC AUC, AUPRC, precision, recall vs depth.
- **Mitigation (follow-up):** compare the plain stack against
  - **`residual`** — skip connections `h ← h + SAGE(h)`, and
  - **`dropedge`** — randomly drop a fraction of edges each training step.

**Runs** — 2 datasets × 5 depths × 3 mitigations × 1 seed = **30** (hidden=64).

Datasets are distinct from the other aspects: `rel-f1/driver-dnf` (small, shallow graph)
and `rel-trial/study-outcome` (medium, clinical-trials domain — unused elsewhere).
Oversmoothing is a depth phenomenon orthogonal to the dataset; `DATASET_TASKS` in
`preprocess.py` is one line to change.

---

## Fairness / isolating the variable

Every run shares the identical encoder + head, `hidden_dim`, optimizer, loader, and
preprocessed graph. Only **depth** and **mitigation** change. The uniform `hidden_dim`
across all layers is what lets residual connections add layer inputs to outputs and
keeps the oversmoothing metrics comparable across depths.

---

## Prerequisites

```bash
conda env create -f ../environment.yml
conda activate structml1
```

## Option A — Notebook

Open `notebook.ipynb` and run all cells. Set `RESULTS_ONLY = True` in the first cell to
skip preprocessing/training and analyze the committed `results/metrics.csv`.

## Option B — Scripts

```bash
python preprocess.py
python run_experiments.py                 # 30 runs (skips existing checkpoints)
python run_experiments.py --dry_run
python train.py --dataset rel-f1 --task driver-dnf --num_layers 8 --mitigation none
python train.py --dataset rel-f1 --task driver-dnf --num_layers 8 --mitigation residual
sbatch run_all.sh
```

## Results

- `results/metrics.csv` — one row per run, carrying both downstream metrics and the
  oversmoothing measures (`mad`, `cos_sim`) for that (depth, mitigation).
- `checkpoints/{dataset}/{task}/sage_{mitigation}_L{layers}_s{seed}.pt` — one per run.

## Notes

- Oversmoothing metrics are computed on the SEED target nodes of the eval loader (the
  first `batch_size` rows), capped at 1000 nodes for the O(N²) pairwise similarity, with
  the same cap across depths so the numbers are comparable.
- Expectation: for `mitigation=none`, `cos_sim → 1` / `MAD → 0` and downstream metrics
  degrade as depth grows; `residual` / `dropedge` should flatten both trends.
