# Aspect 3 — Node Features (Initial Node Representation)

How does the **initial node representation** affect a heterogeneous GNN? We fix an
**HGT** backbone and swap only the input encoder, comparing three strategies on
**(1) downstream performance, (2) model complexity (#params), (3) usability**.

| Strategy | Input | Encoder | Idea |
|----------|-------|---------|------|
| **id**     | `node.gid`   | `Embedding[num_nodes → hidden]` per type | No tuple content — a learnable id embedding per node (structure-only baseline). |
| **column** | `node.x`     | `Linear[feat_dim → hidden]` per type     | Column-wise: each cell encoded by its type (numeric / categorical / datetime). |
| **llm**    | `node.x_llm` | `Linear[384 → hidden]` per type          | Each tuple stringified `"col=val, …"` and embedded with a frozen sentence-transformer. |

**Runs** — 2 datasets × 3 feature modes × 2 seeds = **12** (HGT, L=2, hidden=64).

**Datasets**

- `rel-f1 / driver-dnf` — predict a driver DNF next race. Small, well-balanced; the
  full graph is used (no sampling), so the LLM pass covers every tuple.
- `rel-event / user-repeat` — predict a repeat attendance. Larger; non-target tables
  are capped to 60k rows (see `SAMPLE_CAPS` in `preprocess.py`). The **same sampled
  graph is reused for all three strategies** (assignment requirement).

---

## Fairness / isolating the variable

All three variants share the *identical* HGT backbone, `hidden_dim`, layer count,
heads, optimizer, loader, and preprocessed graph. **Only the input encoder differs.**
`x`, `gid`, and `x_llm` are all built in a single preprocessing pass and stored on
every node, so the graph and sample are guaranteed identical across strategies.

---

## Prerequisites

```bash
conda env create -f ../environment.yml   # now includes sentence-transformers
conda activate structml1
```

## Option A — Notebook

Open `notebook.ipynb` and run all cells. Set `RESULTS_ONLY = True` in the first cell
to skip preprocessing/training and analyze the committed `results/metrics.csv`.

## Option B — Scripts

```bash
python preprocess.py                    # builds x + gid + x_llm (downloads on first run)
python preprocess.py --skip_llm         # x + gid only (fast; disables feat_mode=llm)
python run_experiments.py               # all 12 runs (skips existing checkpoints)
python run_experiments.py --dry_run
python train.py --dataset rel-f1 --task driver-dnf --feat_mode column --seed 0
sbatch run_all.sh                       # SLURM
```

## Results

- `results/metrics.csv` — one row per run (includes `num_params` and `encoder_params`).
- `checkpoints/{dataset}/{task}/hgt_{feat_mode}_L{layers}_s{seed}.pt` — one per run.

## Notes

- `id` embeddings are keyed on a **stable global id** (`gid`, from the full db) so an
  entity has the same embedding in train and test. New entities appearing only at test
  time get an untrained embedding — an inherent limitation of the id strategy, and one
  of the points the discussion should make.
- `gid` is used (not `n_id`) because PyG's `NeighborLoader` writes `batch[nt].n_id`.
- First run: confirm the two dataset/task names resolve in your installed `relbench`
  version; both are in the standard RelBench task suite. Adjust `DATASET_TASKS` if needed.
