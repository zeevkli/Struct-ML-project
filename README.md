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
├── aspect1/             ← GNN directionality experiment (MPNN-U / MPNN-D / Dir-GNN)
│   ├── README.md        ← how to run aspect1
│   ├── notebook.ipynb   ← main entry point
│   ├── preprocess.py
│   ├── models.py
│   ├── train.py
│   ├── run_experiments.py
│   └── run_all.sh
└── aspect3/             ← node-features experiment (ID / column-wise / LLM, HGT)
    ├── README.md        ← how to run aspect3
    ├── notebook.ipynb   ← main entry point
    ├── preprocess.py
    ├── models.py
    ├── train.py
    ├── run_experiments.py
    └── run_all.sh
```

Each aspect is a **self-contained folder** (its own preprocessing, models, training,
runner, and notebook), so aspects can be developed and run independently.
