"""
train.py  —  Train one GNN variant on one (dataset, task) combination.

Usage:
  python train.py --dataset rel-stack --task user-engagement \
                  --arch sage --mode mpnn_u --num_layers 2 --seed 0

The script:
  1. Loads preprocessed .pt files from processed/{dataset}/{task}/
  2. Builds the model (MPNNModel or DirGNNModel)
  3. Trains with NeighborLoader + Adam + BCE loss
  4. Early stopping (patience=10) on val AUPRC
  5. Evaluates best checkpoint on val & test
  6. Saves model checkpoint to checkpoints/{dataset}/{task}/{arch}_{mode}_L{layers}_s{seed}.pt
  7. Appends one row to results/metrics.csv (file-locked)
"""

import argparse
import csv
import json
import time
from pathlib import Path

import filelock
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.loader import NeighborLoader

from models import build_model, count_parameters

ROOT            = Path(__file__).parent
PROCESSED       = ROOT / "processed"
RESULTS         = ROOT / "results"
CHECKPOINTS     = ROOT / "checkpoints"
RESULTS.mkdir(exist_ok=True)
METRICS_CSV     = RESULTS / "metrics.csv"

CSV_COLS = [
    "dataset", "task", "mode", "arch", "num_layers", "hidden_dim", "seed",
    "val_auc", "val_auprc", "val_precision", "val_recall",
    "test_auc", "test_auprc", "test_precision", "test_recall",
    "train_time_sec", "num_params",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VALID_COMBOS = {
    "rel-stack": ["user-engagement"],
    "rel-avito": ["user-visits"],
}


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    required=True)
    p.add_argument("--task",       required=True)
    p.add_argument("--mode",       required=True, choices=["mpnn_u", "mpnn_d", "dir_gnn"])
    p.add_argument("--arch",       required=True, choices=["sage", "gat"])
    p.add_argument("--num_layers", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience",   type=int, default=10)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--skip_if_exists", action="store_true",
                   help="Exit immediately if the checkpoint file already exists.")
    return p.parse_args()


def checkpoint_path(dataset, task, arch, mode, num_layers, seed) -> Path:
    return CHECKPOINTS / dataset / task / f"{arch}_{mode}_L{num_layers}_s{seed}.pt"


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_split(dataset: str, task: str, split: str):
    path = PROCESSED / dataset / task / f"{split}.pt"
    return torch.load(path, weights_only=False, map_location="cpu")


def load_meta(dataset: str, task: str) -> dict:
    with open(PROCESSED / dataset / task / "meta.json") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Neighbor loaders
# ──────────────────────────────────────────────────────────────────────────────

def make_loader(data, target_node: str, mask_attr: str, batch_size: int,
                num_neighbors: int, num_layers: int, shuffle: bool):
    mask = getattr(data[target_node], mask_attr)
    # Scale neighbors per hop so total fan-out stays constant across depths.
    # With many edge types (e.g. 22 for rel-stack), num_neighbors applies per
    # edge type per hop — without scaling, L=2 creates ~10×22×10×22 ≈ 48k
    # nodes per root and L=3 approaches millions, hanging the training loop.
    per_hop = max(2, num_neighbors // num_layers)
    num_n = {et: [per_hop] * num_layers for et in data.edge_types}
    return NeighborLoader(
        data,
        num_neighbors=num_n,
        input_nodes=(target_node, mask),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# One epoch
# ──────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, target_node, train=True):
    model.train(train)
    total_loss = 0.0
    total_n    = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(DEVICE)
            x_dict          = {nt: batch[nt].x for nt in batch.node_types}
            edge_index_dict = batch.edge_index_dict

            mask = batch[target_node].mask
            if not mask.any():
                continue

            y_hat = model(x_dict, edge_index_dict)[mask]
            y     = batch[target_node].y[mask].to(DEVICE)

            valid = ~torch.isnan(y)
            if not valid.any():
                continue
            y_hat, y = y_hat[valid], y[valid]

            loss = F.binary_cross_entropy(y_hat, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * valid.sum().item()
            total_n    += valid.sum().item()

    return total_loss / max(total_n, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, target_node):
    model.eval()
    all_preds  = []
    all_labels = []

    for batch in loader:
        batch = batch.to(DEVICE)
        x_dict          = {nt: batch[nt].x for nt in batch.node_types}
        edge_index_dict = batch.edge_index_dict

        mask = batch[target_node].mask
        if not mask.any():
            continue

        preds  = model(x_dict, edge_index_dict)[mask].cpu().numpy()
        labels = batch[target_node].y[mask].cpu().numpy()

        valid = ~np.isnan(labels)
        all_preds.append(preds[valid])
        all_labels.append(labels[valid])

    if not all_preds:
        return dict(auc=0.0, auprc=0.0, precision=0.0, recall=0.0)

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels).astype(int)
    binary = (preds >= 0.5).astype(int)

    return dict(
        auc       = float(roc_auc_score(labels, preds)),
        auprc     = float(average_precision_score(labels, preds)),
        precision = float(precision_score(labels, binary, zero_division=0)),
        recall    = float(recall_score(labels, binary, zero_division=0)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# CSV append (file-locked for concurrent jobs)
# ──────────────────────────────────────────────────────────────────────────────

def append_csv(row: dict):
    lock_path = str(METRICS_CSV) + ".lock"
    with filelock.FileLock(lock_path):
        write_header = not METRICS_CSV.exists()
        with open(METRICS_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLS)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in CSV_COLS})


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.task not in VALID_COMBOS.get(args.dataset, []):
        print(f"Skipping invalid combo: {args.dataset}/{args.task}")
        return

    ckpt = checkpoint_path(args.dataset, args.task, args.arch,
                            args.mode, args.num_layers, args.seed)
    if args.skip_if_exists and ckpt.exists():
        print(f"Checkpoint exists, skipping: {ckpt}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── load data ───────────────────────────────────────────────────────────
    meta        = load_meta(args.dataset, args.task)
    target_node = meta["target_node"]

    print(f"\nLoading graphs for {args.dataset}/{args.task} …")
    data_train = load_split(args.dataset, args.task, "train")
    data_val   = load_split(args.dataset, args.task, "val")
    data_test  = load_split(args.dataset, args.task, "test")

    # ── build model ─────────────────────────────────────────────────────────
    metadata = data_train.metadata()
    model = build_model(
        metadata         = metadata,
        target_node_type = target_node,
        arch             = args.arch,
        mode             = args.mode,
        num_layers       = args.num_layers,
        hidden_dim       = args.hidden_dim,
        dropout          = args.dropout,
    ).to(DEVICE)

    # Warm-up: one tiny batch to instantiate lazy parameters before counting
    _warm_loader = make_loader(data_train, target_node, "mask",
                               batch_size=32, num_neighbors=1,
                               num_layers=args.num_layers, shuffle=False)
    try:
        _batch = next(iter(_warm_loader)).to(DEVICE)
        with torch.no_grad():
            model({nt: _batch[nt].x for nt in _batch.node_types},
                  _batch.edge_index_dict)
    except StopIteration:
        pass
    del _warm_loader, _batch

    n_params = count_parameters(model)
    print(f"Model: {args.arch}/{args.mode} L={args.num_layers}  params={n_params:,}")

    # ── data loaders ────────────────────────────────────────────────────────
    train_loader = make_loader(data_train, target_node, "mask",
                               batch_size=args.batch_size,
                               num_neighbors=args.num_neighbors,
                               num_layers=args.num_layers, shuffle=True)
    val_loader   = make_loader(data_val, target_node, "mask",
                               batch_size=args.batch_size,
                               num_neighbors=args.num_neighbors,
                               num_layers=args.num_layers, shuffle=False)
    test_loader  = make_loader(data_test, target_node, "mask",
                               batch_size=args.batch_size,
                               num_neighbors=args.num_neighbors,
                               num_layers=args.num_layers, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)

    # ── training loop with early stopping ───────────────────────────────────
    best_val_auprc = -1.0
    best_state     = None
    patience_count = 0
    t0             = time.time()

    print(f"\nTraining on {DEVICE} …")
    for epoch in range(1, args.max_epochs + 1):
        train_loss  = run_epoch(model, train_loader, optimizer, target_node, train=True)
        val_loss    = run_epoch(model, val_loader,   optimizer, target_node, train=False)
        val_metrics = evaluate(model, val_loader, target_node)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_auprc={val_metrics['auprc']:.4f}")

        if val_metrics["auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["auprc"]
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"  Early stopping at epoch {epoch} (best val AUPRC={best_val_auprc:.4f})")
                break

    train_time = time.time() - t0

    # ── final evaluation ────────────────────────────────────────────────────
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    val_metrics  = evaluate(model, val_loader,  target_node)
    test_metrics = evaluate(model, test_loader, target_node)

    print(f"\nResults:")
    print(f"  val  auc={val_metrics['auc']:.4f}  auprc={val_metrics['auprc']:.4f}  "
          f"prec={val_metrics['precision']:.4f}  rec={val_metrics['recall']:.4f}")
    print(f"  test auc={test_metrics['auc']:.4f}  auprc={test_metrics['auprc']:.4f}  "
          f"prec={test_metrics['precision']:.4f}  rec={test_metrics['recall']:.4f}")
    print(f"  train_time={train_time:.1f}s  params={n_params:,}")

    row = dict(
        dataset    = args.dataset,
        task       = args.task,
        mode       = args.mode,
        arch       = args.arch,
        num_layers = args.num_layers,
        hidden_dim = args.hidden_dim,
        seed       = args.seed,
        **{f"val_{k}":  v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        train_time_sec = round(train_time, 1),
        num_params     = n_params,
    )

    # ── save checkpoint ─────────────────────────────────────────────────────
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metrics": row, "args": vars(args)}, ckpt)
    print(f"Checkpoint saved: {ckpt}")

    append_csv(row)
    print(f"Row appended to {METRICS_CSV}")


if __name__ == "__main__":
    main()
