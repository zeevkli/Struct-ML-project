"""
train.py  —  Train one GraphSAGE variant at a given depth / mitigation (Aspect 4).

Usage:
  python train.py --dataset rel-f1 --task driver-dnf --num_layers 8 --mitigation none

After training it also measures oversmoothing (MAD + mean cosine similarity) on the
target-node representations, so each row in results/metrics.csv carries both the
downstream metrics and the oversmoothing metrics for that (depth, mitigation).
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
    average_precision_score, precision_score, recall_score, roc_auc_score,
)
from torch_geometric.loader import NeighborLoader

from models import build_model, count_parameters
from oversmoothing import oversmoothing_metrics

ROOT        = Path(__file__).parent
PROCESSED   = ROOT / "processed"
RESULTS     = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS.mkdir(exist_ok=True)
METRICS_CSV = RESULTS / "metrics.csv"

CSV_COLS = [
    "dataset", "task", "model", "mitigation", "num_layers", "hidden_dim", "seed",
    "val_auc", "val_auprc", "val_precision", "val_recall",
    "test_auc", "test_auprc", "test_precision", "test_recall",
    "mad", "cos_sim", "train_time_sec", "num_params",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OVERSMOOTH_MAX_NODES = 1000   # cap for the O(N^2) pairwise similarity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    required=True)
    p.add_argument("--task",       required=True)
    p.add_argument("--num_layers", type=int, required=True)
    p.add_argument("--mitigation", default="none", choices=["none", "residual", "dropedge"])
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience",   type=int, default=10)
    p.add_argument("--num_neighbors", type=int, default=10)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--dropedge_p", type=float, default=0.2)
    p.add_argument("--skip_if_exists", action="store_true")
    return p.parse_args()


def checkpoint_path(dataset, task, mitigation, num_layers, seed):
    return CHECKPOINTS / dataset / task / f"sage_{mitigation}_L{num_layers}_s{seed}.pt"


def load_split(dataset, task, split):
    return torch.load(PROCESSED / dataset / task / f"{split}.pt", weights_only=False, map_location="cpu")


def load_meta(dataset, task):
    return json.load(open(PROCESSED / dataset / task / "meta.json"))


def make_loader(data, target_node, batch_size, num_neighbors, num_layers, shuffle):
    mask = data[target_node].mask
    per_hop = max(2, num_neighbors // max(num_layers, 1))
    num_n = {et: [per_hop] * num_layers for et in data.edge_types}
    return NeighborLoader(data, num_neighbors=num_n, input_nodes=(target_node, mask),
                          batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(model, loader, optimizer, target_node, train=True):
    model.train(train)
    total_loss, total_n = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(DEVICE)
            x_dict = {nt: batch[nt].x for nt in batch.node_types}
            mask = batch[target_node].mask
            if not mask.any():
                continue
            y_hat = model(x_dict, batch.edge_index_dict)[mask]
            y = batch[target_node].y[mask].to(DEVICE)
            valid = ~torch.isnan(y)
            if not valid.any():
                continue
            y_hat, y = y_hat[valid], y[valid]
            loss = F.binary_cross_entropy(y_hat, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * valid.sum().item(); total_n += valid.sum().item()
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, target_node):
    model.eval()
    preds_all, labels_all = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        x_dict = {nt: batch[nt].x for nt in batch.node_types}
        mask = batch[target_node].mask
        if not mask.any():
            continue
        p = model(x_dict, batch.edge_index_dict)[mask].cpu().numpy()
        l = batch[target_node].y[mask].cpu().numpy()
        v = ~np.isnan(l)
        preds_all.append(p[v]); labels_all.append(l[v])
    if not preds_all:
        return dict(auc=0.0, auprc=0.0, precision=0.0, recall=0.0)
    preds = np.concatenate(preds_all); labels = np.concatenate(labels_all).astype(int)
    binary = (preds >= 0.5).astype(int)
    both = labels.min() != labels.max()
    return dict(
        auc=float(roc_auc_score(labels, preds)) if both else 0.0,
        auprc=float(average_precision_score(labels, preds)) if both else float(labels.mean()),
        precision=float(precision_score(labels, binary, zero_division=0)),
        recall=float(recall_score(labels, binary, zero_division=0)),
    )


@torch.no_grad()
def measure_oversmoothing(model, loader, target_node, cap=OVERSMOOTH_MAX_NODES):
    """Collect final-layer representations of SEED target nodes across batches, then
    compute MAD / cosine similarity on up to `cap` of them."""
    model.eval()
    chunks, collected = [], 0
    for batch in loader:
        batch = batch.to(DEVICE)
        x_dict = {nt: batch[nt].x for nt in batch.node_types}
        emb = model.forward_embeddings(x_dict, batch.edge_index_dict)
        bs = int(getattr(batch[target_node], "batch_size", emb.size(0)))
        chunks.append(emb[:bs].cpu())          # seed nodes are the first bs rows
        collected += bs
        if collected >= cap:
            break
    if not chunks:
        return dict(mad=0.0, cos_sim=1.0)
    H = torch.cat(chunks, dim=0)
    return oversmoothing_metrics(H, max_nodes=cap)


def append_csv(row):
    with filelock.FileLock(str(METRICS_CSV) + ".lock"):
        write_header = not METRICS_CSV.exists()
        with open(METRICS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in CSV_COLS})


def main():
    args = parse_args()
    ckpt = checkpoint_path(args.dataset, args.task, args.mitigation, args.num_layers, args.seed)
    if args.skip_if_exists and ckpt.exists():
        print(f"Checkpoint exists, skipping: {ckpt}")
        return

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    meta = load_meta(args.dataset, args.task)
    target_node = meta["target_node"]

    print(f"\nLoading graphs for {args.dataset}/{args.task} …")
    data_train = load_split(args.dataset, args.task, "train")
    data_val   = load_split(args.dataset, args.task, "val")
    data_test  = load_split(args.dataset, args.task, "test")

    model = build_model(
        metadata=data_train.metadata(), target_node_type=target_node,
        feat_dims=meta["node_feat_dims"], num_layers=args.num_layers,
        mitigation=args.mitigation, hidden_dim=args.hidden_dim,
        dropout=args.dropout, dropedge_p=args.dropedge_p,
    ).to(DEVICE)
    n_params = count_parameters(model)
    print(f"Model: sage/{args.mitigation} L={args.num_layers}  params={n_params:,}")

    tl = make_loader(data_train, target_node, args.batch_size, args.num_neighbors, args.num_layers, True)
    vl = make_loader(data_val,   target_node, args.batch_size, args.num_neighbors, args.num_layers, False)
    tsl = make_loader(data_test, target_node, args.batch_size, args.num_neighbors, args.num_layers, False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_auprc, best_state, patience = -1.0, None, 0
    t0 = time.time()
    print(f"\nTraining on {DEVICE} …")
    for epoch in range(1, args.max_epochs + 1):
        tr = run_epoch(model, tl, optimizer, target_node, True)
        run_epoch(model, vl, optimizer, target_node, False)
        vm = evaluate(model, vl, target_node)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | train_loss={tr:.4f}  val_auprc={vm['auprc']:.4f}")
        if vm["auprc"] > best_auprc:
            best_auprc = vm["auprc"]; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Early stopping at epoch {epoch} (best val AUPRC={best_auprc:.4f})")
                break
    train_time = time.time() - t0

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    vm = evaluate(model, vl, target_node)
    tm = evaluate(model, tsl, target_node)
    osm = measure_oversmoothing(model, vl, target_node)

    print(f"\nResults:")
    print(f"  val  auprc={vm['auprc']:.4f} auc={vm['auc']:.4f}   test auprc={tm['auprc']:.4f} auc={tm['auc']:.4f}")
    print(f"  oversmoothing: MAD={osm['mad']:.4f}  cos_sim={osm['cos_sim']:.4f}")
    print(f"  train_time={train_time:.1f}s  params={n_params:,}")

    row = dict(
        dataset=args.dataset, task=args.task, model="sage", mitigation=args.mitigation,
        num_layers=args.num_layers, hidden_dim=args.hidden_dim, seed=args.seed,
        **{f"val_{k}": v for k, v in vm.items()},
        **{f"test_{k}": v for k, v in tm.items()},
        mad=osm["mad"], cos_sim=osm["cos_sim"],
        train_time_sec=round(train_time, 1), num_params=n_params,
    )
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metrics": row, "args": vars(args)}, ckpt)
    print(f"Checkpoint saved: {ckpt}")
    append_csv(row)
    print(f"Row appended to {METRICS_CSV}")


if __name__ == "__main__":
    main()
