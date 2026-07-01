#!/home/zeev.kliot/miniconda3/envs/structml1/bin/python
"""
preprocess.py  —  Build HeteroData objects for each (dataset, task, split).

Output layout:
  processed/{dataset}/{task}/{split}.pt
  processed/{dataset}/{task}/meta.json

Run:
  python preprocess.py
  python preprocess.py --dataset rel-stack --task user-engagement  # single combo
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch_geometric.data import HeteroData

from relbench.datasets import get_dataset
from relbench.tasks import get_task

ROOT = Path(__file__).parent
PROCESSED = ROOT / "processed"

# Two tasks chosen to test GNN directionality across domains:
#
#   rel-stack / user-engagement  — user is a PARENT node (posts, comments, votes
#       have FK→user). Signal flows naturally UP via FK→PK edges.
#
#   rel-avito / user-visits      — same structural argument: user is a parent of
#       search/ad interaction tables. Different domain (classifieds vs Q&A).
#
# Both tasks favor FK→PK directionality (MPNN-D expected to be competitive).
# The comparison across modes (MPNN-U, MPNN-D, Dir-GNN) measures whether
# reverse edges add meaningful signal in user-level relational prediction.
DATASET_TASKS = [
    ("rel-stack", "user-engagement"),
    ("rel-avito", "user-visits"),
]


# ──────────────────────────────────────────────────────────────────────────────
# rel-stack compatibility patch
# ──────────────────────────────────────────────────────────────────────────────

def patch_rel_stack(dataset):
    """
    After db.upto(cutoff), pkey columns are no longer consecutive integers,
    which causes validate_and_correct_db to raise RuntimeError. We no-op it.
    Dangling FKs are handled naturally in our edge builder (skipped).
    """
    dataset.validate_and_correct_db = lambda db: None


# ──────────────────────────────────────────────────────────────────────────────
# Feature encoding
# ──────────────────────────────────────────────────────────────────────────────

def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not _is_datetime(series)


def _is_categorical(series: pd.Series) -> bool:
    return not _is_numeric(series) and not _is_datetime(series)


MAX_TEXT_LEN = 200  # drop string columns whose median value length exceeds this

def get_feature_df(table_df: pd.DataFrame, table) -> pd.DataFrame:
    """Drop PK, FK, time, and free-text columns; keep only feature columns."""
    exclude = set()
    if table.pkey_col:
        exclude.add(table.pkey_col)
    exclude.update(table.fkey_col_to_pkey_table.keys())
    if table.time_col:
        exclude.add(table.time_col)
    keep = []
    for c in table_df.columns:
        if c in exclude:
            continue
        if table_df[c].dtype == object:
            sample = table_df[c].dropna().head(200)
            if len(sample) > 0 and sample.apply(lambda v: len(str(v))).median() > MAX_TEXT_LEN:
                continue  # skip free-text columns
        keep.append(c)
    return table_df[keep].copy()


def _stringify_complex(series: pd.Series) -> pd.Series:
    """Convert list/array cells to strings so OrdinalEncoder can handle them."""
    return series.apply(lambda v: str(v) if not isinstance(v, str) else v)


MAX_CAT_CARDINALITY = 10_000  # drop text-like columns with too many unique values

def _dt_offsets(feat_df: pd.DataFrame, dt_cols: list, cutoff: pd.Timestamp) -> np.ndarray:
    """Days-since-cutoff for each datetime column, as a [N, len(dt_cols)] float32 array."""
    offs = []
    for col in dt_cols:
        ts = pd.to_datetime(feat_df[col], errors="coerce")
        offset = (cutoff - ts).dt.days.fillna(0).clip(lower=0).values.astype(np.float32)
        offs.append(offset.reshape(-1, 1))
    return np.concatenate(offs, axis=1)


def fit_table_encoder(feat_df: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Fit encoders on training data. Returns encoder dict.

    All three feature kinds (numeric, categorical codes, datetime offsets) are
    standardized to mean 0 / std 1. Without this, OrdinalEncoder integer codes
    and raw day-offsets (which can run into the tens of thousands) dominate the
    properly-scaled numeric columns and the GNN fails to learn (loss plateaus,
    AUC stays ~0.5).
    """
    num_cols = [c for c in feat_df.columns if _is_numeric(feat_df[c])]
    dt_cols  = [c for c in feat_df.columns if _is_datetime(feat_df[c])]
    cat_cols = [c for c in feat_df.columns if _is_categorical(feat_df[c])
                and feat_df[c].nunique() <= MAX_CAT_CARDINALITY]

    scaler = None
    if num_cols:
        num_vals = feat_df[num_cols].fillna(0.0).values.astype(np.float32)
        scaler = StandardScaler().fit(num_vals)

    ord_enc = None
    cat_scaler = None
    if cat_cols:
        cat_vals = feat_df[cat_cols].apply(_stringify_complex).fillna("__missing__").values
        ord_enc = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        ).fit(cat_vals)
        cat_codes = ord_enc.transform(cat_vals).astype(np.float32)
        cat_scaler = StandardScaler().fit(cat_codes)

    dt_scaler = None
    if dt_cols:
        dt_scaler = StandardScaler().fit(_dt_offsets(feat_df, dt_cols, cutoff))

    return dict(
        num_cols=num_cols,
        cat_cols=cat_cols,
        dt_cols=dt_cols,
        scaler=scaler,
        ord_enc=ord_enc,
        cat_scaler=cat_scaler,
        dt_scaler=dt_scaler,
        cutoff=str(cutoff),
    )


def apply_table_encoder(feat_df: pd.DataFrame, enc: dict) -> torch.Tensor:
    """Apply a fitted encoder to a DataFrame split. Returns float32 tensor."""
    parts = []
    cutoff = pd.Timestamp(enc["cutoff"])

    if enc["scaler"] and enc["num_cols"]:
        num_vals = feat_df[enc["num_cols"]].fillna(0.0).values.astype(np.float32)
        num_vals = enc["scaler"].transform(num_vals).astype(np.float32)
        np.nan_to_num(num_vals, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)
        parts.append(num_vals)

    if enc["ord_enc"] and enc["cat_cols"]:
        cat_vals = feat_df[enc["cat_cols"]].apply(_stringify_complex).fillna("__missing__").values
        cat_enc = enc["ord_enc"].transform(cat_vals).astype(np.float32)
        np.nan_to_num(cat_enc, nan=-1.0, copy=False)
        if enc.get("cat_scaler") is not None:
            cat_enc = enc["cat_scaler"].transform(cat_enc).astype(np.float32)
            np.nan_to_num(cat_enc, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)
        parts.append(cat_enc)

    if enc["dt_cols"]:
        dt_vals = _dt_offsets(feat_df, enc["dt_cols"], cutoff)
        if enc.get("dt_scaler") is not None:
            dt_vals = enc["dt_scaler"].transform(dt_vals).astype(np.float32)
            np.nan_to_num(dt_vals, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)
        parts.append(dt_vals)

    if not parts:
        return torch.zeros(len(feat_df), 1, dtype=torch.float32)

    return torch.from_numpy(np.concatenate(parts, axis=1)).float()


# ──────────────────────────────────────────────────────────────────────────────
# HeteroData builder
# ──────────────────────────────────────────────────────────────────────────────

def build_hetero_data(
    split_db,
    db_schema,            # original db (for schema: pkey_col, fkey_col_to_pkey_table, time_col)
    task,
    label_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    node_encoders: dict,  # mutated in-place when fit=True
    fit: bool = False,
) -> HeteroData:
    """
    Build a HeteroData object from the temporally-filtered split_db.

    split_db   : Database after .upto(cutoff)
    db_schema  : Original db (for schema info, since split_db has same schema)
    task       : relbench EntityTask
    label_df   : task label table (entity_col, time_col, target_col columns)
    cutoff     : split cutoff timestamp (used for datetime encoding)
    node_encoders : dict of table_name → encoder dict (fitted on train split)
    fit        : if True, fit encoders on this split's data (use for train only)
    """
    data = HeteroData()

    # ── node features ──────────────────────────────────────────────────────
    pk_to_idx = {}   # table_name → {pk_value: sequential_node_idx}

    for tname, table_schema in db_schema.table_dict.items():
        if tname not in split_db.table_dict:
            continue
        tdf = split_db.table_dict[tname].df
        if len(tdf) == 0:
            continue

        feat_df = get_feature_df(tdf, table_schema)

        if fit or tname not in node_encoders:
            node_encoders[tname] = fit_table_encoder(feat_df, cutoff)

        x = apply_table_encoder(feat_df, node_encoders[tname])
        data[tname].x = x
        data[tname].num_nodes = len(tdf)

        # Build PK → sequential node idx mapping
        pkey = table_schema.pkey_col
        if pkey is not None and pkey in tdf.columns:
            pk_series = tdf[pkey]
        else:
            pk_series = pd.RangeIndex(len(tdf))
        pk_to_idx[tname] = {pk: idx for idx, pk in enumerate(pk_series)}

        # Time attribute for each node (used optionally by NeighborLoader)
        tcol = table_schema.time_col
        if tcol and tcol in tdf.columns:
            ts = pd.to_datetime(tdf[tcol], errors="coerce")
            time_unix = ts.values.astype("int64") // 10**9
            time_unix = np.where(np.isnat(ts.values), 0, time_unix).astype(np.float64)
            data[tname].time = torch.from_numpy(time_unix).float()

    # ── edges (FK→PK and reversed PK→FK) ──────────────────────────────────
    for tname, table_schema in db_schema.table_dict.items():
        if tname not in split_db.table_dict:
            continue
        tdf = split_db.table_dict[tname].df
        if len(tdf) == 0:
            continue
        child_idx_map = pk_to_idx.get(tname)
        if child_idx_map is None:
            continue

        child_seq = np.arange(len(tdf))  # sequential node indices of child rows

        for fk_col, parent_tname in table_schema.fkey_col_to_pkey_table.items():
            if parent_tname not in split_db.table_dict:
                continue
            if fk_col not in tdf.columns:
                continue
            parent_idx_map = pk_to_idx.get(parent_tname)
            if parent_idx_map is None:
                continue

            fk_vals = tdf[fk_col].values
            valid_mask = pd.notna(tdf[fk_col])
            valid_child = child_seq[valid_mask]
            valid_fk    = fk_vals[valid_mask]

            # map FK values to parent sequential indices
            parent_seq_vals = np.array(
                [parent_idx_map.get(int(v) if not isinstance(v, float) or not np.isnan(v) else -1, -1)
                 for v in valid_fk]
            )
            found = parent_seq_vals >= 0

            if not found.any():
                continue

            src = valid_child[found].astype(np.int64)
            dst = parent_seq_vals[found].astype(np.int64)

            edge_name = f"fk_{fk_col}"
            # FK→PK (child → parent)
            data[tname, edge_name, parent_tname].edge_index = torch.tensor(
                np.stack([src, dst]), dtype=torch.long
            )
            # PK→FK reversed (parent → child)
            data[parent_tname, f"rev_{edge_name}", tname].edge_index = torch.tensor(
                np.stack([dst, src]), dtype=torch.long
            )

    # ── labels and mask on target node type ───────────────────────────────
    entity_table = task.entity_table
    entity_col   = task.entity_col
    target_col   = task.target_col

    if entity_table in pk_to_idx:
        eidx_map = pk_to_idx[entity_table]
        n_nodes  = data[entity_table].num_nodes

        y    = torch.full((n_nodes,), float("nan"))
        mask = torch.zeros(n_nodes, dtype=torch.bool)

        # Deduplicate: take the LAST label per entity (most recent timestamp)
        if task.time_col in label_df.columns:
            ldf = label_df.sort_values(task.time_col).groupby(entity_col, sort=False).last().reset_index()
        else:
            ldf = label_df.groupby(entity_col, sort=False).last().reset_index()

        valid_rows = ldf[ldf[entity_col].isin(eidx_map)]
        if len(valid_rows) > 0:
            node_idxs = [eidx_map[eid] for eid in valid_rows[entity_col].values]
            raw = valid_rows[target_col].values
            # Handle PostgreSQL boolean strings ('t'/'f') and Python bools
            if raw.dtype == object or raw.dtype == bool:
                raw = np.where(raw == 't', 1.0,
                      np.where(raw == 'f', 0.0,
                      np.where(raw == True, 1.0,
                      np.where(raw == False, 0.0, raw))))
            labels = raw.astype(np.float32)
            y[node_idxs]    = torch.from_numpy(labels)
            mask[node_idxs] = True

        data[entity_table].y    = y
        data[entity_table].mask = mask

    return data


# ──────────────────────────────────────────────────────────────────────────────
# Main preprocessing loop
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_one(dataset_name: str, task_name: str):
    print(f"\n{'='*60}")
    print(f"  {dataset_name} / {task_name}")
    print(f"{'='*60}")

    out_dir = PROCESSED / dataset_name / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load relbench objects ───────────────────────────────────────────────
    dataset = get_dataset(dataset_name, download=True)
    task    = get_task(dataset_name, task_name, download=True)

    if dataset_name == "rel-stack":
        patch_rel_stack(dataset)
    if dataset_name == "rel-avito":
        patch_rel_stack(dataset)  # same non-consecutive PK issue after .upto()

    db = dataset.get_db()   # filtered to test_timestamp, reindexed PKs/FKs

    # Drop task-specified leakage columns (e.g. eligibilities-adult removes age/
    # adult/child columns that directly encode the target, plus free-text fields).
    remove_cols = getattr(task, "remove_columns", None) or []
    for tname, col in remove_cols:
        if tname in db.table_dict and col in db.table_dict[tname].df.columns:
            db.table_dict[tname].df.drop(columns=[col], inplace=True)

    gc.collect()

    val_ts  = dataset.val_timestamp
    test_ts = dataset.test_timestamp

    splits = {
        "train": val_ts,
        "val":   val_ts,    # same graph, different label set
        "test":  test_ts,
    }

    # ── get label tables ────────────────────────────────────────────────────
    print("Loading task tables …")
    label_tables = {
        "train": task.get_table("train", mask_input_cols=False).df,
        "val":   task.get_table("val",   mask_input_cols=False).df,
        "test":  task.get_table("test",  mask_input_cols=False).df,
    }

    # ── build & save one HeteroData per split ──────────────────────────────
    node_encoders: dict = {}   # fitted on train, reused for val/test
    gc.collect()

    for split_name, cutoff in splits.items():
        print(f"\n  [{split_name}] cutoff = {cutoff}")
        split_db = db.upto(cutoff)
        fit_now  = (split_name == "train")

        hdata = build_hetero_data(
            split_db    = split_db,
            db_schema   = db,
            task        = task,
            label_df    = label_tables[split_name],
            cutoff      = cutoff,
            node_encoders = node_encoders,
            fit         = fit_now,
        )
        del split_db
        gc.collect()

        out_path = out_dir / f"{split_name}.pt"
        n_labeled = int(hdata[task.entity_table].mask.sum()) if hasattr(hdata[task.entity_table], 'mask') else 0
        n_nodes   = hdata[task.entity_table].num_nodes
        torch.save(hdata, out_path)
        del hdata
        gc.collect()
        print(f"    saved {out_path.name}  ({n_nodes} nodes in target table, {n_labeled} labeled)")

    # ── save metadata (fitted on train) ────────────────────────────────────
    # Use train graph for schema
    train_data = torch.load(out_dir / "train.pt", weights_only=False)
    node_types = list(train_data.node_types)
    edge_types = [list(e) for e in train_data.edge_types]
    fwd_types  = [e for e in edge_types if not e[1].startswith("rev_")]
    rev_types  = [e for e in edge_types if e[1].startswith("rev_")]
    feat_dims  = {nt: int(train_data[nt].x.shape[1]) for nt in node_types}

    meta = dict(
        dataset_name   = dataset_name,
        task_name      = task_name,
        target_node    = task.entity_table,
        entity_col     = task.entity_col,
        node_types     = node_types,
        edge_types     = edge_types,
        fwd_edge_types = fwd_types,
        rev_edge_types = rev_types,
        node_feat_dims = feat_dims,
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  meta.json saved. node_types={node_types}")
    print(f"  feat dims: {feat_dims}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--task",    default=None)
    args = parser.parse_args()

    combos = DATASET_TASKS
    if args.dataset and args.task:
        combos = [(args.dataset, args.task)]
    elif args.dataset:
        combos = [(d, t) for d, t in DATASET_TASKS if d == args.dataset]

    for dataset_name, task_name in combos:
        preprocess_one(dataset_name, task_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
