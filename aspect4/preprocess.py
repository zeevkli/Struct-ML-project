"""
preprocess.py  —  Build HeteroData objects for the Oversmoothing ablation (Aspect 4).

Aspect 4 studies how GNN *depth* drives oversmoothing (node representations collapse
toward each other) and how a mitigation (skip connections / DropEdge) helps. The graph
itself is the same relational HeteroData used elsewhere — column-wise node features,
FK→PK edges plus reversed PK→FK edges. No LLM / id features are needed here.

Output layout:
  processed/{dataset}/{task}/{split}.pt
  processed/{dataset}/{task}/meta.json

Run:
  python preprocess.py
  python preprocess.py --dataset rel-f1 --task driver-dnf
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

# Datasets are distinct from the other aspects: rel-f1 (aspect1/3 do not; used here for a
# small, shallow graph) and rel-trial (unused elsewhere — clinical-trials domain, entity =
# study). One small graph and one medium graph let us check the oversmoothing trend
# generalizes across scale/domain. Change freely.
DATASET_TASKS = [
    ("rel-f1",    "driver-dnf"),
    ("rel-trial", "study-outcome"),
]

# Cap non-target tables (only relevant for the larger rel-trial); target table kept whole.
SAMPLE_CAPS = {
    "rel-f1":    None,
    "rel-trial": 60_000,
}
SAMPLE_SEED = 42


def patch_validate(dataset):
    """After db.upto(cutoff), pkeys are no longer consecutive → validate_and_correct_db
    raises. No-op it; dangling FKs are skipped by the edge builder."""
    dataset.validate_and_correct_db = lambda db: None


# ──────────────────────────────────────────────────────────────────────────────
# Feature encoding  (identical scheme to aspect1/aspect3, self-contained)
# ──────────────────────────────────────────────────────────────────────────────

def _is_datetime(s):  return pd.api.types.is_datetime64_any_dtype(s)
def _is_numeric(s):   return pd.api.types.is_numeric_dtype(s) and not _is_datetime(s)
def _is_categorical(s): return not _is_numeric(s) and not _is_datetime(s)

MAX_TEXT_LEN = 200
MAX_CAT_CARDINALITY = 10_000


def get_feature_df(table_df, table):
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
                continue
        keep.append(c)
    return table_df[keep].copy()


def _stringify_complex(series):
    return series.apply(lambda v: str(v) if not isinstance(v, str) else v)


def _dt_offsets(feat_df, dt_cols, cutoff):
    offs = []
    for col in dt_cols:
        ts = pd.to_datetime(feat_df[col], errors="coerce")
        offset = (cutoff - ts).dt.days.fillna(0).clip(lower=0).values.astype(np.float32)
        offs.append(offset.reshape(-1, 1))
    return np.concatenate(offs, axis=1)


def fit_table_encoder(feat_df, cutoff):
    num_cols = [c for c in feat_df.columns if _is_numeric(feat_df[c])]
    dt_cols  = [c for c in feat_df.columns if _is_datetime(feat_df[c])]
    cat_cols = [c for c in feat_df.columns if _is_categorical(feat_df[c])
                and feat_df[c].nunique() <= MAX_CAT_CARDINALITY]

    scaler = None
    if num_cols:
        scaler = StandardScaler().fit(feat_df[num_cols].fillna(0.0).values.astype(np.float32))

    ord_enc = cat_scaler = None
    if cat_cols:
        cat_vals = feat_df[cat_cols].apply(_stringify_complex).fillna("__missing__").values
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1, encoded_missing_value=-1).fit(cat_vals)
        cat_scaler = StandardScaler().fit(ord_enc.transform(cat_vals).astype(np.float32))

    dt_scaler = StandardScaler().fit(_dt_offsets(feat_df, dt_cols, cutoff)) if dt_cols else None

    return dict(num_cols=num_cols, cat_cols=cat_cols, dt_cols=dt_cols,
                scaler=scaler, ord_enc=ord_enc, cat_scaler=cat_scaler,
                dt_scaler=dt_scaler, cutoff=str(cutoff))


def apply_table_encoder(feat_df, enc):
    parts = []
    cutoff = pd.Timestamp(enc["cutoff"])
    if enc["scaler"] and enc["num_cols"]:
        v = enc["scaler"].transform(feat_df[enc["num_cols"]].fillna(0.0).values.astype(np.float32)).astype(np.float32)
        np.nan_to_num(v, nan=0.0, posinf=3.0, neginf=-3.0, copy=False); parts.append(v)
    if enc["ord_enc"] and enc["cat_cols"]:
        cv = feat_df[enc["cat_cols"]].apply(_stringify_complex).fillna("__missing__").values
        v = enc["ord_enc"].transform(cv).astype(np.float32)
        np.nan_to_num(v, nan=-1.0, copy=False)
        if enc.get("cat_scaler") is not None:
            v = enc["cat_scaler"].transform(v).astype(np.float32)
            np.nan_to_num(v, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)
        parts.append(v)
    if enc["dt_cols"]:
        v = _dt_offsets(feat_df, enc["dt_cols"], cutoff)
        if enc.get("dt_scaler") is not None:
            v = enc["dt_scaler"].transform(v).astype(np.float32)
            np.nan_to_num(v, nan=0.0, posinf=3.0, neginf=-3.0, copy=False)
        parts.append(v)
    if not parts:
        return torch.zeros(len(feat_df), 1, dtype=torch.float32)
    return torch.from_numpy(np.concatenate(parts, axis=1)).float()


def sample_db(db, target_table, cap):
    if cap is None:
        return db
    rng = np.random.RandomState(SAMPLE_SEED)
    for tname, table in db.table_dict.items():
        if tname == target_table:
            continue
        tdf = table.df
        if len(tdf) > cap:
            keep = rng.choice(len(tdf), size=cap, replace=False); keep.sort()
            table.df = tdf.iloc[keep].reset_index(drop=True)
            print(f"    sampled {tname}: {len(tdf)} → {cap}")
    return db


# ──────────────────────────────────────────────────────────────────────────────
# HeteroData builder
# ──────────────────────────────────────────────────────────────────────────────

def build_hetero_data(split_db, db_schema, task, label_df, cutoff, node_encoders, fit=False):
    data = HeteroData()
    pk_to_idx = {}

    for tname, tschema in db_schema.table_dict.items():
        if tname not in split_db.table_dict:
            continue
        tdf = split_db.table_dict[tname].df
        if len(tdf) == 0:
            continue
        feat_df = get_feature_df(tdf, tschema)
        if fit or tname not in node_encoders:
            node_encoders[tname] = fit_table_encoder(feat_df, cutoff)
        data[tname].x = apply_table_encoder(feat_df, node_encoders[tname])
        data[tname].num_nodes = len(tdf)

        pkey = tschema.pkey_col
        pk_series = tdf[pkey] if (pkey and pkey in tdf.columns) else pd.RangeIndex(len(tdf))
        pk_to_idx[tname] = {pk: idx for idx, pk in enumerate(pk_series)}

        tcol = tschema.time_col
        if tcol and tcol in tdf.columns:
            ts = pd.to_datetime(tdf[tcol], errors="coerce")
            time_unix = ts.values.astype("int64") // 10**9
            time_unix = np.where(np.isnat(ts.values), 0, time_unix).astype(np.float64)
            data[tname].time = torch.from_numpy(time_unix).float()

    for tname, tschema in db_schema.table_dict.items():
        if tname not in split_db.table_dict:
            continue
        tdf = split_db.table_dict[tname].df
        if len(tdf) == 0 or pk_to_idx.get(tname) is None:
            continue
        child_seq = np.arange(len(tdf))
        for fk_col, parent_tname in tschema.fkey_col_to_pkey_table.items():
            if parent_tname not in split_db.table_dict or fk_col not in tdf.columns:
                continue
            parent_idx_map = pk_to_idx.get(parent_tname)
            if parent_idx_map is None:
                continue
            fk_vals = tdf[fk_col].values
            valid_mask = pd.notna(tdf[fk_col])
            valid_child = child_seq[valid_mask]
            valid_fk = fk_vals[valid_mask]
            parent_seq = np.array(
                [parent_idx_map.get(int(v) if not isinstance(v, float) or not np.isnan(v) else -1, -1)
                 for v in valid_fk])
            found = parent_seq >= 0
            if not found.any():
                continue
            src = valid_child[found].astype(np.int64)
            dst = parent_seq[found].astype(np.int64)
            en = f"fk_{fk_col}"
            data[tname, en, parent_tname].edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
            data[parent_tname, f"rev_{en}", tname].edge_index = torch.tensor(np.stack([dst, src]), dtype=torch.long)

    et, ec, tc = task.entity_table, task.entity_col, task.target_col
    if et in pk_to_idx:
        eidx = pk_to_idx[et]
        n = data[et].num_nodes
        y = torch.full((n,), float("nan")); mask = torch.zeros(n, dtype=torch.bool)
        if task.time_col in label_df.columns:
            ldf = label_df.sort_values(task.time_col).groupby(ec, sort=False).last().reset_index()
        else:
            ldf = label_df.groupby(ec, sort=False).last().reset_index()
        vr = ldf[ldf[ec].isin(eidx)]
        if len(vr) > 0:
            idxs = [eidx[e] for e in vr[ec].values]
            raw = vr[tc].values
            if raw.dtype == object or raw.dtype == bool:
                raw = np.where(raw == 't', 1.0, np.where(raw == 'f', 0.0,
                      np.where(raw == True, 1.0, np.where(raw == False, 0.0, raw))))
            y[idxs] = torch.from_numpy(raw.astype(np.float32));
            for i in idxs: mask[i] = True
        data[et].y = y; data[et].mask = mask
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_one(dataset_name, task_name):
    print(f"\n{'='*60}\n  {dataset_name} / {task_name}\n{'='*60}")
    out_dir = PROCESSED / dataset_name / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(dataset_name, download=True)
    task    = get_task(dataset_name, task_name, download=True)
    patch_validate(dataset)
    db = dataset.get_db()

    for tname, col in (getattr(task, "remove_columns", None) or []):
        if tname in db.table_dict and col in db.table_dict[tname].df.columns:
            db.table_dict[tname].df.drop(columns=[col], inplace=True)

    sample_db(db, task.entity_table, SAMPLE_CAPS.get(dataset_name))
    gc.collect()

    val_ts, test_ts = dataset.val_timestamp, dataset.test_timestamp
    splits = {"train": val_ts, "val": val_ts, "test": test_ts}
    label_tables = {s: task.get_table(s, mask_input_cols=False).df for s in ("train", "val", "test")}

    node_encoders = {}
    for split_name, cutoff in splits.items():
        print(f"\n  [{split_name}] cutoff = {cutoff}")
        split_db = db.upto(cutoff)
        hdata = build_hetero_data(split_db, db, task, label_tables[split_name],
                                  cutoff, node_encoders, fit=(split_name == "train"))
        del split_db; gc.collect()
        n_lab = int(hdata[task.entity_table].mask.sum()) if hasattr(hdata[task.entity_table], "mask") else 0
        torch.save(hdata, out_dir / f"{split_name}.pt")
        print(f"    saved {split_name}.pt  ({hdata[task.entity_table].num_nodes} target nodes, {n_lab} labeled)")
        del hdata; gc.collect()

    train_data = torch.load(out_dir / "train.pt", weights_only=False)
    node_types = list(train_data.node_types)
    edge_types = [list(e) for e in train_data.edge_types]
    feat_dims = {nt: int(train_data[nt].x.shape[1]) for nt in node_types}
    meta = dict(dataset_name=dataset_name, task_name=task_name,
                target_node=task.entity_table, entity_col=task.entity_col,
                node_types=node_types, edge_types=edge_types, node_feat_dims=feat_dims,
                sample_cap=SAMPLE_CAPS.get(dataset_name))
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
    print(f"\n  meta.json saved. node_types={node_types}")
    print(f"  column feat dims: {feat_dims}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None)
    p.add_argument("--task", default=None)
    a = p.parse_args()
    combos = DATASET_TASKS
    if a.dataset and a.task:
        combos = [(a.dataset, a.task)]
    elif a.dataset:
        combos = [(d, t) for d, t in DATASET_TASKS if d == a.dataset]
    for d, t in combos:
        preprocess_one(d, t)
    print("\nDone.")


if __name__ == "__main__":
    main()
