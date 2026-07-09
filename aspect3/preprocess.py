"""
preprocess.py  —  Build HeteroData objects for the Node-Features ablation (Aspect 3).

Aspect 3 studies how the *initial node representation* affects an HGT model.
The three strategies compared are:

  id      — no tuple features; each node gets a learnable embedding keyed on a
            stable global ID (Embedding table, sized per node type).
  column  — column-wise encoding: each cell encoded by its column type
            (numeric → StandardScaler, categorical → OrdinalEncoder, datetime →
            days-since-cutoff), concatenated into a per-tuple vector.
  llm     — each tuple is stringified as "col1=v1, col2=v2, …" and embedded with
            a frozen sentence-transformer.

To guarantee the three strategies are compared on *exactly the same graph and the
same sample* (a requirement of the assignment), we precompute ALL THREE inputs in
a single pass and store them together in each split:

    data[nt].x       — column-wise features        [N, feat_dim_nt]
    data[nt].gid     — stable global node id        [N]         (for `id` mode)
    data[nt].x_llm   — sentence-transformer vectors [N, llm_dim] (for `llm` mode)

`gid` (not `n_id`) is used deliberately: PyG's NeighborLoader writes `batch[nt].n_id`
with the sampled node indices, which would clobber our own attribute.

Output layout:
  processed/{dataset}/{task}/{split}.pt
  processed/{dataset}/{task}/meta.json

Run:
  python preprocess.py
  python preprocess.py --dataset rel-f1 --task driver-dnf
  python preprocess.py --skip_llm      # build x + gid only (quick; llm mode unavailable)
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

# ──────────────────────────────────────────────────────────────────────────────
# Dataset / task selection
#
# Deliberately DIFFERENT datasets from aspect1 (which used rel-stack & rel-avito),
# so the two ablations cover distinct domains.
#
#   rel-f1  / driver-dnf   — predict whether a driver Does-Not-Finish the next race.
#       Small dataset (tens of thousands of rows) with healthy class balance
#       (~15-30% positive). Small enough to run the heavy LLM encoding over the
#       FULL graph, so no sampling is needed here.
#
#   rel-event / user-repeat — predict whether a user attends another event within
#       7 days. Medium/large; we cap non-target tables (see SAMPLE_CAPS) to keep
#       the LLM pass and the id-embedding table tractable. The SAME capped graph is
#       reused for all three feature strategies, satisfying the "same sample" rule.
# ──────────────────────────────────────────────────────────────────────────────
DATASET_TASKS = [
    ("rel-f1",    "driver-dnf"),
    ("rel-event", "user-repeat"),
]

# Max rows kept per NON-target table (target/entity table is never sampled, so the
# full label distribution is preserved). None → keep the full table.
# Uniform row sampling keeps each table's marginal feature distribution intact and
# roughly preserves connectivity density; dangling FKs to dropped rows are skipped
# by the edge builder. All three feature strategies read the identical sampled graph.
SAMPLE_CAPS = {
    "rel-f1":    None,      # small — keep everything
    "rel-event": 60_000,    # cap large interaction tables
}
SAMPLE_SEED = 42

# Sentence-transformer used for the `llm` strategy.
LLM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_DIM        = 384
LLM_BATCH      = 256


# ──────────────────────────────────────────────────────────────────────────────
# relbench compatibility patch (same non-consecutive-PK issue as aspect1)
# ──────────────────────────────────────────────────────────────────────────────

def patch_validate(dataset):
    """After db.upto(cutoff), pkey columns are no longer consecutive integers,
    which trips validate_and_correct_db. We no-op it; dangling FKs are handled in
    the edge builder (skipped)."""
    dataset.validate_and_correct_db = lambda db: None


# ──────────────────────────────────────────────────────────────────────────────
# Feature-column selection (shared by column-wise encoding and LLM stringification)
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


# ──────────────────────────────────────────────────────────────────────────────
# Column-wise encoding  (strategy: column)
# ──────────────────────────────────────────────────────────────────────────────

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
    """Fit encoders on training data. All three feature kinds (numeric, categorical
    codes, datetime offsets) are standardized to mean 0 / std 1 so no single kind
    dominates the GNN inputs (see aspect1 for the failure mode without this)."""
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
        num_cols=num_cols, cat_cols=cat_cols, dt_cols=dt_cols,
        scaler=scaler, ord_enc=ord_enc, cat_scaler=cat_scaler, dt_scaler=dt_scaler,
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
# LLM stringification + embedding  (strategy: llm)
# ──────────────────────────────────────────────────────────────────────────────

def build_row_strings(feat_df: pd.DataFrame) -> list:
    """Vectorized "col1=v1, col2=v2, …" for every row (over feature columns)."""
    if feat_df.shape[1] == 0:
        return ["" for _ in range(len(feat_df))]
    parts = None
    for c in feat_df.columns:
        col = c + "=" + feat_df[c].astype(str)
        parts = col if parts is None else parts + ", " + col
    return parts.fillna("").tolist()


def compute_llm_embeddings(db, model) -> dict:
    """Embed every tuple of every table once, keyed by the table's global row order.

    Returns {table_name: np.ndarray[num_rows, LLM_DIM]}. Row i corresponds to the
    i-th row of db.table_dict[t].df, which is exactly the global id (gid) assigned
    in assign_global_ids(). Computed on the FULL (already-sampled) db so train/val/
    test just index into it — the LLM pass runs only once per table.
    """
    emb = {}
    for tname, table in db.table_dict.items():
        tdf = table.df
        if len(tdf) == 0:
            emb[tname] = np.zeros((0, LLM_DIM), dtype=np.float32)
            continue
        feat_df = get_feature_df(tdf, table)
        strings = build_row_strings(feat_df)
        vecs = model.encode(
            strings, batch_size=LLM_BATCH, convert_to_numpy=True,
            show_progress_bar=False, normalize_embeddings=True,
        ).astype(np.float32)
        emb[tname] = vecs
        print(f"    llm[{tname}]: {vecs.shape}")
    return emb


# ──────────────────────────────────────────────────────────────────────────────
# Stable global node IDs  (strategy: id)
# ──────────────────────────────────────────────────────────────────────────────

def assign_global_ids(db) -> dict:
    """Map each table's PK value → a stable global id (its row position in the full
    db). Shared across train/val/test so a node's learnable id-embedding is the same
    entity in every split (train→test transfer for entities seen during training)."""
    gid_maps = {}
    for tname, table in db.table_dict.items():
        tdf = table.df
        pkey = table.pkey_col
        pk_series = tdf[pkey] if (pkey and pkey in tdf.columns) else pd.RangeIndex(len(tdf))
        gid_maps[tname] = {pk: gid for gid, pk in enumerate(pk_series)}
    return gid_maps


# ──────────────────────────────────────────────────────────────────────────────
# Optional sampling (LLM feasibility / id-embedding size)
# ──────────────────────────────────────────────────────────────────────────────

def sample_db(db, target_table: str, cap):
    """Uniformly downsample every NON-target table to at most `cap` rows (seeded).
    Target/entity table is left intact so the full label distribution is preserved.
    Mutates db in place; the temporal splits (db.upto) inherit the sample."""
    if cap is None:
        return db
    rng = np.random.RandomState(SAMPLE_SEED)
    for tname, table in db.table_dict.items():
        if tname == target_table:
            continue
        tdf = table.df
        if len(tdf) > cap:
            keep = rng.choice(len(tdf), size=cap, replace=False)
            keep.sort()
            table.df = tdf.iloc[keep].reset_index(drop=True)
            print(f"    sampled {tname}: {len(tdf)} → {cap}")
    return db


# ──────────────────────────────────────────────────────────────────────────────
# HeteroData builder
# ──────────────────────────────────────────────────────────────────────────────

def build_hetero_data(
    split_db,
    db_schema,
    task,
    label_df: pd.DataFrame,
    cutoff: pd.Timestamp,
    node_encoders: dict,   # mutated in-place when fit=True
    gid_maps: dict,        # stable global ids (from full db)
    llm_emb: dict,         # {table: np.ndarray[num_global, LLM_DIM]} or None
    fit: bool = False,
) -> HeteroData:
    """Build one HeteroData with x (column), gid (id), x_llm (llm) on every node."""
    data = HeteroData()
    pk_to_idx = {}  # table_name → {pk_value: sequential_node_idx}

    # ── node inputs ────────────────────────────────────────────────────────
    for tname, table_schema in db_schema.table_dict.items():
        if tname not in split_db.table_dict:
            continue
        tdf = split_db.table_dict[tname].df
        if len(tdf) == 0:
            continue

        feat_df = get_feature_df(tdf, table_schema)

        if fit or tname not in node_encoders:
            node_encoders[tname] = fit_table_encoder(feat_df, cutoff)

        # column-wise features
        data[tname].x = apply_table_encoder(feat_df, node_encoders[tname])
        data[tname].num_nodes = len(tdf)

        # PK → sequential idx (this split)
        pkey = table_schema.pkey_col
        pk_series = tdf[pkey] if (pkey and pkey in tdf.columns) else pd.RangeIndex(len(tdf))
        pk_list = list(pk_series)
        pk_to_idx[tname] = {pk: idx for idx, pk in enumerate(pk_list)}

        # stable global ids for this split's nodes (for `id` strategy)
        gmap = gid_maps.get(tname, {})
        gids = np.array([gmap.get(pk, 0) for pk in pk_list], dtype=np.int64)
        data[tname].gid = torch.from_numpy(gids)

        # LLM embeddings for this split's nodes (for `llm` strategy)
        if llm_emb is not None and tname in llm_emb and llm_emb[tname].shape[0] > 0:
            data[tname].x_llm = torch.from_numpy(llm_emb[tname][gids]).float()

        # time attr (used optionally by NeighborLoader temporal sampling)
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
        if pk_to_idx.get(tname) is None:
            continue

        child_seq = np.arange(len(tdf))
        for fk_col, parent_tname in table_schema.fkey_col_to_pkey_table.items():
            if parent_tname not in split_db.table_dict or fk_col not in tdf.columns:
                continue
            parent_idx_map = pk_to_idx.get(parent_tname)
            if parent_idx_map is None:
                continue

            fk_vals    = tdf[fk_col].values
            valid_mask = pd.notna(tdf[fk_col])
            valid_child = child_seq[valid_mask]
            valid_fk    = fk_vals[valid_mask]

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
            data[tname, edge_name, parent_tname].edge_index = torch.tensor(
                np.stack([src, dst]), dtype=torch.long)
            data[parent_tname, f"rev_{edge_name}", tname].edge_index = torch.tensor(
                np.stack([dst, src]), dtype=torch.long)

    # ── labels and mask on target node type ───────────────────────────────
    entity_table = task.entity_table
    entity_col   = task.entity_col
    target_col   = task.target_col

    if entity_table in pk_to_idx:
        eidx_map = pk_to_idx[entity_table]
        n_nodes  = data[entity_table].num_nodes
        y    = torch.full((n_nodes,), float("nan"))
        mask = torch.zeros(n_nodes, dtype=torch.bool)

        if task.time_col in label_df.columns:
            ldf = label_df.sort_values(task.time_col).groupby(entity_col, sort=False).last().reset_index()
        else:
            ldf = label_df.groupby(entity_col, sort=False).last().reset_index()

        valid_rows = ldf[ldf[entity_col].isin(eidx_map)]
        if len(valid_rows) > 0:
            node_idxs = [eidx_map[eid] for eid in valid_rows[entity_col].values]
            raw = valid_rows[target_col].values
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

def preprocess_one(dataset_name: str, task_name: str, skip_llm: bool):
    print(f"\n{'='*60}\n  {dataset_name} / {task_name}\n{'='*60}")
    out_dir = PROCESSED / dataset_name / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_dataset(dataset_name, download=True)
    task    = get_task(dataset_name, task_name, download=True)
    patch_validate(dataset)

    db = dataset.get_db()

    # Drop task-specified leakage columns.
    remove_cols = getattr(task, "remove_columns", None) or []
    for tname, col in remove_cols:
        if tname in db.table_dict and col in db.table_dict[tname].df.columns:
            db.table_dict[tname].df.drop(columns=[col], inplace=True)

    # Optional sampling (shared by all three feature strategies).
    sample_db(db, task.entity_table, SAMPLE_CAPS.get(dataset_name))
    gc.collect()

    # Stable global ids (from the full, sampled db).
    gid_maps = assign_global_ids(db)
    node_num_global = {t: len(db.table_dict[t].df) for t in db.table_dict}

    # LLM embeddings — one pass over the full sampled db.
    llm_emb = None
    if not skip_llm:
        print("Loading sentence-transformer and embedding tuples …")
        from sentence_transformers import SentenceTransformer
        st_device = "cuda" if torch.cuda.is_available() else "cpu"
        st_model = SentenceTransformer(LLM_MODEL_NAME, device=st_device)
        llm_emb = compute_llm_embeddings(db, st_model)
        del st_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    val_ts, test_ts = dataset.val_timestamp, dataset.test_timestamp
    splits = {"train": val_ts, "val": val_ts, "test": test_ts}

    print("Loading task tables …")
    label_tables = {
        s: task.get_table(s, mask_input_cols=False).df for s in ("train", "val", "test")
    }

    node_encoders: dict = {}
    for split_name, cutoff in splits.items():
        print(f"\n  [{split_name}] cutoff = {cutoff}")
        split_db = db.upto(cutoff)
        hdata = build_hetero_data(
            split_db=split_db, db_schema=db, task=task,
            label_df=label_tables[split_name], cutoff=cutoff,
            node_encoders=node_encoders, gid_maps=gid_maps, llm_emb=llm_emb,
            fit=(split_name == "train"),
        )
        del split_db
        gc.collect()

        out_path = out_dir / f"{split_name}.pt"
        n_labeled = int(hdata[task.entity_table].mask.sum()) if hasattr(hdata[task.entity_table], "mask") else 0
        n_nodes   = hdata[task.entity_table].num_nodes
        torch.save(hdata, out_path)
        del hdata
        gc.collect()
        print(f"    saved {out_path.name}  ({n_nodes} target nodes, {n_labeled} labeled)")

    # ── metadata (schema from train) ──────────────────────────────────────
    train_data = torch.load(out_dir / "train.pt", weights_only=False)
    node_types = list(train_data.node_types)
    edge_types = [list(e) for e in train_data.edge_types]
    feat_dims  = {nt: int(train_data[nt].x.shape[1]) for nt in node_types}

    meta = dict(
        dataset_name    = dataset_name,
        task_name       = task_name,
        target_node     = task.entity_table,
        entity_col      = task.entity_col,
        node_types      = node_types,
        edge_types      = edge_types,
        node_feat_dims  = feat_dims,               # column-wise input dims
        node_num_global = node_num_global,         # id-embedding table sizes
        llm_dim         = LLM_DIM if not skip_llm else 0,
        has_llm         = (not skip_llm),
        sample_cap      = SAMPLE_CAPS.get(dataset_name),
    )
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  meta.json saved. node_types={node_types}")
    print(f"  column feat dims: {feat_dims}")
    print(f"  id table sizes  : {node_num_global}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--task",    default=None)
    parser.add_argument("--skip_llm", action="store_true",
                        help="Build x + gid only (fast). `llm` feat_mode will be unavailable.")
    args = parser.parse_args()

    combos = DATASET_TASKS
    if args.dataset and args.task:
        combos = [(args.dataset, args.task)]
    elif args.dataset:
        combos = [(d, t) for d, t in DATASET_TASKS if d == args.dataset]

    for dataset_name, task_name in combos:
        preprocess_one(dataset_name, task_name, args.skip_llm)

    print("\nDone.")


if __name__ == "__main__":
    main()
