"""
Baseline Dataset Preparation Script
====================================
Generates train/valid/test splits for anomaly detection experiments.

Usage:
    python prepare_xy.py

Output:
    data/
    ├── 10pct/
    │   ├── slice_12_a1/
    │   │   ├── seed_2/   (train.npz, valid.npz, test.npz)
    │   │   ├── seed_12/
    │   │   └── ...
    │   ├── slice_12_a2/
    │   ├── slice_12_a3/
    │   ├── slice_24_a1/
    │   ├── slice_24_a2/
    │   └── slice_24_a3/
    ├── 5pct/
    ├── 3pct/
    └── 1pct/

Each npz file contains:
    - X: (N, T*5) flattened features
    - X_seq: (N, T, 5) sequential features [SPEED, COURSE_SIN, COURSE_COS, LON, LAT]
    - y: (N, T) binary labels
    - route_ids: (N,) route identifiers
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


FEATURES = ["SPEED", "COURSE_SIN", "COURSE_COS", "LON", "LAT"]

BATCH_CONFIG = {
    "seeds": [2, 12, 22, 32, 42],
    "anomaly_pcts": ["10pct", "5pct", "3pct", "1pct"],
    "slices": [
        {
            "name": "slice_12",
            "T": 12,
            "csv": "routes_sliced_12_injected.csv",
            "indices": {
                "a1": "indices/indices_12_a1.csv",
                "a2": "indices/indices_12_a2.csv",
                "a3": "indices/indices_12_a3.csv",
            },
        },
        {
            "name": "slice_24",
            "T": 24,
            "csv": "routes_sliced_24_injected.csv",
            "indices": {
                "a1": "indices/indices_24_a1.csv",
                "a2": "indices/indices_24_a2.csv",
                "a3": "indices/indices_24_a3.csv",
            },
        },
    ],
    "modes": ["a1", "a2", "a3"],
    "train": 0.7,
    "valid": 0.15,
    "test": 0.15,
}

PCT_TO_COL = {
    "10pct": "use_10pct",
    "5pct": "use_5pct",
    "3pct": "use_3pct",
    "1pct": "use_1pct",
}


def load_anomaly_route_ids(indices_path: str, pct: str, mode: str) -> set:
    """Load route IDs that should be anomaly for given percentage."""
    col = PCT_TO_COL[pct]
    df = pd.read_csv(indices_path)
    df[col] = df[col].astype(str).str.lower().eq("true")
    route_ids = df.loc[df[col], "ROUTE_ID"].astype(str).tolist()
    # A3 anomalies exist in both original and virtual (A3_ prefixed) route IDs
    if mode.lower() == "a3":
        route_ids = route_ids + [f"A3_{rid}" for rid in route_ids]
    return set(route_ids)


def build_xy(csv_path: str, *, mode: str, T: int, anomaly_route_ids: set | None = None):
    mode = mode.lower()
    if mode not in {"a1", "a2", "a3"}:
        raise ValueError("mode must be one of: a1, a2, a3")

    df = pd.read_csv(csv_path, dtype={"ROUTE_ID": str, "ANOMALY_TYPE": str}, low_memory=False)
    df["ROUTE_ID"] = df["ROUTE_ID"].astype(str)
    df["T_INDEX"] = df["T_INDEX"].astype(int)
    df["ANOMALY"] = df["ANOMALY"].astype(str).str.lower().eq("true")
    df["ANOMALY_TYPE"] = df["ANOMALY_TYPE"].fillna("").astype(str).str.upper()

    if mode in {"a1", "a2"}:
        # Exclude A3 virtual vessel routes (prefixed with "A3_")
        df = df[~df["ROUTE_ID"].str.startswith("A3_")]

    df = df.sort_values(["ROUTE_ID", "T_INDEX"], kind="mergesort")
    g = df.groupby("ROUTE_ID", sort=False)
    agg = g.agg(
        size=("T_INDEX", "size"),
        tmin=("T_INDEX", "min"),
        tmax=("T_INDEX", "max"),
        tn=("T_INDEX", "nunique"),
    )
    good_ids = agg[(agg["size"] == T) & (agg["tn"] == T) & (agg["tmin"] == 0) & (agg["tmax"] == T - 1)].index
    df = df[df["ROUTE_ID"].isin(good_ids)]

    route_ids = df["ROUTE_ID"].drop_duplicates().to_numpy()
    X_seq = df[FEATURES].to_numpy(dtype=np.float32).reshape(-1, T, len(FEATURES))

    # Apply anomaly filtering based on route_ids for this percentage
    is_anomaly = df["ANOMALY"].to_numpy() & (df["ANOMALY_TYPE"].to_numpy() == mode.upper())
    if anomaly_route_ids is not None:
        # Only keep anomaly label if route_id is in the allowed set
        route_id_per_row = df["ROUTE_ID"].to_numpy()
        is_in_pct = np.isin(route_id_per_row, list(anomaly_route_ids))
        is_anomaly = is_anomaly & is_in_pct

    y = is_anomaly.astype(np.int8).reshape(-1, T)
    X_flat = X_seq.reshape(len(route_ids), -1)
    return X_flat, y, route_ids, X_seq


def stratified_split_indices(y: np.ndarray, *, train: float, valid: float, test: float, seed: int = 0):
    if not (0 < train < 1 and 0 < valid < 1 and 0 < test < 1):
        raise ValueError("train/valid/test must be in (0,1)")
    s = train + valid + test
    if abs(s - 1.0) > 1e-6:
        raise ValueError("train+valid+test must sum to 1")

    labels = y.sum(axis=1).astype(int)
    rng = np.random.default_rng(int(seed))

    def _split_group(idxs: np.ndarray):
        idxs = idxs.copy()
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * train))
        n_valid = int(round(n * valid))
        n_train = min(max(n_train, 0), n)
        n_valid = min(max(n_valid, 0), n - n_train)
        n_test = n - n_train - n_valid
        return idxs[:n_train], idxs[n_train : n_train + n_valid], idxs[n_train + n_valid : n_train + n_valid + n_test]

    tr_parts = []
    va_parts = []
    te_parts = []
    for lab in np.unique(labels):
        idxs = np.flatnonzero(labels == int(lab))
        tr_g, va_g, te_g = _split_group(idxs)
        tr_parts.append(tr_g)
        va_parts.append(va_g)
        te_parts.append(te_g)

    tr = np.concatenate(tr_parts) if tr_parts else np.array([], dtype=int)
    va = np.concatenate(va_parts) if va_parts else np.array([], dtype=int)
    te = np.concatenate(te_parts) if te_parts else np.array([], dtype=int)
    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "data")
    cfg = BATCH_CONFIG
    total = len(cfg["anomaly_pcts"]) * len(cfg["seeds"]) * len(cfg["slices"]) * len(cfg["modes"])
    count = 0

    print(f"Output directory: {out_dir}\n")

    for pct in cfg["anomaly_pcts"]:
        print(f"\n{'='*50}")
        print(f"Processing anomaly ratio: {pct}")
        print(f"{'='*50}")

        for slice_cfg in cfg["slices"]:
            csv_path = os.path.join(script_dir, slice_cfg["csv"])
            T = slice_cfg["T"]
            slice_name = slice_cfg["name"]

            if not os.path.exists(csv_path):
                print(f"[SKIP] CSV not found: {csv_path}")
                continue

            for mode in cfg["modes"]:
                # Load anomaly route IDs for this percentage
                indices_path = os.path.join(script_dir, slice_cfg["indices"][mode])
                if not os.path.exists(indices_path):
                    print(f"[SKIP] Indices not found: {indices_path}")
                    continue

                anomaly_route_ids = load_anomaly_route_ids(indices_path, pct, mode)

                print(f"\n=== {slice_name} / {mode.upper()} (T={T}) ===")
                X_flat, y, route_ids, X_seq = build_xy(
                    csv_path, mode=mode, T=T, anomaly_route_ids=anomaly_route_ids
                )
                n_anomaly_samples = (y.sum(axis=1) > 0).sum()
                print(f"    Total samples: {len(route_ids)}, Anomaly samples: {n_anomaly_samples}")

                for seed in cfg["seeds"]:
                    subdir = os.path.join(out_dir, pct, f"{slice_name}_{mode}", f"seed_{seed}")
                    os.makedirs(subdir, exist_ok=True)

                    tr, va, te = stratified_split_indices(
                        y, train=cfg["train"], valid=cfg["valid"], test=cfg["test"], seed=seed
                    )

                    for split_name, idx in [("train", tr), ("valid", va), ("test", te)]:
                        np.savez_compressed(
                            os.path.join(subdir, f"{split_name}.npz"),
                            X=X_flat[idx],
                            X_seq=X_seq[idx],
                            y=y[idx],
                            route_ids=route_ids[idx],
                        )

                    count += 1
                    print(f"    [seed={seed}] train={len(tr)}, valid={len(va)}, test={len(te)}")

    print(f"\n{'='*50}")
    print(f"Done! Generated {count}/{total} dataset combinations in {out_dir}")
    print(f"{'='*50}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
