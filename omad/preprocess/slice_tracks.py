"""
Splits hourly point (long format) CSV per Track_ID into routes of N timestamps.

Requirements
- Sort by TIMESTAMP in ascending order per Track_ID
- Remove trim points from front/back of each Track
- Split remaining sections into non-overlapping window-sized segments
- Keep only routes with exactly window points (discard partial fragments)
- Assign route_id (globally unique ID) and route_point_idx (0..window-1)
- Remove TYPE, MONTH columns from results (default)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment Track_ID time series into fixed-size routes.")
    p.add_argument(
        "--input",
        required=True,
        help="Input CSV path (e.g., data/df_filtered.csv)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output CSV path (e.g., output/routes_long.csv)",
    )
    p.add_argument("--window", type=int, default=10, help="Number of timestamps per route (default: 10)")
    p.add_argument("--trim", type=int, default=0, help="Points to remove from front/back of each Track (default: 0)")
    p.add_argument(
        "--min_track_len",
        type=int,
        default=30,
        help="Minimum timestamp count per original Track_ID (default: 30)",
    )
    p.add_argument(
        "--drop_cols",
        nargs="*",
        default=["TYPE", "MONTH"],
        help='Columns to remove from result (default: "TYPE" "MONTH")',
    )
    return p.parse_args(argv)

def _finalize_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Column names: all uppercase
    - Column order: reorder to user-specified order (existing columns only)
    """
    if df.empty:
        return df.rename(columns={c: c.upper() for c in df.columns})

    df = df.rename(columns={c: c.upper() for c in df.columns})

    desired_order = [
        "ROUTE_ID",
        "ROUTE_POINT_IDX",
        "TIMESTAMP",
        "CRAFT_ID",
        "TRACK_ID",
        "LON",
        "LAT",
        "SPEED",
        "COURSE_SIN",
        "COURSE_COS",
        "ANOMALY",
    ]

    missing = [c for c in desired_order if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in output schema: {missing}")

    # Keep only user-requested schema columns (remove others)
    return df[desired_order]


def build_routes_long(
    df: pd.DataFrame,
    *,
    window: int = 10,
    trim: int = 0,
    min_track_len: int = 30,
    drop_cols: Iterable[str] = ("TYPE", "MONTH"),
) -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive.")
    if trim < 0:
        raise ValueError("trim must be >= 0.")
    if min_track_len < 0:
        raise ValueError("min_track_len must be >= 0.")

    required = {"Track_ID", "TIMESTAMP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input df: {sorted(missing)}")

    # Drop columns if present (ignore missing)
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["TIMESTAMP"]):
        df = df.copy()
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df.dropna(subset=["TIMESTAMP"])

    # Sort
    df = df.sort_values(["Track_ID", "TIMESTAMP"]).reset_index(drop=True)

    # (Optional) Remove duplicate TIMESTAMP within Track_ID: keep first row
    df = df.drop_duplicates(subset=["Track_ID", "TIMESTAMP"], keep="first").reset_index(drop=True)

    # 1) Track length filter (minimum length based on original)
    n0 = df.groupby("Track_ID")["TIMESTAMP"].transform("size")
    df = df[n0 >= min_track_len].copy()

    if df.empty:
        return df.assign(route_id=pd.Series(dtype="int64"), route_point_idx=pd.Series(dtype="int64"))

    # 2) Remove front/back trim (recalculate based on post-filter length)
    i = df.groupby("Track_ID").cumcount()
    n = df.groupby("Track_ID")["TIMESTAMP"].transform("size")
    df = df[(i >= trim) & (i < (n - trim))].copy()

    if df.empty:
        return df.assign(route_id=pd.Series(dtype="int64"), route_point_idx=pd.Series(dtype="int64"))

    # 3) Create window-sized segments
    i2 = df.groupby("Track_ID").cumcount()
    df["_seg"] = (i2 // window).astype("int64")
    df["route_point_idx"] = (i2 % window).astype("int64")

    # 4) Keep only segments filled with window points
    seg_n = df.groupby(["Track_ID", "_seg"])["TIMESTAMP"].transform("size")
    df = df[seg_n == window].copy()

    if df.empty:
        df = df.drop(columns=["_seg"], errors="ignore")
        return df.assign(route_id=pd.Series(dtype="int64"), route_point_idx=pd.Series(dtype="int64"))

    # 5) Assign route_id (Track_ID-Index format)
    df["route_id"] = df["Track_ID"].astype(str) + "-" + df["_seg"].astype(str)

    # Final sort for stability (Track_ID, TIMESTAMP)
    df = df.sort_values(["Track_ID", "TIMESTAMP"]).reset_index(drop=True)

    # Clean internal columns
    df = df.drop(columns=["_seg"])
    return _finalize_output_schema(df)


def validate_routes_long(df: pd.DataFrame, *, window: int) -> None:
    if df.empty:
        print("[INFO] Result is empty.")
        return

    rid_col = "ROUTE_ID" if "ROUTE_ID" in df.columns else "route_id"
    rpi_col = "ROUTE_POINT_IDX" if "ROUTE_POINT_IDX" in df.columns else "route_point_idx"
    ts_col = "TIMESTAMP"

    for col in (rid_col, rpi_col, ts_col):
        if col not in df.columns:
            raise ValueError(f"Missing columns required for validation: {col}")

    # Exactly window rows per route_id
    counts = df.groupby(rid_col).size()
    bad_counts = counts[counts != window]
    if not bad_counts.empty:
        raise ValueError(
            f"Row count per route_id differs from window({window})와 다릅니다. 예: {bad_counts.head(5).to_dict()}"
        )

    # route_point_idx exactly contains 0..window-1
    idx_sets = df.groupby(rid_col)[rpi_col].agg(lambda s: tuple(sorted(set(s.tolist()))))
    expected = tuple(range(window))
    bad_idx = idx_sets[idx_sets != expected]
    if not bad_idx.empty:
        raise ValueError(
            f"route_point_idx per route_id does not exactly contain 0..{window-1}를 정확히 포함하지 않습니다. "
            f"예: {bad_idx.head(5).to_dict()}"
        )

    # Verify TIMESTAMP sort (ascending within each route_id)
    ts_ok = df.groupby(rid_col)[ts_col].apply(lambda s: s.is_monotonic_increasing).all()
    if not bool(ts_ok):
        raise ValueError("TIMESTAMP is not in ascending order in some route_ids.")

    print(f"[OK] validate: routes={df[rid_col].nunique():,}, rows={len(df):,}, window={window}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] read_csv: {in_path}")
    df = pd.read_csv(in_path, parse_dates=["TIMESTAMP"])

    print(
        "[INFO] build_routes_long:",
        f"rows={len(df):,}",
        f"window={args.window}",
        f"trim={args.trim}",
        f"min_track_len={args.min_track_len}",
        f"drop_cols={args.drop_cols}",
    )
    out_df = build_routes_long(
        df,
        window=args.window,
        trim=args.trim,
        min_track_len=args.min_track_len,
        drop_cols=args.drop_cols,
    )

    validate_routes_long(out_df, window=args.window)

    print(f"[INFO] write_csv: {out_path}")
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] done: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

