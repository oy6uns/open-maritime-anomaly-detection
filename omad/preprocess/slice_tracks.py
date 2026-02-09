"""
Track_ID별 1시간 단위 포인트(롱 포맷) CSV를 10개 TIMESTAMP 단위의 route로 분할합니다.

요구사항
- Track_ID별로 TIMESTAMP 오름차순 정렬
- 각 Track에서 앞/뒤 trim개 포인트 제거
- 남은 구간을 window개 단위로 겹치지 않게 분할
- 10개로 꽉 찬 route만 유지(부분 조각 버림)
- route_id(전역 고유 ID), route_point_idx(0..window-1) 부여
- TYPE, MONTH 컬럼은 결과에서 제거(기본)
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
        help="입력 CSV 경로 (예: Local/outputs/df_filtered.csv)",
    )
    p.add_argument(
        "--output",
        required=True,
        help="출력 CSV 경로 (예: Local/outputs/routes_long.csv)",
    )
    p.add_argument("--window", type=int, default=10, help="route 당 TIMESTAMP 개수 (기본: 10)")
    p.add_argument("--trim", type=int, default=0, help="각 Track의 앞/뒤에서 제거할 포인트 수 (기본: 0)")
    p.add_argument(
        "--min_track_len",
        type=int,
        default=30,
        help="원본 Track_ID별 최소 TIMESTAMP 개수 (기본: 30)",
    )
    p.add_argument(
        "--drop_cols",
        nargs="*",
        default=["TYPE", "MONTH"],
        help='결과에서 제거할 컬럼들 (기본: "TYPE" "MONTH")',
    )
    return p.parse_args(argv)

def _finalize_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 컬럼명: 전부 대문자
    - 컬럼 순서: 사용자 지정 순서로 재정렬(존재하는 컬럼만)
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
        raise ValueError(f"출력 스키마에 필요한 컬럼이 없습니다: {missing}")

    # 사용자 요청 스키마만 남김(그 외 컬럼은 제거)
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
        raise ValueError("window는 양수여야 합니다.")
    if trim < 0:
        raise ValueError("trim은 0 이상이어야 합니다.")
    if min_track_len < 0:
        raise ValueError("min_track_len은 0 이상이어야 합니다.")

    required = {"Track_ID", "TIMESTAMP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"입력 df에 필수 컬럼이 없습니다: {sorted(missing)}")

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

    # (선택) 동일 Track_ID 내 TIMESTAMP 중복 제거: 첫 행 유지
    df = df.drop_duplicates(subset=["Track_ID", "TIMESTAMP"], keep="first").reset_index(drop=True)

    # 1) Track 길이 필터 (원본 기준 최소 길이)
    n0 = df.groupby("Track_ID")["TIMESTAMP"].transform("size")
    df = df[n0 >= min_track_len].copy()

    if df.empty:
        return df.assign(route_id=pd.Series(dtype="int64"), route_point_idx=pd.Series(dtype="int64"))

    # 2) 앞/뒤 trim 제거 (필터 이후 길이 기준으로 다시 계산)
    i = df.groupby("Track_ID").cumcount()
    n = df.groupby("Track_ID")["TIMESTAMP"].transform("size")
    df = df[(i >= trim) & (i < (n - trim))].copy()

    if df.empty:
        return df.assign(route_id=pd.Series(dtype="int64"), route_point_idx=pd.Series(dtype="int64"))

    # 3) window 단위 세그먼트 생성
    i2 = df.groupby("Track_ID").cumcount()
    df["_seg"] = (i2 // window).astype("int64")
    df["route_point_idx"] = (i2 % window).astype("int64")

    # 4) window로 꽉 찬 세그먼트만 유지
    seg_n = df.groupby(["Track_ID", "_seg"])["TIMESTAMP"].transform("size")
    df = df[seg_n == window].copy()

    if df.empty:
        df = df.drop(columns=["_seg"], errors="ignore")
        return df.assign(route_id=pd.Series(dtype="int64"), route_point_idx=pd.Series(dtype="int64"))

    # 5) route_id 부여 (Track_ID-Index 형태)
    df["route_id"] = df["Track_ID"].astype(str) + "-" + df["_seg"].astype(str)

    # Final sort for stability (Track_ID, TIMESTAMP)
    df = df.sort_values(["Track_ID", "TIMESTAMP"]).reset_index(drop=True)

    # Clean internal columns
    df = df.drop(columns=["_seg"])
    return _finalize_output_schema(df)


def validate_routes_long(df: pd.DataFrame, *, window: int) -> None:
    if df.empty:
        print("[INFO] 결과가 비어있습니다.")
        return

    rid_col = "ROUTE_ID" if "ROUTE_ID" in df.columns else "route_id"
    rpi_col = "ROUTE_POINT_IDX" if "ROUTE_POINT_IDX" in df.columns else "route_point_idx"
    ts_col = "TIMESTAMP"

    for col in (rid_col, rpi_col, ts_col):
        if col not in df.columns:
            raise ValueError(f"검증에 필요한 컬럼이 없습니다: {col}")

    # route_id 당 정확히 window개 행
    counts = df.groupby(rid_col).size()
    bad_counts = counts[counts != window]
    if not bad_counts.empty:
        raise ValueError(
            f"route_id별 행 수가 window({window})와 다릅니다. 예: {bad_counts.head(5).to_dict()}"
        )

    # route_point_idx가 0..window-1를 정확히 포함
    idx_sets = df.groupby(rid_col)[rpi_col].agg(lambda s: tuple(sorted(set(s.tolist()))))
    expected = tuple(range(window))
    bad_idx = idx_sets[idx_sets != expected]
    if not bad_idx.empty:
        raise ValueError(
            f"route_id별 route_point_idx가 0..{window-1}를 정확히 포함하지 않습니다. "
            f"예: {bad_idx.head(5).to_dict()}"
        )

    # TIMESTAMP 정렬 확인(각 route_id 내에서 오름차순)
    ts_ok = df.groupby(rid_col)[ts_col].apply(lambda s: s.is_monotonic_increasing).all()
    if not bool(ts_ok):
        raise ValueError("일부 route_id에서 TIMESTAMP가 오름차순이 아닙니다.")

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

