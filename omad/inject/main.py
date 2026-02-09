"""
KRISO2026 - Anomaly Injection Main
===================================
Apply A1/A2/A3 anomaly injection to route CSV using LLM JSON outputs.

Usage:
    python main.py --slice 12
    python main.py --input-csv /path/to/input.csv --injected-dir /path/to/json --output-csv /path/to/output.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from a1_injector import inject_a1
from a2_injector import inject_a2
from a3_injector import inject_a3_virtual_rows, load_a3_plans
from inject_utils import build_track_index, iter_routes_csv, load_plans


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inject A1/A2/A3 anomalies into KRISO route CSV using LLM JSON outputs."
    )
    p.add_argument(
        "--slice",
        type=int,
        default=12,
        help="Route slice length (e.g., 12, 24). Used to auto-derive NAS paths unless overridden.",
    )
    p.add_argument(
        "--input-csv",
        default=None,
        help="Input routes_sliced_*_preprocessed.csv",
    )
    p.add_argument(
        "--track-csv",
        default=None,
        help="CSV with full tracks for boundary lookup (e.g., routes_sliced_*.csv)",
    )
    p.add_argument(
        "--injected-dir",
        default=None,
        help="Directory containing qwen_output_route_*_{A1|A2|A3}_*.json",
    )
    p.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for injection noise/signs.",
    )
    p.add_argument(
        "--theta-g",
        type=float,
        default=3.0,
        help="A1 SED multiplier (default: 3.0)",
    )
    p.add_argument(
        "--theta-v",
        type=float,
        default=2.0,
        help="A2 speed/heading multiplier (default: 3.0)",
    )
    return p.parse_args()


def apply_injection_to_csv(
    *,
    input_csv: str,
    injected_dir: str,
    output_csv: str,
    seed: int = 0,
    theta_g: float = 3.0,
    theta_v: float = 2.0,
    track_csv: Optional[str] = None,
) -> Dict[str, int]:
    """
    Apply A1/A2/A3 injection plans and write a new CSV.

    Adds:
      - T_INDEX: 0..T-1 per route
      - ANOMALY_TYPE: A1/A2/A3 on anomaly rows, else ""

    Returns:
        Statistics dictionary
    """
    rng = random.Random(seed)

    # Load injection plans
    plans = load_plans(injected_dir)
    print(f"Loaded {len(plans)} A1/A2 plans")

    # Build track index for boundary-safe injection
    track_source = track_csv or input_csv
    track_index, track_pos = build_track_index(track_source)

    # Load A3 plans
    base_dir = os.path.dirname(injected_dir.rstrip("/"))
    a3_fallback = os.path.join(base_dir, "A3_before")
    a3_plans = load_a3_plans(injected_dir, fallback_dir=a3_fallback)
    if not a3_plans:
        a3_plans = load_a3_plans("/workspace/NAS/KRISO2026/A3_before")
    print(f"Loaded {len(a3_plans)} A3 plans")

    stats = {
        "routes_total": 0,
        "routes_a1_injected": 0,
        "routes_a2_injected": 0,
        "routes_a3_generated": 0,
        "rows_written": 0,
        "a3_rows_written": 0,
    }

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    # Progress tracking
    start_time = time.time()
    last_print_time = start_time
    print_interval = 5.0  # Print every 5 seconds

    with open(output_csv, "w", encoding="utf-8", newline="") as out_f:
        writer: Optional[csv.DictWriter] = None
        out_fieldnames: Optional[List[str]] = None

        for route_id, rows, fieldnames in iter_routes_csv(input_csv):
            stats["routes_total"] += 1

            # Progress reporting
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                elapsed = current_time - start_time
                rate = stats["routes_total"] / elapsed if elapsed > 0 else 0
                print(
                    f"\r[Progress] Routes: {stats['routes_total']:,} | "
                    f"A1: {stats['routes_a1_injected']:,} | "
                    f"A2: {stats['routes_a2_injected']:,} | "
                    f"A3: {stats['routes_a3_generated']:,} | "
                    f"Rate: {rate:.1f} routes/sec | "
                    f"Elapsed: {elapsed:.0f}s",
                    end="",
                    flush=True,
                )
                last_print_time = current_time

            # Initialize CSV writer with extended fieldnames
            if writer is None:
                out_fieldnames = list(fieldnames)
                for col in ["ANOMALY", "T_INDEX", "ANOMALY_TYPE"]:
                    if col not in out_fieldnames:
                        out_fieldnames.append(col)
                writer = csv.DictWriter(out_f, fieldnames=out_fieldnames)
                writer.writeheader()

            # Keep pristine copy for A3 virtual generation
            orig_rows_for_a3 = [dict(r) for r in rows]

            # Initialize per-row metadata
            for t, r in enumerate(rows):
                r.setdefault("ANOMALY", "False")
                r["T_INDEX"] = str(t)
                r.setdefault("ANOMALY_TYPE", "")

            # Check for A1/A2 plans (prefer A2 if both exist)
            plan = plans.get((route_id, "A2")) or plans.get((route_id, "A1"))

            if plan is not None:
                if plan.anomaly_type == "A1":
                    new_rows = inject_a1(
                        rows, plan, rng=rng, theta_g=theta_g,
                        track_index=track_index, track_pos=track_pos
                    )
                    stats["routes_a1_injected"] += 1
                else:
                    new_rows = inject_a2(
                        rows, plan, rng=rng, theta_v=theta_v,
                        track_index=track_index, track_pos=track_pos
                    )
                    stats["routes_a2_injected"] += 1

                # Set anomaly type label
                for r in new_rows:
                    is_anomaly = str(r.get("ANOMALY", "False")).strip().lower() == "true"
                    r["ANOMALY_TYPE"] = plan.anomaly_type if is_anomaly else ""
            else:
                new_rows = rows

            # A3: Generate virtual vessel rows if plan exists
            a3_plan = a3_plans.get(route_id)
            if a3_plan is not None and len(a3_plan.scores) == len(orig_rows_for_a3):
                # Initialize metadata for A3 source rows
                for t, r in enumerate(orig_rows_for_a3):
                    r.setdefault("ANOMALY", "False")
                    r["T_INDEX"] = str(t)
                    r.setdefault("ANOMALY_TYPE", "")

                virtual_rows = inject_a3_virtual_rows(
                    orig_rows_for_a3,
                    route_id=route_id,
                    scores=a3_plan.scores,
                    seed=seed,
                )

                if virtual_rows:
                    # Mark only indices that are anomalous in generated (virtual) rows
                    a3_indices = {
                        i for i, vr in enumerate(virtual_rows)
                        if str(vr.get("ANOMALY", "False")).strip().lower() == "true"
                    }
                    if a3_indices:
                        for i, r in enumerate(new_rows):
                            if i in a3_indices and not (r.get("ANOMALY_TYPE") or "").strip():
                                r["ANOMALY"] = "True"
                                r["ANOMALY_TYPE"] = "A3"

                    writer.writerows(virtual_rows)
                    stats["routes_a3_generated"] += 1
                    stats["a3_rows_written"] += len(virtual_rows)
                    stats["rows_written"] += len(virtual_rows)

            # Write original/modified rows
            writer.writerows(new_rows)
            stats["rows_written"] += len(new_rows)

    # Final progress
    elapsed = time.time() - start_time
    print(
        f"\r[Progress] Routes: {stats['routes_total']:,} | "
        f"A1: {stats['routes_a1_injected']:,} | "
        f"A2: {stats['routes_a2_injected']:,} | "
        f"A3: {stats['routes_a3_generated']:,} | "
        f"Total time: {elapsed:.1f}s"
    )

    return stats


def main() -> int:
    args = _parse_args()

    # Derive paths from slice if not provided
    base_dir = f"/workspace/NAS/KRISO2026/route_sliced_{args.slice}"
    input_csv = args.input_csv or f"{base_dir}/routes_sliced_{args.slice}_preprocessed.csv"
    track_csv = args.track_csv or f"{base_dir}/routes_sliced_{args.slice}.csv"
    injected_dir = args.injected_dir or f"{base_dir}/injected"
    output_csv = args.output_csv or f"{base_dir}/routes_sliced_{args.slice}_injected.csv"

    print(f"Input CSV: {input_csv}")
    print(f"Track CSV: {track_csv}")
    print(f"Injected dir: {injected_dir}")
    print(f"Output CSV: {output_csv}")
    print(f"Seed: {args.seed}")
    print(f"theta_g (A1): {args.theta_g}")
    print(f"theta_v (A2): {args.theta_v}")
    print()

    # Create output directory
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Run injection
    stats = apply_injection_to_csv(
        input_csv=input_csv,
        injected_dir=injected_dir,
        output_csv=output_csv,
        seed=args.seed,
        theta_g=args.theta_g,
        theta_v=args.theta_v,
        track_csv=track_csv,
    )

    print()
    print("[DONE]")
    print("-" * 40)
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("-" * 40)
    print(f"Output: {output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
