"""Preprocess: Data Preprocessing"""

from pathlib import Path
import pandas as pd

from omad.preprocess.slice_tracks import build_routes_long, validate_routes_long
from omad.config import get_config
from omad.paths import resolve_stage1_paths
from omad.preprocess.stratification import generate_indices_csv
from omad.preprocess.prompt_generation import generate_prompts_for_routes
from omad.utils.logging import (
    console,
    log_stage_header,
    log_config,
    create_summary_table,
    log_success,
    log_error,
    log_info,
    log_warning,
)


def run_preprocess(
    T: int = 12,
    input_csv: str | None = None,
    trim: int = 12,
    min_track_len: int = 36,
    ratios: list[int] | None = None,
    skip_stratification: bool = False,
    skip_prompts: bool = False,
    seed: int = 0,
    verbose: bool = False,
):
    """
    Preprocess: Data preprocessing - route slicing, stratification, prompt generation.

    Args:
        T: Time slice length (12/24/48/72)
        input_csv: Input CSV path (long-format AIS data)
        trim: Points to trim from track ends and start
        min_track_len: Minimum track length filter
        ratios: Anomaly ratios list (e.g., [10, 5, 3, 1])
        skip_stratification: Skip indices generation (use existing)
        skip_prompts: Skip user_query generation (use existing)
        seed: Random seed
        verbose: Detailed logging
    """
    log_stage_header(1, "Data Preprocessing")

    config = get_config()
    paths = resolve_stage1_paths(T, config, output_dir=None)

    # Default ratios
    ratio_list = ratios if ratios is not None else [10, 5, 3, 1]

    # Display configuration
    log_config({
        "Slice": T,
        "Input CSV": input_csv or "Auto-detect from omtad.csv",
        "Output Dir": paths["output_dir"],
        "Trim": trim,
        "Min Track Len": min_track_len,
        "Anomaly Ratios": ratio_list,
        "Seed": seed,
        "Skip Stratification": skip_stratification,
        "Skip Prompts": skip_prompts,
    })

    stats = {
        "routes_sliced": 0,
        "indices_generated": 0,
        "prompts_generated": 0,
    }

    try:
        # ==================================================================
        # Step 1: Route Slicing
        # ==================================================================
        log_info("\n[bold]Step 1:[/bold] Route slicing")

        # If input_csv is not set, try to find omtad.csv
        if input_csv is None:
            possible_paths = [
                "./data/omtad.csv",
                "./omtad.csv",
                paths["output_dir"] / "omtad.csv",
            ]
            for p in possible_paths:
                if Path(p).exists():
                    input_csv = str(p)
                    break

        # If input_csv is set but file doesn't exist, auto-generate from raw AIS data
        if input_csv is not None and not Path(input_csv).exists():
            log_info(f"  {input_csv} not found. Generating from raw AIS data...")
            from omad.preprocess.loader import prepare_data_pipeline
            df_loaded = prepare_data_pipeline()
            output_csv_path = Path(input_csv)
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_loaded.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
            console.print(f"  ✓ Generated: {output_csv_path} ({len(df_loaded):,} rows)")
        elif input_csv is None:
            # No path specified and no file found - try to generate at default location
            log_info("  omtad.csv not found. Loading raw AIS data from ./data/ ...")
            from omad.preprocess.loader import prepare_data_pipeline
            df_loaded = prepare_data_pipeline()
            output_csv_path = Path("./data/omtad.csv")
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_loaded.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
            console.print(f"  ✓ Generated: {output_csv_path} ({len(df_loaded):,} rows)")
            input_csv = str(output_csv_path)

        console.print(f"  Reading: {input_csv}")
        df = pd.read_csv(input_csv)

        # If TIMESTAMP is string, convert to datetime
        if "TIMESTAMP" in df.columns and df["TIMESTAMP"].dtype == object:
            df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")

        console.print(f"  Input rows: {len(df):,}")

        # Build routes
        console.print(f"\n  Building routes (T={T}, trim={trim}, min_track_len={min_track_len})...")
        routes_df = build_routes_long(
            df,
            window=T,
            trim=trim,
            min_track_len=min_track_len,
            drop_cols=["TYPE", "MONTH"],
        )

        console.print(f"  Output rows: {len(routes_df):,}")
        console.print(f"  Routes: {routes_df['ROUTE_ID'].nunique():,}")

        # Validate
        validate_routes_long(routes_df, window=T)

        # Save raw routes
        raw_routes_path = paths["raw_routes_csv"]
        raw_routes_path.parent.mkdir(parents=True, exist_ok=True)
        routes_df.to_csv(raw_routes_path, index=False)
        console.print(f"\n  ✓ Saved raw routes: {raw_routes_path}")

        # Save preprocessed routes (same as raw for now)
        preprocessed_path = paths["preprocessed_csv"]
        routes_df.to_csv(preprocessed_path, index=False)
        console.print(f"  ✓ Saved preprocessed routes: {preprocessed_path}")

        stats["routes_sliced"] = routes_df["ROUTE_ID"].nunique()

        # ==================================================================
        # Step 2: Stratification (Generate indices CSV)
        # ==================================================================
        if not skip_stratification:
            log_info("\n[bold]Step 2:[/bold] Generating stratification indices")

            indices_dir = paths["indices_dir"]
            indices_dir.mkdir(parents=True, exist_ok=True)

            for mode in ["a1", "a2", "a3"]:
                console.print(f"\n  Generating indices for {mode.upper()}...")
                indices_path = indices_dir / f"indices_{T}_{mode}.csv"

                indices_df = generate_indices_csv(
                    routes_df=routes_df,
                    T=T,
                    mode=mode,
                    output_path=indices_path,
                    seed=seed,
                    n_bins=5,
                    percentages=ratio_list,
                )

                console.print(f"    Total routes: {len(indices_df):,}")
                for pct in ratio_list:
                    count = indices_df[f"use_{pct}pct"].sum()
                    console.print(f"    {pct}pct: {count:,} routes")

                console.print(f"    ✓ Saved: {indices_path}")
                stats["indices_generated"] += 1

        else:
            log_warning("  Skipping stratification (using existing indices)")

        # ==================================================================
        # Step 3: Prompt Generation
        # ==================================================================
        if not skip_prompts:
            log_info("\n[bold]Step 3:[/bold] Generating LLM prompts")

            user_query_dir = paths["user_query_dir"]
            user_query_dir.mkdir(parents=True, exist_ok=True)

            console.print(f"\n  Output directory: {user_query_dir}")

            prompt_stats = generate_prompts_for_routes(
                routes_df=routes_df,
                indices_dir=paths["indices_dir"],
                output_dir=user_query_dir,
                T=T,
                vessel_type="Cargo",
            )

            console.print(f"\n  ✓ Generated {prompt_stats['total_prompts']:,} prompt files")
            console.print(f"    Routes processed: {prompt_stats['routes_processed']:,}")
            stats["prompts_generated"] = prompt_stats["total_prompts"]

        else:
            log_warning("  Skipping prompt generation (using existing prompts)")

        # ==================================================================
        # Summary
        # ==================================================================
        console.print("\n")
        summary = create_summary_table("Preprocess Summary", stats, title_style="bold green")
        console.print(summary)
        console.print()
        log_success("Preprocess completed successfully")

    except FileNotFoundError as e:
        log_error(f"File not found: {e}")
        raise SystemExit(1)
    except Exception as e:
        log_error(f"Preprocess failed: {e}")
        if verbose:
            console.print_exception()
        raise SystemExit(1)


# Alias for backward compatibility
run_stage1 = run_preprocess
