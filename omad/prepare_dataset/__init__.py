"""Prepare Dataset: NPZ dataset preparation"""

from pathlib import Path
import numpy as np

from omad.prepare_dataset.prepare_xy import build_xy, stratified_split_indices, load_anomaly_route_ids

def run_prepare_dataset_batch(
    base_dir: str = ".",
    out_dir: str | None = None,
    ratios: list[int] | None = None,
    slices: list[int] | None = None,
    modes: list[str] | None = None,
    seeds: list[int] | None = None,
    train: float = 0.7,
    valid: float = 0.15,
    test: float = 0.15,
    verbose: bool = False,
):
    """
    Batch mode: Generate all dataset combinations.

    Args:
        base_dir: Base directory
        out_dir: Output directory
        ratios: List of anomaly ratios (e.g., [10, 5, 3, 1])
        slices: List of T values
        modes: List of anomaly modes
        seeds: List of random seeds
        train: Train split ratio
        valid: Validation split ratio
        test: Test split ratio
        verbose: Detailed logging
    """
    from omad.utils.logging import console, log_stage_header, create_progress, create_summary_table, log_success

    log_stage_header(4, "Dataset Preparation (Batch Mode)")

    if ratios is None:
        ratios = [10, 5, 3, 1]
    if slices is None:
        slices = [12, 24]
    if modes is None:
        modes = ["a1", "a2", "a3"]
    if seeds is None:
        seeds = [2, 12, 32, 42, 52]

    # Convert ratios to percentage strings for internal use
    percentages = [f"{r}pct" for r in ratios]

    base_path = Path(base_dir)
    out_path = Path(out_dir) if out_dir else (base_path / "data" / "dataset")
    out_path.mkdir(parents=True, exist_ok=True)

    total_combinations = len(percentages) * len(slices) * len(modes) * len(seeds)

    console.print(f"[bold]Generating {total_combinations} dataset combinations[/bold]")
    console.print(f"  Ratios: {ratios}")
    console.print(f"  Slices: {slices}")
    console.print(f"  Modes:  {modes}")
    console.print(f"  Seeds:  {seeds}")
    console.print(f"  Split:  train={train}, valid={valid}, test={test}")
    console.print(f"  Output: {out_path}\n")

    stats = {"total": 0, "success": 0, "skipped": 0, "failed": 0}

    with create_progress() as progress:
        task = progress.add_task(
            "[cyan]Processing combinations...",
            total=total_combinations
        )

        for pct in percentages:
            for T in slices:
                csv_path = base_path / "data" / f"route_sliced_{T}" / f"routes_sliced_{T}_injected.csv"

                if not csv_path.exists():
                    console.print(f"[yellow]SKIP:[/yellow] {csv_path} not found")
                    stats["skipped"] += len(modes) * len(seeds)
                    progress.update(task, advance=len(modes) * len(seeds))
                    continue

                for mode in modes:
                    indices_path = base_path / "data" / f"route_sliced_{T}" / f"indices/indices_{T}_{mode}.csv"

                    if not indices_path.exists():
                        console.print(f"[yellow]SKIP:[/yellow] {indices_path} not found")
                        stats["skipped"] += len(seeds)
                        progress.update(task, advance=len(seeds))
                        continue

                    # Load data once per (pct, T, mode)
                    anomaly_route_ids = load_anomaly_route_ids(str(indices_path), pct, mode)
                    X_flat, y, route_ids, X_seq = build_xy(
                        str(csv_path), mode=mode, T=T, anomaly_route_ids=anomaly_route_ids
                    )

                    for seed in seeds:
                        try:
                            subdir = out_path / pct / f"slice_{T}_{mode}" / f"seed_{seed}"
                            subdir.mkdir(parents=True, exist_ok=True)

                            tr, va, te = stratified_split_indices(
                                y, train=train, valid=valid, test=test, seed=seed
                            )

                            for split_name, idx in [("train", tr), ("valid", va), ("test", te)]:
                                np.savez_compressed(
                                    subdir / f"{split_name}.npz",
                                    X=X_flat[idx],
                                    X_seq=X_seq[idx],
                                    y=y[idx],
                                    route_ids=route_ids[idx],
                                )

                            stats["success"] += 1

                        except Exception as e:
                            console.print(f"[red]ERROR:[/red] {pct}/{T}/{mode}/seed_{seed}: {e}")
                            stats["failed"] += 1

                        finally:
                            stats["total"] += 1
                            progress.update(task, advance=1)

    summary = create_summary_table("Prepare Dataset Summary", stats, title_style="bold green")
    console.print("\n")
    console.print(summary)
    console.print(f"\n✓ Datasets saved to: {out_path}\n")
    log_success("Prepare Dataset completed successfully")


def run_prepare_dataset_single(
    csv: str,
    mode: str,
    T: int,
    ratio: int,
    seed: int,
    output_dir: str,
    prefix: str | None = None,
    train: float = 0.7,
    valid: float = 0.15,
    test: float = 0.15,
    verbose: bool = False,
):
    """
    Single mode: Generate one dataset configuration.
    """
    from omad.utils.logging import console, log_stage_header, log_success

    log_stage_header(4, "Dataset Preparation (Single Mode)")

    percentage = f"{ratio}pct"
    console.print(f"Generating dataset: {percentage}/{T}/{mode}/seed_{seed}")

    log_success("Prepare Dataset completed successfully")
