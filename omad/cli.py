"""Main CLI entry point for OMAD with config file support and command aliases."""

import typer
from typing_extensions import Annotated
from pathlib import Path

app = typer.Typer(
    name="omad",
    help="Ocean Maritime Anomaly Detection - Pipeline CLI Tool",
    add_completion=False,
)


# Stage 1: Preprocessing
@app.command(name="preprocess")
def preprocess_command(
    T: Annotated[int, typer.Option(help="Time slice length (12/24/48/72)")] = 12,
    input_csv: Annotated[str | None, typer.Option(help="Input AIS CSV path")] = None,
    trim: Annotated[int, typer.Option(help="Points to trim from track ends and start")] = 0,
    min_track_len: Annotated[int, typer.Option(help="Minimum track length")] = 30,
    ratios: Annotated[str, typer.Option("--r", help="Anomaly ratios (e.g., '10,5,3,1')")] = "10,5,3,1",
    skip_stratification: Annotated[bool, typer.Option(help="Skip indices generation")] = False,
    skip_prompts: Annotated[bool, typer.Option(help="Skip prompt generation")] = False,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    config: Annotated[str | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Detailed logging")] = False,
):
    """Stage 1: Data preprocessing - route slicing, stratification, prompts."""
    from omad.preprocess import run_preprocess
    from omad.config_loader import load_yaml_config, get_default_config_path

    # Auto-detect config if not specified
    if config is None:
        default_path = get_default_config_path()
        if default_path.exists():
            config = str(default_path)

    # Load config file if available
    if config:
        cfg = load_yaml_config(config)
        # Merge global settings
        if 'T' in cfg and T == 12:  # Default value, use config
            T = cfg['T']
        if 'seed' in cfg and seed == 0:
            seed = cfg['seed']
        if 'verbose' in cfg and not verbose:
            verbose = cfg['verbose']

        # Merge preprocess-specific settings
        if 'preprocess' in cfg:
            prep_cfg = cfg['preprocess']
            if input_csv is None and 'input_csv' in prep_cfg:
                input_csv = prep_cfg['input_csv']
            if trim == 0 and 'trim' in prep_cfg:
                trim = prep_cfg['trim']
            if min_track_len == 30 and 'min_track_len' in prep_cfg:
                min_track_len = prep_cfg['min_track_len']
            if ratios == "10,5,3,1" and 'ratios' in prep_cfg:
                ratios = prep_cfg['ratios']
            if not skip_stratification and 'skip_stratification' in prep_cfg:
                skip_stratification = prep_cfg['skip_stratification']
            if not skip_prompts and 'skip_prompts' in prep_cfg:
                skip_prompts = prep_cfg['skip_prompts']

    run_preprocess(
        T=T,
        input_csv=input_csv,
        trim=trim,
        min_track_len=min_track_len,
        ratios=ratios,
        skip_stratification=skip_stratification,
        skip_prompts=skip_prompts,
        seed=seed,
        verbose=verbose,
    )


# Stage 2: LLM Scoring
@app.command(name="score")
def score_command(
    T: Annotated[int | None, typer.Option(help="Time slice (12/24) - auto-derives paths")] = None,
    batch_root: Annotated[str | None, typer.Option(help="Batch root with user_query/ subdirs")] = None,
    batch_dir: Annotated[str | None, typer.Option(help="Flat directory with .txt files")] = None,
    anomaly_type: Annotated[str | None, typer.Option(help="Force anomaly type (A1/A2)")] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Max LLM tokens")] = 256,
    max_retries: Annotated[int, typer.Option(help="Validation retries (-1=unlimited)")] = -1,
    output_dir: Annotated[str | None, typer.Option(help="JSON output directory")] = None,
    log_dir: Annotated[str | None, typer.Option(help="Log directory")] = None,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Interactive prompt mode")] = False,
    stdin: Annotated[bool, typer.Option("--stdin", help="Single query from stdin")] = False,
    config: Annotated[str | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Detailed logging")] = False,
):
    """Stage 2: Generate LLM anomaly suitability scores."""
    from omad.score import run_score
    from omad.config_loader import load_yaml_config, get_default_config_path

    # Auto-detect config if not specified
    if config is None:
        default_path = get_default_config_path()
        if default_path.exists():
            config = str(default_path)

    # Load and merge config if available
    if config:
        cfg = load_yaml_config(config)
        # Merge global settings
        if 'T' in cfg and T is None:
            T = cfg['T']
        if 'verbose' in cfg and not verbose:
            verbose = cfg['verbose']

        # Merge score-specific settings
        if 'score' in cfg:
            score_cfg = cfg['score']
            if batch_root is None and 'batch_root' in score_cfg:
                batch_root = score_cfg['batch_root']
            if batch_dir is None and 'batch_dir' in score_cfg:
                batch_dir = score_cfg['batch_dir']
            if anomaly_type is None and 'anomaly_type' in score_cfg:
                anomaly_type = score_cfg['anomaly_type']
            if max_new_tokens == 256 and 'max_new_tokens' in score_cfg:
                max_new_tokens = score_cfg['max_new_tokens']
            if max_retries == -1 and 'max_retries' in score_cfg:
                max_retries = score_cfg['max_retries']
            if output_dir is None and 'output_dir' in score_cfg:
                output_dir = score_cfg['output_dir']
            if log_dir is None and 'log_dir' in score_cfg:
                log_dir = score_cfg['log_dir']

    run_score(
        T=T,
        batch_root=batch_root,
        batch_dir=batch_dir,
        anomaly_type=anomaly_type,
        max_new_tokens=max_new_tokens,
        max_retries=max_retries,
        output_dir=output_dir,
        log_dir=log_dir,
        interactive=interactive,
        stdin_mode=stdin,
        verbose=verbose,
    )


# Stage 3: Anomaly Injection
@app.command(name="inject")
def inject_command(
    T: Annotated[int, typer.Option(help="Route slice length (12/24/48/72)")] = 12,
    input_csv: Annotated[str | None, typer.Option(help="Input preprocessed CSV")] = None,
    track_csv: Annotated[str | None, typer.Option(help="Full track CSV")] = None,
    injected_dir: Annotated[str | None, typer.Option(help="LLM JSON directory")] = None,
    output_csv: Annotated[str | None, typer.Option(help="Output injected CSV")] = None,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    theta_g: Annotated[float, typer.Option(help="A1 SED multiplier")] = 3.0,
    theta_v: Annotated[float, typer.Option(help="A2 speed/heading multiplier")] = 2.0,
    config: Annotated[str | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Detailed logging")] = False,
):
    """Stage 3: Inject anomalies into routes using LLM scores."""
    from omad.inject import run_inject
    from omad.config_loader import load_yaml_config, get_default_config_path

    # Auto-detect config if not specified
    if config is None:
        default_path = get_default_config_path()
        if default_path.exists():
            config = str(default_path)

    # Load and merge config if available
    if config:
        cfg = load_yaml_config(config)
        # Merge global settings
        if 'T' in cfg and T == 12:
            T = cfg['T']
        if 'seed' in cfg and seed == 0:
            seed = cfg['seed']
        if 'verbose' in cfg and not verbose:
            verbose = cfg['verbose']

        # Merge inject-specific settings
        if 'inject' in cfg:
            inject_cfg = cfg['inject']
            if input_csv is None and 'input_csv' in inject_cfg:
                input_csv = inject_cfg['input_csv']
            if track_csv is None and 'track_csv' in inject_cfg:
                track_csv = inject_cfg['track_csv']
            if injected_dir is None and 'injected_dir' in inject_cfg:
                injected_dir = inject_cfg['injected_dir']
            if output_csv is None and 'output_csv' in inject_cfg:
                output_csv = inject_cfg['output_csv']
            if theta_g == 3.0 and 'theta_g' in inject_cfg:
                theta_g = inject_cfg['theta_g']
            if theta_v == 2.0 and 'theta_v' in inject_cfg:
                theta_v = inject_cfg['theta_v']

    run_inject(
        T=T,
        input_csv=input_csv,
        track_csv=track_csv,
        injected_dir=injected_dir,
        output_csv=output_csv,
        seed=seed,
        theta_g=theta_g,
        theta_v=theta_v,
        verbose=verbose,
    )


# Stage 4: Dataset Preparation
@app.command(name="prepare-dataset")
def prepare_dataset_command(
    batch: Annotated[bool, typer.Option("--batch", help="Generate all combinations")] = False,
    base_dir: Annotated[str, typer.Option(help="Base directory")] = "/workspace/NAS/KRISO2026/shared_folder",
    out_dir: Annotated[str | None, typer.Option(help="Output directory")] = None,
    csv: Annotated[str | None, typer.Option(help="Input CSV (single mode)")] = None,
    mode: Annotated[str | None, typer.Option(help="Anomaly type (a1/a2/a3)")] = None,
    T: Annotated[int, typer.Option(help="Time slice length")] = 12,
    percentage: Annotated[str, typer.Option(help="Anomaly percentage")] = "10pct",
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    train: Annotated[float, typer.Option(help="Train ratio")] = 0.7,
    valid: Annotated[float, typer.Option(help="Valid ratio")] = 0.15,
    test: Annotated[float, typer.Option(help="Test ratio")] = 0.15,
    output_dir: Annotated[str | None, typer.Option(help="Output dir (single mode)")] = None,
    prefix: Annotated[str | None, typer.Option(help="File prefix (single mode)")] = None,
    config: Annotated[str | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Detailed logging")] = False,
):
    """Stage 4: Prepare final NPZ datasets for model training."""
    from omad.prepare_dataset import run_prepare_dataset_batch, run_prepare_dataset_single
    from omad.config_loader import load_yaml_config, get_default_config_path

    # Auto-detect config if not specified
    if config is None:
        default_path = get_default_config_path()
        if default_path.exists():
            config = str(default_path)

    # Load and merge config if available
    if config:
        cfg = load_yaml_config(config)
        # Merge global settings
        if 'T' in cfg and T == 12:
            T = cfg['T']
        if 'seed' in cfg and seed == 0:
            seed = cfg['seed']
        if 'verbose' in cfg and not verbose:
            verbose = cfg['verbose']

        # Merge prepare-dataset-specific settings
        if 'prepare-dataset' in cfg:
            prep_cfg = cfg['prepare-dataset']
            if base_dir == "/workspace/NAS/KRISO2026/shared_folder" and 'base_dir' in prep_cfg:
                base_dir = prep_cfg['base_dir']
            if out_dir is None and 'out_dir' in prep_cfg:
                out_dir = prep_cfg['out_dir']
            if percentage == "10pct" and 'percentage' in prep_cfg:
                percentage = prep_cfg['percentage']
            if train == 0.7 and 'train' in prep_cfg:
                train = prep_cfg['train']
            if valid == 0.15 and 'valid' in prep_cfg:
                valid = prep_cfg['valid']
            if test == 0.15 and 'test' in prep_cfg:
                test = prep_cfg['test']

    if batch:
        run_prepare_dataset_batch(
            base_dir=base_dir,
            out_dir=out_dir,
            verbose=verbose
        )
    else:
        if not csv or not mode:
            from omad.utils.logging import log_error
            log_error("--csv and --mode are required for single mode")
            raise typer.Exit(1)
        run_prepare_dataset_single(
            csv=csv,
            mode=mode,
            T=T,
            percentage=percentage,
            seed=seed,
            output_dir=output_dir or ".",
            prefix=prefix,
            train=train,
            valid=valid,
            test=test,
            verbose=verbose
        )


# Pipeline command
@app.command()
def pipeline(
    T: Annotated[int, typer.Option(help="Time slice length")] = 12,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    percentage: Annotated[str, typer.Option(help="Anomaly percentage")] = "10pct",
    skip_stage1: Annotated[bool, typer.Option(help="Skip preprocessing (use existing)")] = True,
    skip_stage2: Annotated[bool, typer.Option(help="Skip LLM scoring (use existing)")] = False,
    theta_g: Annotated[float, typer.Option(help="A1 multiplier")] = 3.0,
    theta_v: Annotated[float, typer.Option(help="A2 multiplier")] = 2.0,
    config: Annotated[str | None, typer.Option("--config", "-c", help="Config file path")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Detailed logging")] = False,
):
    """Run complete pipeline: Stage2 → Stage3 → Stage4."""
    from omad.preprocess import run_preprocess
    from omad.score import run_score
    from omad.inject import run_inject
    from omad.prepare_dataset import run_prepare_dataset_single
    from omad.utils.logging import console, log_info
    from omad.config_loader import load_yaml_config, get_default_config_path

    # Auto-detect config if not specified
    if config is None:
        default_path = get_default_config_path()
        if default_path.exists():
            config = str(default_path)

    # Load and merge config if available
    if config:
        cfg = load_yaml_config(config)
        # Merge global settings
        if 'T' in cfg and T == 12:
            T = cfg['T']
        if 'seed' in cfg and seed == 0:
            seed = cfg['seed']
        if 'verbose' in cfg and not verbose:
            verbose = cfg['verbose']

        # Merge pipeline-specific settings
        if 'pipeline' in cfg:
            pipe_cfg = cfg['pipeline']
            if not skip_stage1 and 'skip_stage1' in pipe_cfg:
                skip_stage1 = pipe_cfg['skip_stage1']
            if not skip_stage2 and 'skip_stage2' in pipe_cfg:
                skip_stage2 = pipe_cfg['skip_stage2']
            if percentage == "10pct" and 'percentage' in pipe_cfg:
                percentage = pipe_cfg['percentage']
            if theta_g == 3.0 and 'theta_g' in pipe_cfg:
                theta_g = pipe_cfg['theta_g']
            if theta_v == 2.0 and 'theta_v' in pipe_cfg:
                theta_v = pipe_cfg['theta_v']

    console.print("\n[bold blue]═══════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]   OMAD Pipeline Execution[/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════[/bold blue]\n")

    try:
        # Stage 1: Preprocessing (usually run once)
        if not skip_stage1:
            log_info("[bold cyan]>>> Running Stage 1: Preprocessing[/bold cyan]")
            run_preprocess(T=T, seed=seed, verbose=verbose)
            console.print("\n")

        # Stage 2: LLM Scoring
        if not skip_stage2:
            log_info("[bold cyan]>>> Running Stage 2: LLM Scoring[/bold cyan]")
            run_score(T=T, verbose=verbose)
            console.print("\n")

        # Stage 3: Anomaly Injection
        log_info("[bold cyan]>>> Running Stage 3: Anomaly Injection[/bold cyan]")
        run_inject(T=T, seed=seed, theta_g=theta_g, theta_v=theta_v, verbose=verbose)
        console.print("\n")

        # Stage 4: Dataset Preparation (for all modes)
        log_info("[bold cyan]>>> Running Stage 4: Dataset Preparation[/bold cyan]")
        csv_path = f"/workspace/NAS/KRISO2026/route_sliced_{T}/routes_sliced_{T}_injected.csv"

        for mode in ["a1", "a2", "a3"]:
            console.print(f"\n[bold]Processing mode: {mode.upper()}[/bold]")
            run_prepare_dataset_single(
                csv=csv_path,
                mode=mode,
                T=T,
                percentage=percentage,
                seed=seed,
                output_dir=f"/workspace/NAS/KRISO2026/shared_folder/data/{percentage}/slice_{T}_{mode}/seed_{seed}",
                train=0.7,
                valid=0.15,
                test=0.15,
                verbose=verbose
            )

        # Success
        console.print("\n[bold blue]═══════════════════════════════════════════════[/bold blue]")
        console.print("[bold green]✓ Pipeline completed successfully![/bold green]")
        console.print("[bold blue]═══════════════════════════════════════════════[/bold blue]\n")

    except SystemExit:
        console.print("\n[bold red]✗ Pipeline failed[/bold red]\n")
        raise
    except Exception as e:
        console.print(f"\n[bold red]✗ Pipeline failed: {e}[/bold red]\n")
        if verbose:
            console.print_exception()
        raise SystemExit(1)


# Config management command
@app.command()
def config(
    init: Annotated[bool, typer.Option("--init", help="Create default config.yaml")] = False,
    show: Annotated[bool, typer.Option("--show", help="Show current config")] = False,
    path: Annotated[str | None, typer.Option(help="Config file path")] = None,
):
    """Manage OMAD configuration files."""
    from omad.config_loader import create_default_config, load_yaml_config, get_default_config_path
    from omad.utils.logging import console
    import yaml

    if init:
        output_path = path or "./config.yaml"
        create_default_config(output_path)
        console.print(f"\n[bold]Created config file:[/bold] {output_path}")
        console.print("\n[bold]Usage:[/bold]")
        console.print("  # Edit the config file, then run:")
        console.print("  omad preprocess --config config.yaml")
        console.print("  omad pipeline --config config.yaml\n")
    elif show:
        config_path = path or get_default_config_path()
        if not Path(config_path).exists():
            console.print(f"[red]Config file not found:[/red] {config_path}")
            console.print("\n[yellow]Tip:[/yellow] Create one with: omad config --init")
            raise typer.Exit(1)

        cfg = load_yaml_config(config_path)
        console.print(f"\n[bold]Configuration from:[/bold] {config_path}\n")
        console.print(yaml.dump(cfg, default_flow_style=False))
    else:
        console.print("[yellow]No action specified.[/yellow] Use --init or --show")
        console.print("\n[bold]Examples:[/bold]")
        console.print("  omad config --init              # Create default config.yaml")
        console.print("  omad config --init --path my.yaml  # Create config at custom path")
        console.print("  omad config --show              # Show current config")


@app.callback()
def callback():
    """
    OMAD - Ocean Maritime Anomaly Detection Pipeline CLI Tool

    Provides commands to run different stages of the maritime anomaly detection pipeline:

    Commands:
    - preprocess: Data preprocessing
    - score: LLM scoring
    - inject: Anomaly injection
    - prepare-dataset: Dataset preparation
    - pipeline: Run complete pipeline
    - config: Manage configuration files
    """
    pass


if __name__ == "__main__":
    app()
