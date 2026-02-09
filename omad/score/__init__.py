"""Score: LLM Anomaly Scoring"""

from omad.score.main import _batch_from_root, _batch_from_dir, _run_once, _run_interactive

def run_score(
    T: int | None = None,
    batch_root: str | None = None,
    batch_dir: str | None = None,
    anomaly_type: str | None = None,
    max_new_tokens: int = 256,
    max_retries: int = -1,
    output_dir: str | None = None,
    log_dir: str | None = None,
    interactive: bool = False,
    stdin_mode: bool = False,
    verbose: bool = False,
):
    """
    Score: Generate LLM anomaly suitability scores.
    
    Args:
        T: Time slice (12/24) - auto-derives paths
        batch_root: Batch root with user_query/ subdirs
        batch_dir: Flat directory with .txt files
        anomaly_type: Force anomaly type (A1/A2)
        max_new_tokens: Max LLM tokens
        max_retries: Validation retries (-1=unlimited)
        output_dir: JSON output directory
        log_dir: Log directory
        interactive: Interactive prompt mode
        stdin_mode: Single query from stdin
        verbose: Detailed logging
    """
    from omad.config import get_config
    from omad.paths import resolve_stage2_paths
    from omad.utils.logging import console, log_stage_header, log_config, create_summary_table, log_success
    import typer
    
    log_stage_header(2, "LLM Anomaly Scoring")
    
    config = get_config()
    
    # Resolve paths
    if T is not None and not batch_root:
        paths = resolve_stage2_paths(T, config, output_dir=output_dir, log_dir=log_dir)
        batch_root = str(paths["batch_root"])
        output_dir = str(paths["output_dir"])
        log_dir = str(paths["log_dir"])
    
    # Display configuration
    log_config({
        "Batch Root": batch_root or "N/A",
        "Batch Dir": batch_dir or "N/A",
        "Anomaly Type": anomaly_type or "Auto-detect",
        "Max Tokens": max_new_tokens,
        "Max Retries": max_retries if max_retries >= 0 else "Unlimited",
        "Output Dir": output_dir,
        "Log Dir": log_dir,
    })
    
    # Execute based on mode
    try:
        if interactive:
            _run_interactive(anomaly_type, max_new_tokens, max_retries)
        elif stdin_mode:
            _run_once(anomaly_type, max_new_tokens, max_retries)
        elif batch_root:
            stats = _batch_from_root(
                batch_root, anomaly_type, max_new_tokens, max_retries,
                output_dir, log_dir
            )
            summary = create_summary_table("Score Summary", stats)
            console.print(summary)
        elif batch_dir:
            stats = _batch_from_dir(
                batch_dir, anomaly_type, max_new_tokens, max_retries,
                output_dir, log_dir
            )
            summary = create_summary_table("Score Summary", stats)
            console.print(summary)
        else:
            console.print("[red]Error:[/red] Must specify --T, --batch-root, or --batch-dir")
            raise typer.Exit(1)
        
        log_success("Score completed successfully\n")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

# Alias for backward compatibility
run_stage2 = run_score
