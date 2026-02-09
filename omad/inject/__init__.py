"""Inject: Anomaly Injection"""

from pathlib import Path

def run_inject(
    T: int = 12,
    input_csv: str | None = None,
    track_csv: str | None = None,
    injected_dir: str | None = None,
    output_csv: str | None = None,
    seed: int = 0,
    theta_g: float = 3.0,
    theta_v: float = 2.0,
    verbose: bool = False,
):
    """
    Inject: Inject anomalies into routes using LLM scores.
    
    Args:
        T: Route slice length (12/24/48/72)
        input_csv: Input preprocessed CSV
        track_csv: Full track CSV
        injected_dir: LLM JSON directory
        output_csv: Output injected CSV
        seed: Random seed
        theta_g: A1 SED multiplier
        theta_v: A2 speed/heading multiplier
        verbose: Detailed logging
    """
    from omad.inject.main import apply_injection_to_csv
    from omad.config import get_config
    from omad.paths import resolve_stage3_paths
    from omad.utils.logging import console, log_stage_header, log_config, create_summary_table, log_success
    import typer
    
    log_stage_header(3, "Anomaly Injection (A1/A2/A3)")
    
    config = get_config()
    paths = resolve_stage3_paths(T, config,
        input_csv=input_csv,
        track_csv=track_csv,
        injected_dir=injected_dir,
        output_csv=output_csv
    )
    
    # Display configuration
    log_config({
        "T (Slice Length)": T,
        "Input CSV": paths["input_csv"],
        "Track CSV": paths["track_csv"],
        "Injected Dir": paths["injected_dir"],
        "Output CSV": paths["output_csv"],
        "Seed": seed,
        "Theta G (A1)": theta_g,
        "Theta V (A2)": theta_v,
    })
    
    # Ensure output directory exists
    Path(paths["output_csv"]).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run injection
        stats = apply_injection_to_csv(
            input_csv=str(paths["input_csv"]),
            track_csv=str(paths["track_csv"]),
            injected_dir=str(paths["injected_dir"]),
            output_csv=str(paths["output_csv"]),
            seed=seed,
            theta_g=theta_g,
            theta_v=theta_v,
        )
        
        # Display summary
        summary = create_summary_table("Inject Summary", stats, title_style="bold green")
        console.print("\n")
        console.print(summary)
        console.print(f"\n✓ Output saved to: {paths['output_csv']}\n")
        log_success("Inject completed successfully")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)

# Alias for backward compatibility  
run_stage3 = run_inject
