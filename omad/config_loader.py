"""YAML configuration loader for OMAD."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from omad.utils.logging import console, log_info, log_warning


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary of configuration values
    """
    config_path = Path(config_path)

    if not config_path.exists():
        log_warning(f"Config file not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    log_info(f"Loaded configuration from: {config_path}")
    return config or {}


def merge_config_with_cli(yaml_config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge YAML config with CLI arguments (CLI takes precedence).

    Args:
        yaml_config: Configuration from YAML file
        cli_args: Arguments from command line

    Returns:
        Merged configuration dictionary
    """
    # Start with YAML config
    merged = yaml_config.copy()

    # Override with CLI args (only if not None or default)
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value

    return merged


def resolve_paths(config: Dict[str, Any], base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Resolve relative paths in configuration to absolute paths.

    Args:
        config: Configuration dictionary
        base_dir: Base directory for relative paths (default: current working directory)

    Returns:
        Configuration with resolved paths
    """
    if base_dir is None:
        base_dir = Path.cwd()

    resolved = config.copy()

    # List of path keys that should be resolved
    path_keys = [
        "input_csv", "output_dir", "batch_root", "batch_dir",
        "injected_dir", "output_csv", "track_csv", "base_dir",
        "csv", "output_dir", "log_dir"
    ]

    for key in path_keys:
        if key in resolved and resolved[key] is not None:
            path = Path(resolved[key])
            if not path.is_absolute():
                resolved[key] = str(base_dir / path)

    return resolved


def get_default_config_path() -> Path:
    """
    Get default configuration file path.

    Searches for config.yaml in:
    1. Current directory
    2. ~/.omad/config.yaml
    3. Project directory

    Returns:
        Path to default config file (may not exist)
    """
    search_paths = [
        Path.cwd() / "config.yaml",
        Path.cwd() / "omad.yaml",
        Path.home() / ".omad" / "config.yaml",
        Path(__file__).parent.parent / "config.yaml",
    ]

    for path in search_paths:
        if path.exists():
            return path

    # Return first path as default (even if it doesn't exist)
    return search_paths[0]


def create_default_config(output_path: str | Path):
    """
    Create a default configuration file with all available options.

    Args:
        output_path: Path where to save the configuration file
    """
    config_content = """# ════════════════════════════════════════════════════════════════════════════
# OMAD Configuration File
# Ocean Maritime Anomaly Detection - Pipeline Configuration
# ════════════════════════════════════════════════════════════════════════════
# All paths can be absolute or relative to this config file

# ────────────────────────────────────────────────────────────────────────────
# Global Settings
# ────────────────────────────────────────────────────────────────────────────
T: 12                    # Time slice length in hours (12/24/48/72)
                         # Routes are divided into segments of this duration
                         # Example: T=12 means each route segment is 12 hours long

seed: 0                  # Random seed for reproducibility
                         # Used in: stratification, injection, dataset splitting
                         # Same seed ensures identical results across runs

verbose: false           # Enable detailed logging output
                         # true = show detailed debug information
                         # false = show only essential messages

# ────────────────────────────────────────────────────────────────────────────
# Stage 1: Preprocess
# ────────────────────────────────────────────────────────────────────────────
preprocess:
  input_csv: ./data/omtad.csv  # Input AIS CSV file path
                               # Path to your raw AIS trajectory data

  trim: 0                # Number of points to trim from track ends and start
                         # Removes unstable/noisy data at track boundaries
                         # 0 = no trimming

  min_track_len: 30      # Minimum track length filter (in points)
                         # Tracks shorter than this are discarded
                         # Ensures sufficient data for route segmentation

  ratios: "10,5,3,1"     # Anomaly injection percentage ratios
                         # Creates stratified samples at these percentages
                         # Example: "10,5,3,1" → 10%, 5%, 3%, 1% anomaly rates

  skip_stratification: false  # Skip stratification indices generation
                              # true = use existing indices files
                              # false = regenerate stratification

  skip_prompts: false    # Skip LLM prompt generation
                         # true = use existing prompt files
                         # false = regenerate all prompts

# ────────────────────────────────────────────────────────────────────────────
# Stage 2: Score (LLM Anomaly Scoring)
# ────────────────────────────────────────────────────────────────────────────
score:
  batch_root: null       # Root directory containing user_query/ subdirectories
                         # null = auto-derived from T parameter
                         # Example: /workspace/NAS/KRISO2026/route_sliced_12

  batch_dir: null        # Flat directory with .txt prompt files
                         # Alternative to batch_root for custom layouts

  anomaly_type: null     # Force specific anomaly type (A1/A2/A3)
                         # null = auto-detect from file paths

  max_new_tokens: 256    # Maximum LLM generation tokens
                         # Controls response length from language model

  max_retries: -1        # JSON validation retry limit
                         # -1 = unlimited retries until valid JSON
                         # N = retry up to N times before failing

# ────────────────────────────────────────────────────────────────────────────
# Stage 3: Inject (Anomaly Injection)
# ────────────────────────────────────────────────────────────────────────────
inject:
  input_csv: null        # Input preprocessed CSV path
                         # null = auto-derived from T
                         # Example: route_sliced_12/routes_sliced_12_preprocessed.csv

  track_csv: null        # Full track CSV for boundary calculations
                         # null = auto-derived from T

  injected_dir: null     # Directory with LLM JSON score files
                         # null = auto-derived (route_sliced_{T}/injected/)
                         # Input from Stage 2 output

  theta_g: 3.0           # A1 anomaly multiplier (position displacement)
                         # Controls magnitude of cross-track errors
                         # Higher = more severe position anomalies

  theta_v: 2.0           # A2 anomaly multiplier (speed/heading changes)
                         # Controls magnitude of velocity anomalies
                         # Higher = more severe speed/heading deviations

# ────────────────────────────────────────────────────────────────────────────
# Stage 4: Prepare Dataset (NPZ Generation)
# ────────────────────────────────────────────────────────────────────────────
prepare-dataset:
  base_dir: .                # Base directory (project root)
                             # Uses route_sliced_{T}/ created by Stage 1-3

  percentage: "10pct"    # Anomaly percentage to use
                         # Options: "10pct", "5pct", "3pct", "1pct"

  train: 0.7             # Training set ratio (0.0-1.0)
  valid: 0.15            # Validation set ratio (0.0-1.0)
  test: 0.15             # Test set ratio (0.0-1.0)
                         # NOTE: train + valid + test must sum to 1.0

# ────────────────────────────────────────────────────────────────────────────
# Pipeline Settings (End-to-End Execution)
# ────────────────────────────────────────────────────────────────────────────
pipeline:
  skip_stage1: true      # Skip preprocessing stage
                         # true = use existing preprocessed files (typical)
                         # false = re-run data loading and route slicing

  skip_stage2: false     # Skip LLM scoring stage
                         # true = use existing JSON score files
                         # false = re-run LLM inference (slow, requires GPU)

  percentage: "10pct"    # Default percentage for dataset generation
  theta_g: 3.0           # A1 multiplier for pipeline injection
  theta_v: 2.0           # A2 multiplier for pipeline injection

# ════════════════════════════════════════════════════════════════════════════
# Usage Examples:
# ════════════════════════════════════════════════════════════════════════════
#
# 1. Use default config (auto-detected):
#    omad preprocess
#    omad score
#    omad inject
#    omad prepare-dataset --batch
#
# 2. Use custom config:
#    omad preprocess --config my_config.yaml
#    omad pipeline --config experiments/exp1.yaml
#
# 3. Override specific parameters:
#    omad preprocess --T 24 --seed 42
#    omad inject --theta-g 5.0 --theta-v 3.0
#
# ════════════════════════════════════════════════════════════════════════════
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(config_content)

    console.print(f"[green]✓[/green] Created default config: {output_path}")
