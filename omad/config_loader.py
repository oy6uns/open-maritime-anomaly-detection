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
    config_content = """# OMAD Configuration

# ── Global ──────────────────────────────────────────────
slices: [12, 24]              # Time slice lengths to process (12/24/48/72)
seed: 0                       # Random seed
verbose: false                # Detailed logging

# ── Preprocess ──────────────────────────────────────────
preprocess:
  input_csv: ./data/omtad.csv # Input AIS CSV file path
  trim: 12                    # Points to trim from track start/end
  min_track_len: 36           # Minimum track length (in points) for use
  ratios: [10, 5, 3, 1]       # ratios for determining Injection and Stratification (e.g., 10%, 5%, 3%, 1%)

# ── Inject ──────────────────────────────────────────────
inject:
  theta_g: 3.0                # A1 multiplier (position displacement)
  theta_v: 2.0                # A2 multiplier (speed/heading change)

# ── Prepare Dataset ─────────────────────────────────────
prepare-dataset:
  ratios: [10, 5, 3, 1]       # Which ratios to generate
  modes: [a1, a2, a3]         # Which anomaly types
  seeds: [2, 12, 32, 42, 52]  # Random seed for Dataset Split
  train: 0.7                  # Train split
  valid: 0.15                 # Validation split
  test: 0.15                  # Test split
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(config_content)

    console.print(f"[green]✓[/green] Created default config: {output_path}")
