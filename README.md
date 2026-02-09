# OMAD - Ocean Maritime Anomaly Detection

Maritime anomaly detection pipeline using LLM-based scoring and synthetic anomaly injection.

## Features

- **Preprocess**: Route slicing, stratification, and prompt generation
- **Score**: LLM-based anomaly suitability scoring (Qwen/Qwen3-8B)
- **Inject**: Synthetic anomaly injection (A1: position, A2: velocity, A3: close approach)
- **Prepare Dataset**: NPZ dataset generation for model training
- **Pipeline**: End-to-end execution with customizable parameters

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (for LLM scoring)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd OpenMaritimeAnomalyDetection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

## Quick Start

### 1. Prepare Data

```bash
# Create data directory
mkdir -p data

# Place your AIS CSV file in the data directory
# Example: copy your AIS trajectory data to data/omtad.csv
cp /path/to/your/ais_data.csv data/omtad.csv
```

### 2. Initialize Configuration

```bash
# Create default config.yaml
omad config --init

# Edit config.yaml to set your data paths (usually just input_csv)
nano config.yaml
```

### 3. Configuration File

The OMAD CLI automatically searches for `config.yaml` in the following locations:
1. Current directory (`./config.yaml`)
2. Current directory (`./omad.yaml`)
3. User home directory (`~/.omad/config.yaml`)
4. Project directory

**Automatic Config Detection:**
```bash
# No --config needed - automatically uses config.yaml if found
omad preprocess
omad score
omad inject
```

**Custom Config Path:**
```bash
# Use a specific config file
omad preprocess --config my_config.yaml
omad pipeline --config experiments/exp1.yaml
```

The `config.yaml` file contains detailed comments explaining each parameter:

```yaml
# Global settings
T: 12              # Time slice length in hours (12/24/48/72)
seed: 0            # Random seed for reproducibility
verbose: false     # Enable detailed logging

# Preprocess - Stage 1
preprocess:
  input_csv: ./data/omtad.csv    # Input AIS CSV file (only required setting!)
  trim: 0                         # Points to trim from track ends/start
  min_track_len: 30               # Minimum track length
  ratios: "10,5,3,1"              # Anomaly percentage ratios

# Score - Stage 2 (LLM Anomaly Scoring)
score:
  batch_root: null                # Auto-derived from T parameter
  max_new_tokens: 256             # Maximum LLM generation tokens
  max_retries: -1                 # JSON validation retry limit

# Inject - Stage 3 (Anomaly Injection)
inject:
  input_csv: null                 # Auto-derived from T
  theta_g: 3.0                    # A1 anomaly multiplier (position)
  theta_v: 2.0                    # A2 anomaly multiplier (velocity)

# Prepare Dataset - Stage 4
prepare-dataset:
  base_dir: .                     # Project root (uses route_sliced_{T}/)
  percentage: "10pct"             # Anomaly percentage (10pct/5pct/3pct/1pct)
  train: 0.7                      # Training set ratio
  valid: 0.15                     # Validation set ratio
  test: 0.15                      # Test set ratio
```

See the generated `config.yaml` file for comprehensive parameter documentation.

### 4. Run Pipeline

```bash
# Run individual stages (auto-detects config.yaml)
omad preprocess
omad score
omad inject
omad prepare-dataset --batch

# Or run complete pipeline
omad pipeline

# Override config location if needed
omad preprocess --config experiments/custom.yaml
omad pipeline --config my_config.yaml
```

## Commands

### Preprocess

Data preprocessing: route slicing, stratification, prompt generation.

```bash
omad preprocess --T 12 --r 10,5,3,1 --input-csv ./data/omtad.csv
```

Options:
- `--T`: Time slice length (12/24/48/72)
- `--input-csv`: Input AIS CSV path
- `--output-dir`: Output directory
- `--r`: Anomaly ratios (comma-separated: 10,5,3,1)
- `--seed`: Random seed
- `--skip-stratification`: Skip indices generation
- `--skip-prompts`: Skip prompt generation

### Score

Generate LLM anomaly suitability scores.

```bash
omad score --T 12
```

Options:
- `--T`: Time slice (auto-derives paths)
- `--batch-root`: Batch root with user_query/ subdirs
- `--max-new-tokens`: Max LLM tokens (default: 256)
- `--max-retries`: Validation retries (-1=unlimited)

### Inject

Inject synthetic anomalies into routes.

```bash
omad inject --T 12 --seed 42 --theta-g 3.0 --theta-v 2.0
```

Options:
- `--T`: Route slice length
- `--seed`: Random seed
- `--theta-g`: A1 SED multiplier (default: 3.0)
- `--theta-v`: A2 speed/heading multiplier (default: 2.0)

### Prepare Dataset

Prepare NPZ datasets for model training.

```bash
# Batch mode: generate all combinations
omad prepare-dataset --batch

# Single mode: specific configuration
omad prepare-dataset --csv routes_injected.csv --mode a1 --T 12 --seed 2
```

Options:
- `--batch`: Generate all combinations
- `--csv`: Input CSV (single mode)
- `--mode`: Anomaly type (a1/a2/a3)
- `--percentage`: Anomaly percentage (10pct/5pct/3pct/1pct)

### Pipeline

Run complete pipeline with one command.

```bash
omad pipeline --T 12 --seed 42 --percentage 10pct
```

Options:
- `--skip-preprocess`: Skip preprocessing (default: true)
- `--skip-score`: Skip LLM scoring (default: false)
- `--theta-g`: A1 multiplier
- `--theta-v`: A2 multiplier

## Project Structure

```
OpenMaritimeAnomalyDetection/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── README.md               # This file
└── omad/                   # Main package
    ├── cli.py              # CLI entry point
    ├── config.py           # Configuration management
    ├── config_loader.py    # YAML config loader
    ├── paths.py            # Path resolution
    ├── preprocess/         # Stage 1: Preprocessing
    │   ├── __init__.py
    │   ├── loader.py
    │   ├── slice_tracks.py
    │   ├── stratification.py
    │   └── prompt_generation.py
    ├── score/              # Stage 2: LLM Scoring
    │   ├── __init__.py
    │   ├── main.py
    │   ├── llm_core.py
    │   └── prompts.py
    ├── inject/             # Stage 3: Anomaly Injection
    │   ├── __init__.py
    │   ├── main.py
    │   ├── a1_injector.py
    │   ├── a2_injector.py
    │   ├── a3_injector.py
    │   └── inject_utils.py
    ├── prepare_dataset/    # Stage 4: Dataset Preparation
    │   ├── __init__.py
    │   └── prepare_xy.py
    └── utils/              # Utilities
        ├── logging.py
        └── validation.py
```

## Data Format

### Input (AIS Data)

CSV with columns:
- `TIMESTAMP`: Datetime
- `CRAFT_ID`: Vessel identifier
- `TRACK_ID`: Track identifier
- `LON`, `LAT`: Coordinates
- `SPEED`: Speed over ground (knots)
- `COURSE`: Course over ground (degrees)

### Output (NPZ Dataset)

Each NPZ file contains:
- `X`: Flattened features (n_samples, n_features)
- `X_seq`: Sequential features (n_samples, T, n_features)
- `y`: Binary anomaly labels (n_samples, T)
- `route_ids`: Route identifiers (n_samples,)

## Configuration Management

### Create Config

```bash
omad config --init                    # Create config.yaml in current directory
omad config --init --path my.yaml     # Create at custom path
```

### View Config

```bash
omad config --show                    # Show current config
omad config --show --path my.yaml     # Show specific config
```

### Use Config

```bash
# All commands support --config parameter
omad preprocess --config config.yaml
omad pipeline --config config.yaml
```

## Advanced Usage

### Custom Paths

Override config file settings with CLI arguments:

```bash
omad preprocess --config config.yaml --T 24 --output-dir ./custom_output
```

### Interactive LLM Scoring

```bash
omad score --interactive
```

### Verbose Logging

```bash
omad preprocess --verbose
omad pipeline --verbose
```

## Environment Variables

You can use environment variables in your workflow:

```bash
export OMAD_DATA_DIR=/path/to/your/data
export OMAD_OUTPUT_DIR=/path/to/output
```

Then reference them in your config.yaml or scripts.

## Troubleshooting

### Import Errors

Make sure the package is installed:
```bash
pip install -e .
```

### CUDA Out of Memory

Reduce batch size or use CPU:
```bash
# Edit config.yaml or use smaller max_new_tokens
omad score --max-new-tokens 128
```

### File Not Found

Check your config.yaml paths are correct:
```bash
omad config --show
```

## License

[Your License Here]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{omad2024,
  title={OMAD: Ocean Maritime Anomaly Detection},
  author={},
  year={2024}
}
```

## Contact

For questions or issues, please open an issue on GitHub.
