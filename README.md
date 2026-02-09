# OMAD - Ocean Maritime Anomaly Detection

Maritime anomaly detection pipeline using LLM-based scoring and synthetic anomaly injection.

## Features

- **Preprocess**: Route slicing, stratification, and prompt generation
- **Score**: LLM-based anomaly plausibility scoring (Qwen/Qwen3-8B)
- **Inject**: Synthetic anomaly injection (A1: Unexpected AIS Activity, A2: Route Deviation, A3: Close Approach)
- **Prepare Dataset**: .zpz dataset generation for model training
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

### 3. Run Pipeline

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

## Project Structure

```
open-maritime-anomaly-detection/
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── setup.py                 # Package setup
├── README.md               
└── omad/                   
    ├── cli.py              # CLI entry point
    ├── config.py           
    ├── config_loader.py    
    ├── paths.py            
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

<!--
## Citation

If you use this code in your research, please cite:

```bibtex
@software{omad2024,
  title={OMAD: Ocean Maritime Anomaly Detection},
  author={},
  year={2024}
}
```
-->

## Contact

For questions or issues, please open an issue on GitHub.
