# ElectricPatterns - Household Energy Consumption Analysis

This repository contains the code and pipeline for analyzing electricity consumption patterns in households. The project focuses on event-based segmentation, device identification, and temporal consumption analysis.

## Project Objectives
- Identify distinct electricity usage patterns in household data
- Analyze device-specific energy consumption (water heaters, air conditioners, high-power appliances)
- Develop efficient segmentation and classification methods for power events
- Create visualizations to showcase consumption trends and anomalies

## Features
- **Event Detection**: Automatic detection of ON/OFF power events based on configurable thresholds
  - Sharp event detection (single-sample threshold)
  - Smart gradual detection (multi-minute ramp-ups/downs)
  - Progressive window search for better event separation
- **Experiment Framework**: Systematic configuration management for reproducible experiments
- **Event Matching**: Smart pairing of ON/OFF events with phase validation
- **Data Segmentation**: Role-based segregation of consumption into event-related and background power
- **Evaluation Tools**: Quality metrics for segmentation performance
- **Visualization**: Interactive plots showing consumption patterns across phases

## Repository Structure
```
.
├── experiment_pipeline/     # Main pipeline code
│   ├── INPUT/              # Input data directory (gitignored)
│   ├── OUTPUT/             # All outputs: results, logs, errors (gitignored)
│   ├── tests/              # Comprehensive test suite
│   ├── scripts/            # Helper scripts
│   ├── detection_config.py # Experiment configuration management
│   ├── on_off_log.py       # Event detection (sharp + gradual)
│   ├── new_matcher.py      # Event matching
│   ├── segmentation.py     # Data segmentation
│   ├── eval_segmentation.py # Evaluation metrics
│   ├── visualization_with_mark.py # Interactive plots
│   └── data_util.py        # Configuration and utilities
├── user_plot_requests/     # Interactive plotting tools
├── harvesting_data/        # Data collection utilities
└── requirements.txt        # Python dependencies
```

See [experiment_pipeline/README.md](experiment_pipeline/README.md) for detailed pipeline documentation.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- conda (recommended) or virtualenv
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hilakhadad/ElectricPatterns.git
   cd ElectricPatterns
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n electric_patterns python=3.9
   conda activate electric_patterns
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start

#### Running Tests
```bash
cd experiment_pipeline

# Run the test suite
python tests/run_all_tests.py
```

#### Running Experiments
```bash
cd experiment_pipeline

# Test on a single house with experiment configuration
python scripts/test_single_house.py

# Configuration is in scripts/test_single_house.py:
# - HOUSE_ID: Which house to test (e.g., "1", "2039")
# - EXPERIMENT_NAME: Which experiment to run (e.g., "exp003_progressive_search")
# - MAX_ITERATIONS: Number of iterations (default: 5)
```

**Available Experiments** (see `detection_config.py`):
- `exp000_baseline`: Original detection (TH=1600W, no gradual detection)
- `exp001_gradual_detection`: Smart gradual detection (TH=1600W)
- `exp002_lower_TH`: Lower threshold (TH=1500W) with gradual detection
- `exp003_progressive_search`: Progressive window search (TH=1500W, ±1→2→3 min)

For detailed usage instructions, see the [Pipeline README](experiment_pipeline/README.md).

## Pipeline Overview

The analysis pipeline consists of five main stages:

1. **On/Off Detection** (`on_off_log.py`)
   - Sharp event detection: Single-sample changes ≥ threshold
   - Gradual event detection: Multi-minute ramp-ups/downs (80-130% of threshold)
   - Progressive window search: Try ±1min, ±2min, ±3min windows sequentially
   - Output: `on_off_{threshold}.csv`

2. **Event Matching** (`new_matcher.py`)
   - Pairs ON/OFF events with temporal and phase validation
   - Classifies as spikes (≤2 min) or steady-state events
   - Output: `matches_{house_id}.csv`

3. **Segmentation** (`segmentation.py`)
   - Segregates consumption into event-specific and background power
   - Processes events by duration (shortest to longest)
   - Output: `segmented_{house_id}.csv`, `summarized_{house_id}.csv`

4. **Evaluation** (`eval_segmentation.py`)
   - Calculates separation quality metrics
   - Key metric: Explained Information Percentage
   - Output: `evaluation_history_{house_id}.csv`

5. **Visualization** (`visualization_with_mark.py`)
   - Generates interactive Plotly plots (12-hour windows)
   - 4 rows: original, remaining, short/medium/long duration events
   - Output: HTML plots in `plots/{house_id}/`

## Testing

The project includes comprehensive unit and integration tests:

```bash
# Run all tests
python experiment_pipeline/tests/run_all_tests.py

# Run specific test suites
python experiment_pipeline/tests/test_unit.py
python experiment_pipeline/tests/test_pipeline.py
```

## Configuration

### Experiment Configuration
Experiments are defined in `experiment_pipeline/detection_config.py` using the `ExperimentConfig` dataclass:

```python
@dataclass
class ExperimentConfig:
    exp_id: str                          # e.g., "exp003"
    description: str                     # Human-readable description
    threshold: int                       # Main detection threshold (W)
    off_threshold_factor: float          # OFF threshold = threshold × factor
    expand_event_factor: float           # Extend events if 5% additional change
    use_gradual_detection: bool          # Enable gradual detection
    gradual_window_minutes: int          # Search window (±N minutes)
    progressive_window_search: bool      # Try 1→2→3 min windows sequentially
```

### System Configuration
All paths and parameters are configured in `experiment_pipeline/data_util.py`. The pipeline uses relative paths and works with local INPUT/OUTPUT directories by default.

## Contributing

When making changes:
1. Run tests before committing: `python tests/run_all_tests.py`
2. Ensure all tests pass before pushing
3. Use meaningful commit messages

## License

[Add license information]

## Contact

[Add contact information]
