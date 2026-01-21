# Experiment Pipeline

Core pipeline for energy consumption analysis - detecting, matching, and segmenting power events.

## Structure

```
experiment_pipeline/
├── src/                           # Source code (modular structure)
│   ├── core/                      # Configuration, paths, logging
│   ├── detection/                 # Event detection (sharp, gradual)
│   ├── matching/                  # Event matching (stage1, stage2)
│   ├── segmentation/              # Power segmentation & evaluation
│   ├── visualization/             # Interactive & static plots
│   ├── pipeline/                  # Orchestration (process_* functions)
│   └── legacy/                    # Backup of original code
├── scripts/                       # Execution scripts
│   ├── test_single_house.py       # Run pipeline on one house
│   └── test_array_of_houses.py    # Run on all houses (parallel)
├── tests/                         # Test suite
├── INPUT/HouseholdData/           # Input CSV files (gitignored)
└── OUTPUT/experiments/            # Results (gitignored)
```

## Quick Start

```bash
# Run on single house
python scripts/test_single_house.py

# Run on all houses in parallel
python scripts/test_array_of_houses.py
```

## Scripts

### test_single_house.py

Run the full pipeline on one house:

```python
# Configuration (edit in script)
HOUSE_ID = "125"                           # House to process
EXPERIMENT_NAME = "exp005_asymmetric_windows"  # Experiment config
MAX_ITERATIONS = 2                          # Number of iterations
```

Can also be imported:
```python
from test_single_house import run_pipeline_for_house

result = run_pipeline_for_house(
    house_id="125",
    experiment_name="exp005_asymmetric_windows",
    output_path="/path/to/output",
    max_iterations=2
)
```

### test_array_of_houses.py

Run pipeline on all houses in parallel:
- Auto-detects houses from `INPUT/HouseholdData/`
- Uses parallel workers
- Creates timestamped output directory

## Module Structure

### core/
Central configuration and utilities:
- `config.py` - ExperimentConfig dataclass, experiment definitions
- `paths.py` - Path constants, PathManager class
- `logging_setup.py` - Logging configuration

### detection/
Event detection algorithms:
- `sharp.py` - Single-minute power jumps
- `gradual.py` - Multi-minute power ramps
- `merger.py` - Merge overlapping events
- `expander.py` - Expand events to include adjacent changes

### matching/
Event matching algorithms:
- `stage1.py` - Exact matching within time window
- `stage2.py` - Tolerant matching with power tolerance
- `stack_matcher.py` - Stack-based ON/OFF matching
- `validator.py` - Match validation

### segmentation/
Power segmentation:
- `processor.py` - Core segmentation logic
- `summarizer.py` - Duration category summarization
- `evaluation.py` - Metrics calculation

### visualization/
Plotting utilities:
- `interactive.py` - Plotly interactive plots
- `static.py` - Matplotlib static plots

### pipeline/
Orchestration functions that tie everything together:
- `process_detection()` - Run detection step
- `process_matching()` - Run matching step
- `process_segmentation()` - Run segmentation step
- `process_evaluation()` - Run evaluation step
- `process_visualization()` - Run visualization step

## Experiments

Defined in `src/core/config.py`:

| Name | Threshold | Gradual | Description |
|------|-----------|---------|-------------|
| exp000_baseline | 1600W | No | Original detection |
| exp001_gradual_detection | 1600W | Yes | Smart gradual detection |
| exp002_lower_TH | 1500W | Yes | Lower threshold |
| exp003_progressive_search | 1500W | Yes | Progressive window search |
| exp004_noisy_matching | 1500W | Yes | + Noisy event matching |
| exp005_asymmetric_windows | 1500W | Yes | Asymmetric time windows |

## Output Structure

```
OUTPUT/experiments/{exp_name}_{timestamp}/
├── experiment_metadata.json    # Configuration used
├── run_0/house_{id}/
│   ├── on_off_1500.csv         # Detected events
│   ├── matches_{id}.csv        # Matched events
│   ├── summarized_{id}.csv     # Segmented data
│   └── evaluation_history_{id}.csv
├── run_1/...
├── plots/{id}/                 # Interactive HTML plots
└── logs/test_{id}.log
```

## Usage Example

```python
from pipeline import (
    process_detection,
    process_matching,
    process_segmentation,
    process_evaluation,
    process_visualization
)
from core import get_experiment

# Get experiment configuration
config = get_experiment("exp005_asymmetric_windows")

# Run pipeline steps
process_detection(house_id="125", run_number=0, threshold=config.threshold, config=config)
process_matching(house_id="125", run_number=0, threshold=config.threshold)
process_segmentation(house_id="125", run_number=0)
process_evaluation(house_id="125", run_number=0, threshold=config.threshold)
process_visualization(house_id="125", run_number=0, threshold=config.threshold)
```
