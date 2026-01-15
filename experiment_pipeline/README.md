# Experiment Pipeline

Core pipeline for energy consumption analysis.

## Structure

```
experiment_pipeline/
├── src/                        # Core modules
│   ├── on_off_log.py           # Stage 1: Event detection
│   ├── new_matcher.py          # Stage 2: Event matching
│   ├── segmentation.py         # Stage 3: Data segmentation
│   ├── eval_segmentation.py    # Stage 4: Evaluation
│   ├── visualization_with_mark.py  # Stage 5: Visualization
│   ├── detection_config.py     # Experiment configurations
│   └── data_util.py            # Paths and utilities
├── scripts/                    # Execution scripts
│   ├── test_single_house.py    # Run pipeline on one house
│   ├── test_array_of_houses.py # Run on all houses (parallel)
│   └── analyze_results.py      # Summarize experiment results
├── tests/                      # Test suite
│   ├── run_all_tests.py
│   ├── test_unit.py
│   └── test_pipeline.py
├── INPUT/HouseholdData/        # Input CSV files (gitignored)
└── OUTPUT/experiments/         # Results (gitignored)
```

## Quick Start

```bash
# Run on single house
python scripts/test_single_house.py

# Run on all houses in parallel
python scripts/test_array_of_houses.py

# Run tests
python tests/run_all_tests.py
```

## Scripts

### test_single_house.py

Run the full pipeline on one house:

```python
# Configuration (edit in script)
HOUSE_ID = "1"                          # House to process
EXPERIMENT_NAME = "exp004_noisy_matching"  # Experiment config
MAX_ITERATIONS = 2                       # Number of iterations
```

Can also be imported:
```python
from test_single_house import run_pipeline_for_house

result = run_pipeline_for_house(
    house_id="1",
    experiment_name="exp004_noisy_matching",
    output_path="/path/to/output",
    max_iterations=2
)
```

### test_array_of_houses.py

Run pipeline on all houses in parallel:
- Auto-detects houses from `INPUT/HouseholdData/`
- Uses 8 parallel workers
- Creates timestamped output directory

### analyze_results.py

Generate summary report from experiment results:
```bash
python scripts/analyze_results.py /path/to/experiment/output
```

## Experiments

Defined in `src/detection_config.py`:

| Name | Threshold | Gradual | Progressive | Description |
|------|-----------|---------|-------------|-------------|
| exp000_baseline | 1600W | No | No | Original detection |
| exp001_gradual_detection | 1600W | Yes | No | Smart gradual detection |
| exp002_lower_TH | 1500W | Yes | No | Lower threshold |
| exp003_progressive_search | 1500W | Yes | Yes | Progressive window search |
| exp004_noisy_matching | 1500W | Yes | Yes | + Noisy event matching |

## Output Structure

```
OUTPUT/experiments/{exp_name}_{timestamp}/
├── experiment_metadata.json    # Configuration used
├── house_{id}/
│   ├── run_0/
│   │   └── house_{id}/
│   │       ├── on_off_1500.csv
│   │       ├── matches_{id}.csv
│   │       ├── segmented_{id}.csv
│   │       └── evaluation_history_{id}.csv
│   ├── run_1/
│   │   └── ...
│   └── plots/
│       └── {id}/
│           └── *.html
└── logs/
    └── test_{id}.log
```

## Testing

```bash
# All tests
python tests/run_all_tests.py

# Unit tests only
python tests/test_unit.py

# Integration tests only
python tests/test_pipeline.py
```
