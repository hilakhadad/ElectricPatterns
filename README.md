# ElectricPatterns - Household Energy Consumption Analysis

Analysis pipeline for detecting and segregating electricity consumption patterns in households. Identifies device-specific events (ON/OFF), matches them, and separates consumption into device-related and background power.

## Features

- **Event Detection**: Automatic detection of ON/OFF power events
  - Sharp detection (single-sample threshold crossing)
  - Smart gradual detection (multi-minute ramp-ups/downs)
  - Progressive window search for better event separation
- **Event Matching**: Smart pairing of ON/OFF events with phase validation
  - Noisy matching for events with interference
  - Classification: SPIKE, NON-M (normal), NOISY
- **Data Segmentation**: Role-based segregation into event-specific and background power
- **Experiment Framework**: Systematic configuration management for reproducible experiments
- **Visualization**: Interactive Plotly plots showing consumption patterns

## Project Structure

```
.
├── experiment_pipeline/
│   ├── src/                    # Core pipeline modules
│   │   ├── on_off_log.py       # Event detection
│   │   ├── new_matcher.py      # Event matching
│   │   ├── segmentation.py     # Data segmentation
│   │   ├── eval_segmentation.py
│   │   ├── visualization_with_mark.py
│   │   ├── detection_config.py # Experiment configurations
│   │   └── data_util.py        # Paths and utilities
│   ├── scripts/                # Execution scripts
│   │   ├── test_single_house.py    # Run on one house
│   │   ├── test_array_of_houses.py # Run on all houses (parallel)
│   │   └── analyze_results.py      # Summarize experiment results
│   ├── tests/                  # Test suite
│   ├── INPUT/                  # Input data (gitignored)
│   └── OUTPUT/                 # Results (gitignored)
├── harvesting_data/            # Data collection utilities
├── user_plot_requests/         # Interactive plotting tools
└── requirements.txt
```

## Getting Started

### Installation

```bash
git clone https://github.com/hilakhadad/ElectricPatterns.git
cd ElectricPatterns

# Create environment
conda create -n electric_patterns python=3.9
conda activate electric_patterns

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

**Single house:**
```bash
cd experiment_pipeline/scripts
python test_single_house.py
```

Configure in the script:
- `HOUSE_ID`: Which house to process (e.g., "1", "1001")
- `EXPERIMENT_NAME`: Which experiment configuration to use
- `MAX_ITERATIONS`: Number of iterations (default: 2)

**All houses in parallel:**
```bash
cd experiment_pipeline/scripts
python test_array_of_houses.py
```

This will:
- Auto-detect all houses in `INPUT/HouseholdData/`
- Run with 8 parallel workers
- Save results to `OUTPUT/experiments/{experiment}_{timestamp}/`

### Available Experiments

Defined in `src/detection_config.py`:

| Experiment | Description |
|------------|-------------|
| `exp000_baseline` | Original detection (1600W threshold, no gradual) |
| `exp001_gradual_detection` | Smart gradual detection (1600W) |
| `exp002_lower_TH` | Lower threshold (1500W) with gradual |
| `exp003_progressive_search` | Progressive window search (±1→2→3 min) |
| `exp004_noisy_matching` | Stage 2 noisy matching for events with interference |

## Pipeline Stages

1. **On/Off Detection** (`on_off_log.py`)
   - Detects power changes above threshold on each phase (w1, w2, w3)
   - Output: `on_off_{threshold}.csv`

2. **Event Matching** (`new_matcher.py`)
   - Pairs ON/OFF events with phase validation
   - Classifies as SPIKE (≤2 min), NON-M (normal), or NOISY
   - Output: `matches_{house_id}.csv`

3. **Segmentation** (`segmentation.py`)
   - Removes event power from total consumption
   - Creates remaining power for next iteration
   - Output: `segmented_{house_id}.csv`, next iteration input

4. **Evaluation** (`eval_segmentation.py`)
   - Calculates separation quality metrics
   - Output: `evaluation_history_{house_id}.csv`

5. **Visualization** (`visualization_with_mark.py`)
   - Interactive 12-hour window plots
   - Output: HTML files in `plots/{house_id}/`

## Testing

```bash
cd experiment_pipeline
python tests/run_all_tests.py
```

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, plotly, tqdm, requests
