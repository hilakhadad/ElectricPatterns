# ElectricPatterns - Household Energy Consumption Analysis

Analysis pipeline for detecting and segregating electricity consumption patterns in households. Identifies device-specific events (ON/OFF), matches them, and separates consumption into device-related and background power.

## Features

- **Event Detection**: Automatic detection of ON/OFF power events
  - Sharp detection (single-sample threshold crossing)
  - Smart gradual detection (multi-minute ramp-ups/downs)
  - Progressive window search for better event separation
- **Event Matching**: Smart pairing of ON/OFF events with validation
  - Stage 1: Clean matching (stable power between ON/OFF)
  - Stage 2: Noisy matching (with interference from other devices)
  - Magnitude validation to prevent negative residuals
  - Classification: SPIKE, NON-M (normal), NOISY
- **Data Segmentation**: Role-based segregation into event-specific and background power
- **Experiment Framework**: Systematic configuration management for reproducible experiments
- **Visualization**: Interactive Plotly plots showing consumption patterns

## Project Structure

```
.
├── experiment_pipeline/
│   ├── src/                        # Core pipeline modules
│   │   ├── core/                   # Core utilities
│   │   │   ├── config.py           # Experiment configurations
│   │   │   ├── paths.py            # Path management
│   │   │   └── logging_setup.py    # Logging configuration
│   │   ├── detection/              # Event detection
│   │   │   ├── sharp.py            # Sharp event detection
│   │   │   ├── gradual.py          # Gradual event detection
│   │   │   ├── expander.py         # Event boundary expansion
│   │   │   └── merger.py           # Overlapping event merger
│   │   ├── matching/               # Event matching
│   │   │   ├── stage1.py           # Clean matching (no noise)
│   │   │   ├── stage2.py           # Noisy matching
│   │   │   ├── validator.py        # Match validation
│   │   │   └── io.py               # Match I/O operations
│   │   ├── segmentation/           # Data segmentation
│   │   │   ├── processor.py        # Segmentation logic
│   │   │   ├── evaluation.py       # Quality metrics
│   │   │   └── errors.py           # Error detection
│   │   ├── visualization/          # Plotting
│   │   │   ├── interactive.py      # Interactive Plotly plots
│   │   │   └── static.py           # Static matplotlib plots
│   │   ├── pipeline/               # Pipeline orchestration
│   │   │   ├── detection.py        # Detection step
│   │   │   ├── matching.py         # Matching step
│   │   │   ├── segmentation.py     # Segmentation step
│   │   │   ├── evaluation.py       # Evaluation step
│   │   │   └── visualization.py    # Visualization step
│   │   └── legacy/                 # Old code (deprecated)
│   ├── scripts/                    # Execution scripts
│   │   ├── test_single_house.py    # Run on one house
│   │   └── test_array_of_houses.py # Run on all houses (parallel)
│   ├── INPUT/                      # Input data (gitignored)
│   └── OUTPUT/                     # Results (gitignored)
├── harvesting_data/                # Data collection utilities
├── user_plot_requests/             # Interactive plotting tools
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
- `HOUSE_ID`: Which house to process (e.g., "140", "1001")
- `EXPERIMENT_NAME`: Which experiment configuration to use
- `MAX_ITERATIONS`: Number of iterations (default: 2)

**All houses in parallel:**
```bash
cd experiment_pipeline/scripts
python test_array_of_houses.py
```

This will:
- Process houses defined in `HOUSE_IDS` list
- Run with 8 parallel workers
- Save results to `OUTPUT/experiments/{experiment}_{timestamp}/`

### Available Experiments

Defined in `src/core/config.py`:

| Experiment | Description |
|------------|-------------|
| `exp000_baseline` | Original detection (1600W threshold, no gradual) |
| `exp001_gradual_detection` | Smart gradual detection (1600W) |
| `exp002_lower_TH` | Lower threshold (1500W) with gradual |
| `exp003_progressive_search` | Progressive window search (±1→2→3 min) |
| `exp004_noisy_matching` | Stage 2 noisy matching for interference |
| `exp005_asymmetric_windows` | Asymmetric window search for edge events |

## Pipeline Stages

1. **Detection** (`pipeline/detection.py`)
   - Detects power changes above threshold on each phase (w1, w2, w3)
   - Uses sharp + gradual detection with progressive window search
   - Output: `on_off_{threshold}.csv`

2. **Matching** (`pipeline/matching.py`)
   - Stage 1: Pairs ON/OFF with stable power between them
   - Stage 2: Pairs remaining ON/OFF with noise tolerance
   - Validates magnitude similarity to prevent negative residuals
   - Output: `matches_{house_id}.csv`

3. **Segmentation** (`pipeline/segmentation.py`)
   - Removes event power from total consumption
   - Creates remaining power for next iteration
   - Output: `segmented_{house_id}.csv`

4. **Evaluation** (`pipeline/evaluation.py`)
   - Calculates separation quality metrics
   - Detects negative values in output
   - Output: `evaluation_history_{house_id}.csv`

5. **Visualization** (`pipeline/visualization.py`)
   - Interactive 12-hour window plots with event markers
   - Shows original, remaining, and segregated power
   - Output: HTML files

## Performance Optimizations

- **Indexed lookups**: Uses pandas timestamp index for O(1) lookups instead of O(n)
- **Pre-filtering**: Filters candidates by magnitude before detailed validation
- **Progressive windows**: Starts with small time windows and expands (15min → 6hr)
- **Parallel processing**: Runs multiple houses concurrently

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, plotly, tqdm
