# ElectricPatterns - Household Energy Consumption Analysis

Analysis pipeline for detecting and segregating electricity consumption patterns in households. Identifies device-specific events (ON/OFF), matches them, and separates consumption into device-related and background power.

## Project Modules

| Module | Purpose | Entry Point |
|--------|---------|-------------|
| [experiment_pipeline](experiment_pipeline/) | Core segmentation pipeline | `scripts/test_single_house.py` |
| [experiment_analysis](experiment_analysis/) | Analyze experiment results | `scripts/run_analysis.py` |
| [house_analysis](house_analysis/) | Pre-analyze data quality | `scripts/run_analysis.py` |
| [user_plot_requests](user_plot_requests/) | Interactive web visualization | `app.py` |
| [harvesting_data](harvesting_data/) | Fetch data from API | `python -m harvesting_data.cli` |

## Features

### Event Detection
- **Sharp detection**: Single-sample threshold crossing
- **Gradual detection**: Multi-minute ramp-ups/downs
- **Near-threshold detection**: Captures events at 85-100% of threshold
- **Tail extension**: Extends OFF events through residual power decay
- **Progressive window search**: Better event separation

### Event Matching (3 Stages)
- **Stage 1**: Clean matching (stable power between ON/OFF)
- **Stage 2**: Noisy matching (with interference from other devices)
- **Stage 3**: Partial matching (mismatched ON/OFF magnitudes)
- Magnitude validation to prevent negative residuals
- Classification: SPIKE, NON-M (normal), NOISY, PARTIAL

### Analysis & Reporting
- Matching performance metrics
- Segmentation quality scores
- Device detection (Central AC, Regular AC, Boiler)
- Interactive HTML reports with charts
- Monthly breakdown analysis

## Project Structure

```
.
├── experiment_pipeline/      # Core pipeline (detection, matching, segmentation)
│   ├── src/                  # Modular source code
│   │   ├── detection/        # Event detection algorithms
│   │   ├── matching/         # Stage 1, 2, 3 matching
│   │   ├── segmentation/     # Power segmentation
│   │   └── pipeline/         # Orchestration
│   └── scripts/              # Execution scripts
│
├── experiment_analysis/      # Experiment result analysis
│   ├── src/
│   │   ├── metrics/          # Matching, segmentation, pattern metrics
│   │   ├── reports/          # Report generation
│   │   └── visualization/    # HTML reports & charts
│   └── scripts/
│
├── house_analysis/           # Pre-analysis of raw data
│   ├── src/
│   │   ├── metrics/          # Coverage, quality, temporal metrics
│   │   ├── reports/          # House reports
│   │   └── visualization/    # Charts & HTML
│   └── scripts/
│
├── user_plot_requests/       # Web visualization app
│   ├── src/                  # Data loading & plot generation
│   └── app.py                # Flask server
│
├── harvesting_data/          # Data acquisition from API
│   ├── api.py                # EnergyHive API client
│   ├── fetcher.py            # Fetch logic
│   └── cli.py                # Command-line interface
│
└── INPUT/HouseholdData/      # Input CSV files (gitignored)
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

### Typical Workflow

1. **Fetch data** (if needed):
   ```bash
   python -m harvesting_data.cli --parallel
   ```

2. **Pre-analyze data quality**:
   ```bash
   cd house_analysis/scripts
   python run_analysis.py
   ```

3. **Run segmentation pipeline**:
   ```bash
   cd experiment_pipeline/scripts
   python test_array_of_houses.py
   ```

4. **Analyze results**:
   ```bash
   cd experiment_analysis/scripts
   python run_analysis.py --full
   ```

5. **View interactive plots**:
   ```bash
   cd user_plot_requests
   python app.py
   # Open http://localhost:5000
   ```

## Pipeline Stages

### 1. Detection
Detects power changes above threshold on each phase (w1, w2, w3).
- Sharp detection for sudden changes
- Gradual detection for slow ramps
- Output: `on_off_{threshold}.csv`

### 2. Matching
Pairs ON/OFF events with validation:

| Stage | Description | Tag |
|-------|-------------|-----|
| Stage 1 | Stable power between events | NON-M, SPIKE |
| Stage 2 | Tolerates noise from other devices | NOISY |
| Stage 3 | Handles magnitude mismatch (>350W diff) | PARTIAL |

Output: `matches_{house_id}.csv`

### 3. Segmentation
Removes event power from total consumption.
- Creates remaining power for next iteration
- Categorizes by duration (short/medium/long)
- Output: `summarized_{house_id}.csv`

### 4. Analysis
Calculates performance metrics:
- Matching rate, segmentation ratio
- Device detection (AC, boiler patterns)
- Monthly breakdown
- Output: HTML reports with interactive charts

## Device Detection

### Central AC
- Synchronized events across all 3 phases
- High power (>800W per phase)

### Regular AC
Detection criteria:
- Power >= 800W
- Cycle duration: 3-30 minutes
- Session: 2+ cycles, 30+ min total
- Magnitude consistency (std < 20%)

### Boiler
- Single phase, high power
- Long duration (>30 min)

## Available Experiments

Defined in `experiment_pipeline/src/core/config.py`:

| Experiment | Threshold | Features |
|------------|-----------|----------|
| exp000_baseline | 1600W | Original detection |
| exp001_gradual_detection | 1600W | + Gradual detection |
| exp002_lower_TH | 1500W | Lower threshold |
| exp003_progressive_search | 1500W | + Progressive windows |
| exp004_noisy_matching | 1500W | + Stage 2 matching |
| exp005_asymmetric_windows | 1500W | Asymmetric time windows |
| exp006_partial_matching | 1500W | + Stage 3 partial matching |
| exp007_symmetric_threshold | 1300W | Symmetric ON/OFF (factor=1.0) |
| exp008_tail_extension | 1300W | + Tail extension for OFF events |

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, plotly, tqdm, flask
