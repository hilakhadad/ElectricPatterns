# ElectricPatterns - Household Energy Consumption Analysis

Analysis pipeline for detecting and segregating electricity consumption patterns in Israeli households. Identifies device-specific events (ON/OFF) in 3-phase power data, matches ON→OFF pairs, classifies devices (boiler, AC), and separates consumption into device-related and background power.

Data source: EnergyHive API, 3-phase power (w1, w2, w3), 1-minute resolution.

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
- **NaN imputation**: Runtime gap filling to prevent false events at NaN boundaries

### Event Matching (3 Stages)
- **Stage 1**: Clean matching (stable power between ON/OFF)
- **Stage 2**: Noisy matching (with interference from other devices)
- **Stage 3**: Partial matching (mismatched ON/OFF magnitudes)
- Magnitude validation to prevent negative residuals

### Device Classification
- **Boiler**: Single-phase, >=1500W, >=25min duration, isolated
- **Central AC**: Synchronized events across 2+ phases within 10 minutes
- **Regular AC**: 800W+, compressor cycling pattern (3-30min cycles)

### Dynamic Threshold (Default)
The default experiment (`exp010`) uses decreasing thresholds across iterations to progressively detect smaller devices:
- Iteration 0: 2000W (boilers, large appliances)
- Iteration 1: 1500W (strong AC)
- Iteration 2: 1100W (medium AC)
- Iteration 3: 800W (small AC)

### Analysis & Reporting
- Matching performance metrics
- Segmentation quality scores
- Interactive HTML reports with Plotly charts
- Monthly breakdown analysis
- Dynamic report for multi-threshold experiments

## Project Structure

```
.
├── experiment_pipeline/      # Core pipeline (detection → matching → segmentation → classification)
│   ├── src/
│   │   ├── core/             # Config, paths, data loading, NaN imputation
│   │   ├── detection/        # Event detection (sharp, gradual, near-threshold, tail)
│   │   ├── matching/         # Stage 1, 2, 3 matching + validation
│   │   ├── segmentation/     # Power segmentation & evaluation
│   │   ├── classification/   # Device type classification (boiler, AC)
│   │   ├── output/           # Unified JSON output builder
│   │   ├── pipeline/         # Orchestration (runner, process_* steps)
│   │   └── visualization/    # Interactive plots
│   ├── scripts/              # Entry points
│   └── tests/                # Regression tests (pytest)
│
├── experiment_analysis/      # Post-run analysis & reports
│   ├── src/
│   │   ├── metrics/          # Matching, segmentation, pattern, classification metrics
│   │   ├── reports/          # Report generation
│   │   └── visualization/    # HTML reports & charts (static + dynamic)
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
├── harvesting_data/          # Data acquisition from EnergyHive API
│   ├── api.py                # API client
│   ├── fetcher.py            # Fetch logic with retry
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

3. **Run segmentation pipeline** (default: dynamic threshold):
   ```bash
   cd experiment_pipeline/scripts
   python test_single_house.py --house_id 305
   python test_array_of_houses.py --skip_visualization   # all houses
   ```

4. **Analyze results**:
   ```bash
   cd experiment_analysis/scripts
   python run_analysis.py --full
   python run_dynamic_report.py    # for dynamic threshold experiments
   ```

5. **View interactive plots**:
   ```bash
   cd user_plot_requests
   python app.py
   # Open http://localhost:5000
   ```

## Pipeline Overview

```
Iteration 0 (threshold=2000W):
  Raw CSV data → Detection → Matching → Segmentation → Remaining power

Iteration 1 (threshold=1500W):
  Remaining power → Detection → Matching → Segmentation → Remaining power

Iteration 2 (threshold=1100W):
  ...same flow, finds progressively smaller devices...

Iteration 3 (threshold=800W):
  ...final pass...

→ Classification (boiler, central AC, regular AC)
→ Unified JSON output (device_activations_{house_id}.json)
→ Evaluation & Visualization
```

Each iteration subtracts detected device power from the total, revealing smaller devices hidden underneath.

## Available Experiments

Defined in `experiment_pipeline/src/core/config.py`:

| Experiment | Threshold | Key Feature |
|------------|-----------|-------------|
| exp000_baseline | 1600W | Original detection |
| exp001_gradual_detection | 1600W | + Gradual detection |
| exp002_lower_TH | 1500W | Lower threshold |
| exp003_progressive_search | 1500W | + Progressive windows |
| exp004_noisy_matching | 1500W | + Stage 2 matching |
| exp005_asymmetric_windows | 1500W | Asymmetric time windows |
| exp006_partial_matching | 1500W | + Stage 3 partial matching |
| exp007_symmetric_threshold | 1300W | Symmetric ON/OFF (factor=1.0) |
| exp008_tail_extension | 1300W | + Tail extension for OFF events |
| **exp010_dynamic_threshold** | **[2000→1500→1100→800]** | **Dynamic threshold + classification (DEFAULT)** |
| exp012_nan_imputation | [2000→1500→1100→800] | exp010 + runtime NaN gap filling |

## Requirements

- Python 3.9+
- pandas, numpy, matplotlib, plotly, tqdm, flask, requests
