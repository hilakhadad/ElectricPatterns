# ElectricPatterns - Household Energy Consumption Analysis

Analysis pipeline for detecting and segregating electricity consumption patterns in Israeli households. Identifies device ON/OFF events in 3-phase power data, matches ON/OFF pairs, classifies devices (boiler, AC), and separates consumption into device-related and background power.

Data source: EnergyHive API, 3-phase power (w1, w2, w3), 1-minute resolution.

## Architecture

The pipeline has two core modules:

- **Module 1 (Disaggregation)** - Signal-level processing: detects power events, matches ON/OFF pairs, extracts device power from the aggregate signal. Runs iteratively at decreasing thresholds [2000, 1500, 1100, 800]W.
- **Module 2 (Identification)** - Session-level classification: filters transient noise, groups matches into usage sessions, classifies sessions as device types (boiler, central AC, regular AC). Runs once after all M1 iterations.

```
Module 1 (iterative):
  Iteration 0 (2000W): Raw CSV -> Detection -> Matching -> Segmentation -> Remaining power
  Iteration 1 (1500W): Remaining -> Detection -> Matching -> Segmentation -> Remaining power
  Iteration 2 (1100W): Remaining -> Detection -> Matching -> Segmentation -> Remaining power
  Iteration 3  (800W): Remaining -> Detection -> Matching -> Segmentation -> Remaining power

Module 2 (once):
  All matches -> Filter spikes (<3 min) -> Group sessions (30-min gap) -> Split sessions
             -> Classify (boiler -> central AC -> regular AC) -> Confidence score -> JSON output
```

Each iteration subtracts detected device power from the total, revealing smaller devices hidden underneath.

## Project Modules

| Module | Purpose | Entry Point |
|--------|---------|-------------|
| [experiment_pipeline](experiment_pipeline/) | Core pipeline (M1 + M2) | `scripts/test_single_house.py` |
| [disaggregation_analysis](disaggregation_analysis/) | M1 post-run reports | `scripts/run_dynamic_report.py` |
| [identification_analysis](identification_analysis/) | M2 post-run reports | `scripts/run_identification_report.py` |
| [house_analysis](house_analysis/) | Pre-analysis (data quality) | `scripts/run_analysis.py` |
| [harvesting_data](harvesting_data/) | Fetch data from API | `python -m harvesting_data.cli` |
| [user_plot_requests](user_plot_requests/) | Interactive web visualization | `app.py` |

## Quick Start

```bash
git clone https://github.com/hilakhadad/ElectricPatterns.git
cd ElectricPatterns
pip install -r requirements.txt
```

### Typical Workflow

```bash
# 1. Fetch data (if needed)
python -m harvesting_data.cli --parallel

# 2. Pre-analyze data quality
cd house_analysis/scripts && python run_analysis.py

# 3. Run pipeline (default: exp010 dynamic threshold)
cd experiment_pipeline/scripts
python test_single_house.py --house_id 305
python test_array_of_houses.py --skip_visualization   # all houses

# 4. Generate M1 disaggregation report
cd disaggregation_analysis/scripts && python run_dynamic_report.py

# 5. Run M2 identification (standalone, or automatically at end of pipeline)
cd experiment_pipeline/scripts
python run_identification.py --experiment_dir OUTPUT/experiments/exp010_... --house_id 305

# 6. Generate M2 identification report
cd identification_analysis/scripts
python run_identification_report.py --experiment <path>

# 7. Interactive visualization
cd user_plot_requests && python app.py   # http://localhost:5000
```

## Experiments

Defined in `experiment_pipeline/src/core/config.py`:

**Active** (`EXPERIMENTS`):

| Experiment | Threshold | Description |
|------------|-----------|-------------|
| **exp010_dynamic_threshold** | [2000, 1500, 1100, 800] | Dynamic threshold + identification (DEFAULT) |
| exp012_nan_imputation | [2000, 1500, 1100, 800] | exp010 + runtime NaN gap filling |

**Legacy** (`LEGACY_EXPERIMENTS`): exp000-exp008 document the evolution from baseline (1600W) to the current dynamic approach. Kept for backward compatibility.

## Project Structure

```
role_based_segregation_dev/
├── experiment_pipeline/       # Core algorithm (M1 disaggregation + M2 identification)
│   ├── src/
│   │   ├── core/              # Config, paths, data loading
│   │   ├── disaggregation/    # M1: detection/, matching/, segmentation/, pipeline/
│   │   ├── identification/    # M2: session_grouper, session_classifier, session_output
│   │   ├── pipeline/          # Unified runner (runner.py)
│   │   └── visualization/     # Interactive plots
│   ├── scripts/               # Entry points
│   └── tests/                 # Regression tests
├── disaggregation_analysis/   # M1 post-run HTML reports
├── identification_analysis/   # M2 post-run HTML reports
├── house_analysis/            # Pre-analysis (data quality scoring)
├── harvesting_data/           # EnergyHive API data acquisition
├── user_plot_requests/        # Web visualization (Flask)
├── INPUT/HouseholdData/       # Raw CSV data (gitignored)
└── investigations/            # One-off debug scripts (gitignored)
```

## Requirements

- Python 3.9+
- pandas, numpy, matplotlib, plotly, tqdm, flask, requests
