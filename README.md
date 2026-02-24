# ElectricPatterns - Household Energy Consumption Analysis

Analysis pipeline for detecting and segregating electricity consumption patterns in Israeli households. Identifies device ON/OFF events in 3-phase power data, matches ON/OFF pairs, classifies devices (boiler, AC, three-phase devices), and separates consumption into device-related and background power.

Data source: EnergyHive API, 3-phase power (w1, w2, w3), 1-minute resolution.

## Architecture

The pipeline has two core modules:

- **Module 1 (Disaggregation)** - Signal-level processing: detects power events, matches ON/OFF pairs, extracts device power from the aggregate signal. Runs iteratively at decreasing detection thresholds, peeling away large devices first so smaller ones become visible in the remaining signal.
- **Module 2 (Identification)** - Session-level classification: filters transient noise, groups matches into usage sessions, classifies sessions as device types. Runs once after all M1 iterations.

The pipeline supports configurable detection parameters (thresholds, matching tolerances, session gaps) to adapt to different household characteristics.

```
Module 1 (iterative):
  Iteration 0 (highest threshold): Raw CSV -> Detection -> Matching -> Segmentation -> Remaining power
  Iteration 1 (lower threshold):   Remaining -> Detection -> Matching -> Segmentation -> Remaining power
  ...continues at progressively lower thresholds...

  Optional recovery passes:
    Guided recovery  -> re-detect missed events using M2 classification hints
    Wave recovery    -> detect and extract AC cycling patterns from remaining signal

Module 2 (once):
  All matches -> Filter spikes -> Group sessions
             -> Classify (boiler -> three-phase -> central AC -> regular AC -> unknown)
             -> Confidence score -> JSON output
```

Each iteration subtracts detected device power from the total, revealing smaller devices hidden underneath.

### Device Types

The classification module identifies five device categories:

| Device | Description |
|--------|-------------|
| **Boiler** | High-power single-phase heating element, long continuous operation |
| **Three-phase device** | Synchronized high-power events across all 3 phases (likely EV charger) |
| **Central AC** | Multi-phase air conditioning with synchronized compressor cycles |
| **Regular AC** | Single-phase air conditioning with cycling compressor pattern |
| **Unknown** | Sessions that do not match any known device signature |

After observing recurring patterns across houses, additional device categories were added. Three-phase device (likely EV charger) was identified from synchronized high-power patterns across all 3 phases. Future work includes clustering unknown sessions to discover more device types.

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
python -m harvesting_data.cli

# 2. Pre-analyze data quality
cd house_analysis/scripts && python run_analysis.py

# 3. Run pipeline on a single house
cd experiment_pipeline/scripts
python test_single_house.py --house_id 305
python test_array_of_houses.py --skip_visualization   # all houses

# 4. Generate M1 disaggregation report
cd disaggregation_analysis/scripts && python run_dynamic_report.py

# 5. Run M2 identification (standalone, or automatically at end of pipeline)
cd experiment_pipeline/scripts
python run_identification.py --experiment_dir OUTPUT/experiments/<experiment_folder> --house_id 305

# 6. Generate M2 identification report
cd identification_analysis/scripts
python run_identification_report.py --experiment <path>

# 7. Interactive visualization
cd user_plot_requests && python app.py   # http://localhost:5000
```

## Project Structure

```
role_based_segregation_dev/
├── experiment_pipeline/       # Core algorithm (M1 disaggregation + M2 identification)
│   ├── src/
│   │   ├── core/              # Config, paths, data loading
│   │   ├── disaggregation/    # M1: signal-level processing
│   │   │   ├── rectangle/     #   Core detection, matching, segmentation, pipeline
│   │   │   └── wave_recovery/ #   AC wave pattern recovery from remaining signal
│   │   ├── identification/    # M2: session-level classification
│   │   │   ├── classifiers/   #   Device-specific classifiers (boiler, AC, central AC, unknown)
│   │   │   ├── session_builder.py, spike_stats.py
│   │   │   └── session_classifier.py, session_output.py
│   │   ├── pipeline/          # Orchestration (runner.py, pipeline_setup.py, post_pipeline.py)
│   │   └── visualization/     # Interactive plots
│   ├── scripts/               # Entry points
│   └── tests/                 # Regression tests
├── disaggregation_analysis/   # M1 post-run HTML reports
├── identification_analysis/   # M2 post-run HTML reports
├── house_analysis/            # Pre-analysis (data quality scoring)
├── harvesting_data/           # EnergyHive API data acquisition
├── user_plot_requests/        # Web visualization (Tkinter + Flask)
├── shared/                    # Shared utilities (html_utils.py)
├── INPUT/HouseholdData/       # Raw CSV data (gitignored)
└── investigations/            # One-off debug scripts (gitignored)
```

## Requirements

- Python 3.9+
- pandas, numpy, matplotlib, plotly, tqdm, tkcalendar, requests

## Legacy

### Experiments Table (removed 2026-02-24)

Previously this section listed specific experiment names and configurations. Experiments are now internal implementation details managed via `experiment_pipeline/src/core/config.py`. The pipeline supports multiple experiment configurations with configurable thresholds, and legacy experiments are preserved for backward compatibility.
