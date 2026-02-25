# ElectricPatterns - Household Energy Consumption Analysis

Analysis pipeline for detecting and segregating electricity consumption patterns in Israeli households. Identifies device ON/OFF events in 3-phase power data, matches ON/OFF pairs, classifies devices (boiler, AC, three-phase devices), and separates consumption into device-related and background power.

Data source: EnergyHive API, 3-phase power (w1, w2, w3), 1-minute resolution.

## Architecture

The pipeline has two core modules:

- **Module 1 (Disaggregation)** - Signal-level processing: detects power events, matches ON/OFF pairs, extracts device power from the aggregate signal. Runs iteratively at decreasing detection thresholds, peeling away large devices first so smaller ones become visible in the remaining signal.
- **Module 2 (Identification)** - Session-level classification: filters transient noise, groups matches into usage sessions, classifies sessions as device types. Runs once after all M1 iterations.

The pipeline supports configurable detection parameters (thresholds, matching tolerances, session gaps) to adapt to different household characteristics.

```
Module 1 (iterative rectangle matching):
  Iteration 0 (2000W): Raw CSV -> Detection -> Matching -> Segmentation -> Remaining power
  Iteration 1 (1500W): Remaining -> Detection -> Matching -> Segmentation -> Remaining power
  Iteration 2 (1100W): Remaining -> ...
  Iteration 3 (800W):  Remaining -> ...

  Post-iteration recovery passes (optional, enabled in exp013+):
    Guided recovery  -> re-detect missed compressor cycles using matched session templates
    Wave recovery    -> detect wave-shaped patterns (sharp rise -> gradual decay) from remaining
    Hole repair      -> fix rectangle matches that extracted wave-shaped events as flat rectangles

Module 2 (once):
  All matches -> Filter spikes -> Group sessions
             -> Classify (boiler -> three-phase -> central AC -> regular AC -> unknown)
             -> Confidence score -> JSON output
```

Each iteration subtracts detected device power from the total, revealing smaller devices hidden underneath.

### Device Types

The classification module identifies six device categories:

| Device | Description |
|--------|-------------|
| **Boiler** | High-power single-phase heating element, long continuous operation |
| **Three-phase device** | Synchronized high-power events across all 3 phases (likely EV charger) |
| **Central AC** | Multi-phase air conditioning with synchronized compressor cycles |
| **Regular AC** | Single-phase air conditioning with cycling compressor pattern |
| **Recurring pattern** | DBSCAN-discovered clusters of sessions with similar magnitude + duration |
| **Unknown** | Sessions that do not match any known device signature |

Three-phase device (likely EV charger) was identified from synchronized high-power patterns across all 3 phases. Recurring patterns are discovered via DBSCAN clustering and matched across houses to assign global device names (Device A, B, C...).

## Project Modules

| Module | Purpose | Entry Point |
|--------|---------|-------------|
| [experiment_pipeline](experiment_pipeline/) | Core pipeline (M1 + M2) | `scripts/test_single_house.py`, `scripts/run_local_batch.py` |
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

# 3. Run pipeline on a single house (default experiment: exp015_hole_repair)
cd experiment_pipeline/scripts
python test_single_house.py --house_id 305
python test_single_house.py --house_id 305 --experiment_name exp015_hole_repair

# 3b. Run pipeline on multiple houses (local batch)
python run_local_batch.py --houses 305,1234,2008
python run_local_batch.py --shortest 10   # 10 smallest houses

# 3c. HPC: month-level parallel processing (SLURM)
bash scripts/sbatch_run_houses.sh          # Submit all houses

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
│   │   │   │   ├── detection/ #     Sharp, gradual, settling, near-threshold, tail extension
│   │   │   │   ├── matching/  #     Stage 1-3, stack matcher, validator
│   │   │   │   ├── segmentation/ #  Power extraction & evaluation
│   │   │   │   └── pipeline/  #     Orchestration steps (detection, matching, segmentation, recovery)
│   │   │   └── wave_recovery/ #   AC wave pattern recovery from remaining signal
│   │   │       ├── detection/ #     Wave pattern detector
│   │   │       ├── matching/  #     Cross-phase wave matcher
│   │   │       ├── segmentation/ #  Wave segmentor & validator
│   │   │       └── pipeline/  #     Wave recovery step, hole repair, I/O
│   │   ├── identification/    # M2: session-level classification
│   │   │   ├── classifiers/   #   Device-specific classifiers (boiler, AC, central AC, recurring pattern, unknown)
│   │   │   ├── session_builder.py, spike_stats.py
│   │   │   └── session_classifier.py, session_output.py
│   │   ├── pipeline/          # Orchestration (runner.py, pipeline_setup.py, post_pipeline.py)
│   │   └── visualization/     # Interactive plots
│   ├── scripts/               # Entry points (test_single_house.py, run_local_batch.py, run_single_month.py)
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
- pandas, numpy, matplotlib, plotly, tqdm, tkcalendar, requests, scipy

## Legacy

### Experiments Table (removed 2026-02-24)

Previously this section listed specific experiment names and configurations. Experiments are now internal implementation details managed via `experiment_pipeline/src/core/config.py`. The pipeline supports multiple experiment configurations with configurable thresholds, and legacy experiments are preserved for backward compatibility.
