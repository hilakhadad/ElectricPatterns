# CLAUDE.md — ElectricPatterns Project Guide

## Safety Rules (MANDATORY)

1. **Never delete source files** — no `rm` on `.py`, `.csv`, `.pkl`, `.ipynb` files
2. **Backup before overwrite** — copy original before replacing any data or config file
3. **Work on branches** — never commit directly to `main`. Use `feature/` or `fix/` branches
4. **Never push without explicit user approval** — always ask before `git push`
5. **Run tests after code changes** — `cd experiment_pipeline && python -m pytest tests/ -v`
6. **Ask user when uncertain** — if requirements are ambiguous, ask before implementing
7. **Don't modify OUTPUT files** — pipeline output (pkl, csv) under `OUTPUT/` is generated data
8. **Preserve data integrity** — never modify files under `INPUT/HouseholdData/`

## Project Overview

**ElectricPatterns** — Household electricity consumption analysis pipeline by Hila Khadad.

Detects device ON/OFF events in 3-phase power data (w1, w2, w3), matches ON→OFF pairs, and segregates total power into device-specific + background consumption. Data comes from the EnergyHive API at 1-minute resolution.

**Two-module architecture**:
- **Module 1 (Disaggregation)** — Signal-level: detects events, matches ON→OFF pairs, extracts device power from aggregate signal. Runs iteratively at [2000, 1500, 1100, 800]W thresholds.
- **Module 2 (Identification)** — Session-level: filters transient noise, groups matches into sessions, classifies sessions as device types (boiler, central AC, regular AC). Runs once after all M1 iterations complete.

**Core invariant**: Every detected event must exist in exactly ONE of:
- `matches/` (as `on_event_id` / `off_event_id`)
- `unmatched_on/` or `unmatched_off/` (as `event_id`)

## Project Structure

```
role_based_segregation_dev/
├── experiment_pipeline/       # Core algorithm (M1 disaggregation + M2 identification)
│   ├── src/
│   │   ├── core/              # Config, paths, data loading
│   │   ├── disaggregation/    # M1: Signal-level processing
│   │   │   ├── detection/     #   Sharp + gradual event detection
│   │   │   ├── matching/      #   Stage 1-3 ON/OFF matching
│   │   │   ├── segmentation/  #   Power extraction from total
│   │   │   └── pipeline/      #   Orchestration steps (*_step.py)
│   │   ├── identification/    # M2: Session-level classification
│   │   │   ├── config.py      #   Constants (session gap, thresholds)
│   │   │   ├── session_grouper.py    # Load, filter spikes, group sessions
│   │   │   ├── session_classifier.py # Classify: boiler→central_ac→regular_ac
│   │   │   ├── session_output.py     # Session + backward-compat JSON output
│   │   │   └── cleanup.py           # Intermediate file cleanup
│   │   ├── pipeline/          # Unified runner (runner.py)
│   │   └── visualization/     # Interactive plots
│   ├── scripts/               # Entry points
│   │   ├── test_single_house.py      # Single house (default exp010)
│   │   ├── test_array_of_houses.py   # Batch processing
│   │   ├── run_identification.py     # Standalone M2 on existing M1 output
│   │   └── run_single_month.py       # Month-level HPC parallelism
│   ├── tests/                 # Regression tests (pytest)
│   └── OUTPUT/                # Generated experiment results
├── disaggregation_analysis/   # Post-run M1 analysis & HTML reports
│   ├── src/
│   │   ├── metrics/           # matching, segmentation, patterns, events, monthly, iterations
│   │   ├── reports/           # experiment_report, aggregate_report
│   │   └── visualization/     # dynamic_html_report, charts (Plotly)
│   └── scripts/               # run_analysis.py, run_dynamic_report.py
├── identification_analysis/   # Post-run M2 analysis & HTML reports
│   ├── src/
│   │   ├── metrics/           # classification_quality, confidence_scoring, population_statistics
│   │   └── visualization/     # identification_html_report, identification_charts, classification_charts
│   ├── scripts/               # run_identification_report.py
│   └── tests/                 # test_classification_quality, test_population_statistics
├── house_analysis/            # Pre-analysis (data quality checks)
│   ├── src/
│   │   ├── metrics/           # coverage, quality, temporal, power_stats
│   │   ├── reports/           # house_report
│   │   └── visualization/     # html_report, charts
│   └── scripts/               # run_analysis.py (--house, --list)
├── harvesting_data/           # EnergyHive API data acquisition
│   ├── api.py, fetcher.py, storage.py, config.py, cli.py
│   └── fetch_single_house.py
├── user_plot_requests/        # Web visualization (Flask) + Colab notebook
│   ├── app.py                 # Flask server (port 5000)
│   ├── device_plots_colab.ipynb
│   └── plot_by_request.py
├── investigations/            # One-off debug/investigation scripts (gitignored)
└── CLAUDE.md                  # This file
```

## Core Module Details

### core/ — Configuration & Data Loading

| File | Key Exports | Description |
|------|-------------|-------------|
| `config.py` | `ExperimentConfig`, `EXPERIMENTS`, `LEGACY_EXPERIMENTS`, `get_experiment()`, `list_experiments()` | Active experiments (exp010, exp012) + legacy (exp000-exp008) |
| `paths.py` | `PathManager`, `RAW_INPUT_DIRECTORY`, `OUTPUT_ROOT` | Path management, threshold constants |
| `data_loader.py` | `load_power_data()`, `find_house_data_path()`, `find_previous_run_summarized()` | Loads CSV/pkl; handles raw and summarized columns |
| `logging_setup.py` | `setup_logging()` | Per-house log files in `{output}/logs/` |

### disaggregation/detection/ — Event Detection

| File | Key Exports | Description |
|------|-------------|-------------|
| `sharp.py` | `detect_sharp_events()` | Single-minute power jumps >= threshold |
| `gradual.py` | `detect_gradual_events()` | Multi-minute ramps; `partial_factor=0.8`, `max_factor=1.3`, windows=[1,2,3] min |
| `expander.py` | `expand_event()` | Includes adjacent small changes; `expand_factor=0.05` |
| `merger.py` | `merge_overlapping_events()`, `merge_consecutive_on/off_events()` | Merges when both_instantaneous AND gap<=1min |

**Detection flow**: `detect_sharp_events()` → group consecutive → `detect_gradual_events()` → `expand_event()` → `merge_overlapping_events()` → `merge_consecutive_on/off_events()`

### disaggregation/matching/ — ON/OFF Pair Matching

| File | Key Exports | Constants | Description |
|------|-------------|-----------|-------------|
| `stage1.py` | `find_match()` | `MAX_MAGNITUDE_DIFF_FILTER=350W` | Clean matching: progressive windows [15,30,60,120,240,360 min] |
| `stage2.py` | `find_noisy_match()` | `min_allowed = baseline - 200` | Noisy matching: tolerates interference from other devices |
| `stage3.py` | `find_partial_match()` | `min_ratio=1.50` | Partial matching: creates remainder events for unmatched portion |
| `validator.py` | `is_valid_event_removal()`, `build_match_tag()` | `MAX_EVENT_CV=0.30` | CV stability, min power ratio, negative remaining checks |

**Match tags format**: `[NOISY-|PARTIAL-]{EXACT|CLOSE|APPROX|LOOSE}-{SPIKE|QUICK|MEDIUM|EXTENDED}[-CORRECTED]`

**Matching flow**: Stage 1 (clean) → Stage 2 (noisy) → Stage 3 (partial)

### disaggregation/segmentation/ — Power Extraction

| File | Key Exports | Description |
|------|-------------|-------------|
| `processor.py` | `process_phase_segmentation()` | 3 paths: standard, NOISY, PARTIAL. Clips to remaining power. |
| `summarizer.py` | `summarize_segmentation()` | Duration categories: short (<=2min), medium (3-24min), long (>=25min) |
| `evaluation.py` | `calculate_phase_metrics()` | Explained power %, minutes explained |
| `restore.py` | `restore_skipped_to_unmatched()` | Bug #14: restores skipped matches to unmatched |

### identification/ — Session-Level Device Classification (M2)

| File | Key Exports | Description |
|------|-------------|-------------|
| `config.py` | `IdentificationConfig`, `MIN_EVENT_DURATION_MINUTES=3` | Constants for session grouping and classification |
| `session_grouper.py` | `load_all_matches()`, `filter_transient_events()`, `group_into_sessions()` | Load matches from all iterations, filter spikes (<3 min), group by 30-min gap |
| `session_classifier.py` | `classify_sessions()`, `ClassifiedSession` | Priority: boiler → central AC → regular AC → unknown. Phase exclusivity for boilers. |
| `session_output.py` | `build_session_json()` | Session JSON + backward-compatible activations JSON |
| `cleanup.py` | `cleanup_intermediate_files()` | Remove intermediate pkl files |

**M2 flow**: Load all matches → Filter spikes (<3 min) → Group into sessions (30-min gap) → Phase sync (central AC) → Classify → Confidence score → JSON output

### pipeline/ — Orchestration

| File | Function | Description |
|------|----------|-------------|
| `runner.py` | `run_pipeline()` | **Unified entry point** — runs M1 iterations then M2 identification |

Pipeline steps live at `disaggregation/pipeline/*_step.py` and are re-exported from `pipeline/__init__.py`.

## Data Format

**Input**: CSV files with columns `timestamp, 1, 2, 3` (or `w1, w2, w3`)
- Location: `INPUT/HouseholdData/{house_id}.csv` or monthly folders `{house_id}/{house_id}_MM_YYYY.csv`
- Timestamp: `DD/MM/YYYY HH:MM` or ISO format
- Values: watts (W), 1-minute resolution

**Internal (pkl)**: All intermediate files use pickle format
- `on_off_{id}_{MM}_{YYYY}.pkl` — detected events
- `matches_{id}_{MM}_{YYYY}.pkl` — matched ON→OFF pairs
- `unmatched_on/off_{id}_{MM}_{YYYY}.pkl` — unmatched events
- `summarized_{id}_{MM}_{YYYY}.pkl` — remaining power after extraction

**3 phases**: `w1`, `w2`, `w3` represent a 3-phase electrical system in Israeli households.

## Pipeline Flow

```
Module 1 — Disaggregation (iterative):
  Iteration 0 (threshold=2000W):  Raw CSV → Detection → Matching → Segmentation → Evaluation
  Iteration 1 (threshold=1500W):  Remaining power → Detection → Matching → Segmentation → Evaluation
  Iteration 2 (threshold=1100W):  Remaining power → ...
  Iteration 3 (threshold=800W):   Remaining power → ...

Module 2 — Identification (runs once):
  All matches from all iterations
    → Filter spikes (<3 min)
    → Group into sessions (30-min gap)
    → Phase synchronization (central AC detection)
    → Classify (boiler → central AC → regular AC → unknown)
    → Confidence scoring
    → JSON output (device_sessions + device_activations)
```

## Experiments (core/config.py)

**Active experiments** (in `EXPERIMENTS`):

| Name | Threshold | Key Feature |
|------|-----------|-------------|
| **`exp010_dynamic_threshold`** | **[2000,1500,1100,800]** | **DEFAULT — Dynamic threshold + identification + unified JSON** |
| `exp012_nan_imputation` | [2000,1500,1100,800] | exp010 + runtime NaN gap filling |

**Legacy experiments** (in `LEGACY_EXPERIMENTS`, accessed via `get_experiment(name, include_legacy=True)`):
exp000–exp008 document the evolution from baseline to dynamic threshold. Kept for backward compatibility.

**Default experiment**: `exp010_dynamic_threshold`

## How to Run

### Pipeline (single house)
```bash
cd experiment_pipeline
python scripts/test_single_house.py --house_id 305
python scripts/test_single_house.py --house_id 305 --skip_visualization --minimal_output
```

### Pipeline (multiple houses)
```bash
cd experiment_pipeline
python scripts/test_array_of_houses.py --skip_visualization
```

### Standalone Identification (M2 only, on existing M1 output)
```bash
cd experiment_pipeline
python scripts/run_identification.py --experiment_dir OUTPUT/experiments/exp010_XXX --house_id 305
```

### Tests
```bash
cd experiment_pipeline && python -m pytest tests/ -v          # 66 tests
cd identification_analysis && python -m pytest tests/ -v      # 40 tests
```

### Disaggregation Analysis (M1 report)
```bash
cd disaggregation_analysis
python scripts/run_dynamic_report.py --experiment <path>
python scripts/run_analysis.py --experiment <path>  # Full analysis with HTML report
```

### Identification Analysis (M2 report)
```bash
cd identification_analysis
python scripts/run_identification_report.py --experiment_dir <path> --house_id 305
```

### House Pre-Analysis
```bash
cd house_analysis
python scripts/run_analysis.py --house 305 --input <path_to_household_data>
```

### Data Harvesting
```bash
python -m harvesting_data.cli --parallel   # Fetch all houses
python -m harvesting_data.cli --house 305  # Fetch single house
```

## Conventions

- Each module is self-contained with `src/`, `scripts/`, `OUTPUT/`
- `importlib.reload()` is used in `pipeline/runner.py` to update global paths between runs
- Monthly data: pipeline processes per-month files and stores per-month results
- Logging: per-house log files at `{output}/logs/test_{house_id}.log`
- Visualization: 4 rows (Original, Remaining, Segregated, Events) x 3 columns (w1, w2, w3)
- Device classification:
  - **Boiler**: Single-phase, >=25min duration, >=1500W, isolated (no nearby compressor cycles)
  - **Central AC**: 2+ phases synchronized within ±10 min, cycling pattern
  - **Regular AC**: 800W+, 3-30min compressor cycles, single phase

## Known Issues & Bug History

| Bug | Description | Fix | Status |
|-----|-------------|-----|--------|
| #1 | `event_seg` used fixed `on_magnitude` instead of tracking via diffs | Chain device_power through on→event→off | Fixed |
| #2 | Upper clip prevented tracking power increases during events | Removed upper clip on cumsum | Fixed |
| #3 | Negative remaining in iteration 1 due to magnitude mismatch | Use `on_seg[-1]` not `on_magnitude` for event period | Fixed |
| #4 | OFF segment started from `on_magnitude` instead of event_seg end | Chain `device_power` from event_seg to off_seg | Fixed |
| #12 | Missing CV stability check for spiky events | Added `MAX_EVENT_CV=0.30` check in validator | Fixed |
| #13 | Missing min power ratio check for cross-device matches | Added `MIN_EVENT_STABILITY_RATIO=0.50` check | Fixed |
| #14 | Skipped matches deleted without returning to unmatched | `restore_skipped_to_unmatched()` in segmentation | Fixed |
| #15 | `tag == 'NOISY'` never matched real tags like `NOISY-LOOSE-EXTENDED` | Changed to `'NOISY' in tag` substring check | Fixed |
| #16 | Standard events not clipped to remaining → negative remaining → skip | Clip `event_seg` to remaining power | Fixed |
