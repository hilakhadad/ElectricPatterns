# Experiment Pipeline

Core pipeline for energy consumption analysis — detecting, matching, segmenting power events, and classifying devices.

## Structure

```
experiment_pipeline/
├── src/
│   ├── core/                      # Config, paths, data loading, NaN imputation
│   ├── detection/                 # Event detection (sharp, gradual, near-threshold, tail)
│   ├── matching/                  # Event matching (stage1, stage2, stage3)
│   ├── segmentation/              # Power segmentation & evaluation
│   ├── classification/            # Device type classification (boiler, AC)
│   ├── output/                    # Unified JSON output builder
│   ├── visualization/             # Interactive & static plots
│   ├── pipeline/                  # Orchestration (runner, process_* steps)
│   └── legacy/                    # Backup of original monolithic code
├── scripts/
│   ├── test_single_house.py       # Run pipeline on one house
│   └── test_array_of_houses.py    # Run on all houses (parallel)
├── tests/                         # Regression tests (pytest)
├── INPUT/HouseholdData/           # Input CSV files (gitignored)
└── OUTPUT/experiments/            # Results (gitignored)
```

## Quick Start

```bash
# Default: dynamic threshold (exp010)
python scripts/test_single_house.py --house_id 305

# With options:
python scripts/test_single_house.py --house_id 305 --skip_visualization --minimal_output

# Static experiment (legacy):
python scripts/test_single_house.py --house_id 305 --experiment_name exp007_symmetric_threshold

# Run on all houses in parallel
python scripts/test_array_of_houses.py --skip_visualization
```

Both scripts auto-detect static vs dynamic mode from the experiment config.

## Algorithm Overview

### Dynamic Threshold Mode (Default)

The default experiment (`exp010_dynamic_threshold`) uses decreasing thresholds per iteration:

```
┌─────────────────────────────────────────────────────────────────┐
│  ITERATION 0 — threshold = 2000W (boilers, large appliances)   │
│                                                                 │
│   RAW POWER DATA                                                │
│        ↓                                                        │
│   Detection → Matching → Segmentation                           │
│        ↓                                                        │
│   REMAINING POWER                                               │
├─────────────────────────────────────────────────────────────────┤
│  ITERATION 1 — threshold = 1500W (strong AC)                   │
│   REMAINING POWER → Detection → Matching → Segmentation        │
├─────────────────────────────────────────────────────────────────┤
│  ITERATION 2 — threshold = 1100W (medium AC)                   │
│   REMAINING POWER → Detection → Matching → Segmentation        │
├─────────────────────────────────────────────────────────────────┤
│  ITERATION 3 — threshold = 800W (small AC)                     │
│   REMAINING POWER → Detection → Matching → Segmentation        │
├─────────────────────────────────────────────────────────────────┤
│  CLASSIFICATION                                                 │
│   All matches → classify (boiler / central_ac / regular_ac)     │
│        ↓                                                        │
│   device_activations_{house_id}.json                            │
└─────────────────────────────────────────────────────────────────┘
```

### Static Mode (Legacy)

Static experiments (exp000–exp008) use a fixed threshold and multiple iterations at the same threshold. Each iteration finds events that were masked before.

### Detection

**Goal**: Find when devices turned ON and OFF

1. Calculate power difference: `diff = power[t] - power[t-1]`
2. Detect **ON**: diff >= threshold
3. Detect **OFF**: diff <= -threshold × off_factor
4. **Expand** events — include adjacent small changes in same direction
5. **Merge** consecutive ON/OFF events (multi-stage appliance turn-ons)
6. **Near-threshold** detection: capture events at 85-100% of threshold
7. **Tail extension** (OFF only): extend through residual power decay (max 10min)

**Output**: ON/OFF events with timestamp, magnitude, duration, phase

### Matching (3 Stages)

**Goal**: Pair each ON event with its corresponding OFF event

**Stage 1 — Clean Matching**
- Find OFF after ON on **same phase**
- **Progressive window**: [15, 30, 60, 120, 240, 360 min]
- Require **stable power** between ON and OFF
- Magnitude difference < 350W

**Stage 2 — Noisy Matching**
- Match events that failed Stage 1
- Allow fluctuations (other devices running in parallel)
- Power never drops below baseline

**Stage 3 — Partial Matching**
- ON magnitude ≠ OFF magnitude (diff > 350W)
- Match using smaller magnitude
- Create remainder event for next iteration

**Validation**: Every match is checked — subtraction must not create negative values.

### Segmentation

**Goal**: Separate device power from total consumption

For each matched ON/OFF pair:
1. **ON segment**: on_start → on_end (gradual rise)
2. **Steady segment**: on_end → off_start (constant consumption)
3. **OFF segment**: off_start → off_end (gradual descent)

Subtract device power from total → **remaining power** for next iteration.

Duration categories: **short** (≤2min), **medium** (3–24min), **long** (≥25min)

### Classification

After all iterations, matched events are classified:
- **Boiler**: Single-phase, ≥1500W, ≥25min, isolated (no nearby compressor cycles)
- **Central AC**: Synchronized across 2+ phases within 10 minutes
- **Regular AC**: 800W+, compressor cycling (3-30min cycles, sessions with 4+ cycles)

## Module Structure

### core/
- `config.py` — ExperimentConfig dataclass, experiment definitions, `get_experiment()`
- `paths.py` — Path constants, PathManager class
- `data_loader.py` — Load CSV/pkl data, find house data paths
- `nan_imputation.py` — Runtime NaN gap filling (ffill ≤5min, interpolate ≤60min)
- `monthly_io.py` — Monthly file I/O utilities
- `logging_setup.py` — Per-house logging configuration

### detection/
- `sharp.py` — Single-minute power jumps ≥ threshold
- `gradual.py` — Multi-minute power ramps (windows: 1-3 min)
- `near_threshold.py` — Events at 85-100% of threshold
- `tail_extension.py` — Extend OFF events through residual decay
- `merger.py` — Merge overlapping and consecutive events
- `expander.py` — Expand events to include adjacent changes

### matching/
- `stage1.py` — Clean matching (stable power between events)
- `stage2.py` — Noisy matching (tolerates interference)
- `stage3.py` — Partial matching (magnitude mismatch >350W)
- `stack_matcher.py` — Stack-based ON/OFF matching
- `validator.py` — Match validation (CV stability, min power ratio, negative check)

### segmentation/
- `processor.py` — Core segmentation logic (standard, noisy, partial paths)
- `summarizer.py` — Duration category summarization
- `evaluation.py` — Metrics calculation (explained power %, minutes)
- `restore.py` — Restore skipped matches to unmatched files
- `errors.py` — Segmentation error types

### classification/
- `device_classifier.py` — Classify matches into boiler, central_ac, regular_ac

### output/
- `activation_builder.py` — Build unified `device_activations_{house_id}.json`

### pipeline/
- `runner.py` — **Unified entry point** (`run_pipeline()`), handles static + dynamic mode
- `detection.py` — `process_detection()`, orchestrates per-phase detection
- `matching.py` — `process_matching()`, runs 3 matching stages
- `segmentation.py` — `process_segmentation()`, month-by-month processing
- `evaluation.py` — `process_evaluation()`, per-phase metrics
- `visualization.py` — `process_visualization()`, interactive HTML plots
- `evaluation_summary.py` — `generate_dynamic_evaluation_summary()` (dynamic mode)
- `cleanup.py` — `cleanup_intermediate_files()` for `--minimal_output` mode

### visualization/
- `interactive.py` — Plotly interactive plots
- `static.py` — Matplotlib static plots

## Experiments

Defined in `src/core/config.py`:

| Name | Threshold | Description |
|------|-----------|-------------|
| exp000_baseline | 1600W | Original detection |
| exp001_gradual_detection | 1600W | + Gradual detection |
| exp002_lower_TH | 1500W | Lower threshold |
| exp003_progressive_search | 1500W | + Progressive window search |
| exp004_noisy_matching | 1500W | + Stage 2 noisy matching |
| exp005_asymmetric_windows | 1500W | Asymmetric time windows |
| exp006_partial_matching | 1500W | + Stage 3 partial matching |
| exp007_symmetric_threshold | 1300W | Symmetric ON/OFF (factor=1.0) |
| exp008_tail_extension | 1300W | + Tail extension for OFF events |
| **exp010_dynamic_threshold** | **[2000,1500,1100,800]** | **Dynamic threshold + classification (DEFAULT)** |
| exp012_nan_imputation | [2000,1500,1100,800] | exp010 + runtime NaN gap filling |

## Output Structure

### Dynamic mode (exp010)
```
OUTPUT/experiments/exp010_dynamic_threshold_{timestamp}/
├── experiment_metadata.json
├── iteration_0_TH2000/house_{id}/
│   ├── on_off_{id}_{MM}_{YYYY}.pkl
│   ├── matches/matches_{id}_{MM}_{YYYY}.pkl
│   ├── unmatched_on/unmatched_on_{id}_{MM}_{YYYY}.pkl
│   ├── unmatched_off/unmatched_off_{id}_{MM}_{YYYY}.pkl
│   └── summarized/summarized_{id}_{MM}_{YYYY}.pkl
├── iteration_1_TH1500/house_{id}/...
├── iteration_2_TH1100/house_{id}/...
├── iteration_3_TH800/house_{id}/...
├── classification/house_{id}/device_activations_{id}.json
├── evaluation/dynamic_evaluation_summary_{id}.csv
├── plots/{id}/                    # Interactive HTML plots
└── logs/test_{id}.log
```

### Static mode (legacy)
```
OUTPUT/experiments/{exp_name}_{timestamp}/
├── experiment_metadata.json
├── run_0/house_{id}/
│   ├── on_off_{id}_{MM}_{YYYY}.pkl
│   ├── matches/, unmatched_on/, unmatched_off/
│   └── summarized/
├── run_1/...
├── plots/{id}/
└── logs/test_{id}.log
```

## Tests

```bash
python -m pytest tests/ -v
```

20 regression tests covering documented bugs (#1-4, #12-16).
