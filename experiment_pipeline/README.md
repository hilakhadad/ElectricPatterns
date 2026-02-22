# Experiment Pipeline

Core pipeline for energy consumption analysis. Contains both Module 1 (disaggregation) and Module 2 (identification).

## Structure

```
experiment_pipeline/
├── src/
│   ├── core/                  # Config, paths, data loading, NaN imputation
│   ├── disaggregation/        # Module 1: signal-level processing
│   │   ├── detection/         #   Sharp, gradual, near-threshold, tail extension
│   │   ├── matching/          #   Stage 1 (clean), 2 (noisy), 3 (partial)
│   │   ├── segmentation/      #   Power extraction & evaluation
│   │   └── pipeline/          #   Orchestration steps (*_step.py)
│   ├── identification/        # Module 2: session-level classification
│   │   ├── config.py          #   Constants (session gap, thresholds)
│   │   ├── session_grouper.py #   Load, filter spikes, group sessions
│   │   ├── session_classifier.py  # Classify: boiler -> central_ac -> regular_ac
│   │   └── session_output.py  #   JSON output builder
│   ├── pipeline/              # Unified runner (runner.py)
│   └── visualization/         # Interactive & static plots
├── scripts/
│   ├── test_single_house.py   # Run pipeline on one house
│   ├── test_array_of_houses.py    # Run on all houses (parallel)
│   ├── run_identification.py  # Standalone M2 on existing M1 output
│   └── run_single_month.py    # Month-level HPC parallelism
├── tests/                     # Regression tests (pytest)
└── OUTPUT/experiments/        # Results (gitignored)
```

## Quick Start

```bash
# Default: dynamic threshold (exp010)
python scripts/test_single_house.py --house_id 305

# With options:
python scripts/test_single_house.py --house_id 305 --skip_visualization --minimal_output

# Run on all houses in parallel
python scripts/test_array_of_houses.py --skip_visualization

# Standalone identification (M2 only, on existing M1 output):
python scripts/run_identification.py --experiment_dir OUTPUT/experiments/exp010_... --house_id 305
```

Both `test_single_house.py` and `test_array_of_houses.py` auto-detect static vs dynamic mode from the experiment config.

## Module 1 - Disaggregation

### Detection

Finds when devices turned ON and OFF:

1. **Sharp detection**: Single-minute power jumps >= threshold
2. **Gradual detection**: Multi-minute ramps (windows: 1-3 min, 80-130% range)
3. **Near-threshold**: Events at 85-100% of threshold
4. **Tail extension**: Extends OFF events through residual power decay (max 10 min)
5. **Expand + Merge**: Include adjacent changes, merge consecutive events

### Matching (3 Stages)

Pairs each ON event with its corresponding OFF event:

| Stage | Name | Condition | Description |
|-------|------|-----------|-------------|
| 1 | Clean | Stable power between ON/OFF | Progressive windows [15, 30, 60, 120, 240, 360 min], magnitude diff < 350W |
| 2 | Noisy | Interference from other devices | Same windows, allows fluctuations, power >= baseline |
| 3 | Partial | ON/OFF magnitude mismatch > 350W | Matches using smaller magnitude, creates remainder for next iteration |

Every match is validated: subtraction must not create negative remaining power.

**Match tags**: `[NOISY-|PARTIAL-]{EXACT|CLOSE|APPROX|LOOSE}-{SPIKE|QUICK|MEDIUM|EXTENDED}[-CORRECTED]`

### Segmentation

Extracts device power from total consumption. For each matched pair:
1. **ON segment**: on_start -> on_end (gradual rise)
2. **Steady segment**: on_end -> off_start (constant consumption)
3. **OFF segment**: off_start -> off_end (gradual descent)

Subtract device power -> **remaining power** for next iteration.

Duration categories: **short** (<=2 min), **medium** (3-24 min), **long** (>=25 min)

## Module 2 - Identification

Runs once after all M1 iterations complete.

### Pipeline

1. **Load all matches** from all iterations (all `matches/*.pkl` files)
2. **Filter spikes** - Remove events < 3 minutes (transient noise from devices not targeted by identification)
3. **Group into sessions** - Events on the same phase within 30-min gap belong to one session
4. **Split sessions** - If short prefix events precede a significantly longer event (>=3x median, >=10 min), split them into separate sessions
5. **Classify** (priority order):
   - **Boiler**: >=1500W, >=15 min avg duration, <=2 events, isolated (no medium events nearby). Phase exclusivity enforced as post-processing (one boiler per household = one phase)
   - **Central AC**: Must first pass AC-candidate pre-filter (>=800W, >=4 cycles, magnitude CV <=30%, duration CV <=40%, gap CV <=50%), then 2+ phases synchronized within 10 min
   - **Regular AC**: 800W+, >=4 cycles (first >=15 min), magnitude CV <=20%. When overall CV fails due to multi-iteration mixing, falls back to per-iteration check
   - **Unknown**: Doesn't match any criteria
6. **Confidence scoring** - 0-1 score per session based on how well it matches classification criteria
7. **JSON output** - `device_sessions_{house_id}.json` + backward-compatible `device_activations_{house_id}.json`

## Experiments

Defined in `src/core/config.py`:

**Active** (`EXPERIMENTS`):

| Name | Threshold | Description |
|------|-----------|-------------|
| **exp010_dynamic_threshold** | [2000, 1500, 1100, 800] | Dynamic threshold + identification (DEFAULT) |
| exp012_nan_imputation | [2000, 1500, 1100, 800] | exp010 + NaN gap filling |

**Legacy** (`LEGACY_EXPERIMENTS`): exp000-exp008, accessed via `get_experiment(name, include_legacy=True)`.

## Output Structure (Dynamic Mode)

```
OUTPUT/experiments/exp010_dynamic_threshold_{timestamp}/
├── experiment_metadata.json
├── run_0_th2000/house_{id}/
│   ├── on_off/on_off_{id}_{MM}_{YYYY}.pkl
│   ├── matches/matches_{id}_{MM}_{YYYY}.pkl
│   ├── unmatched_on/, unmatched_off/
│   └── summarized/summarized_{id}_{MM}_{YYYY}.pkl
├── run_1_th1500/house_{id}/...
├── run_2_th1100/house_{id}/...
├── run_3_th800/house_{id}/...
├── device_sessions/device_sessions_{id}.json
├── device_activations/device_activations_{id}.json
├── evaluation/dynamic_evaluation_summary_{id}.csv
├── plots/{id}/
└── logs/test_{id}.log
```

## Tests

```bash
python -m pytest tests/ -v
```

66 regression tests covering bugs #1-4, #12-16, missing summarized files, and NaN imputation.
