# Experiment Pipeline

Core pipeline for energy consumption analysis. Contains both Module 1 (disaggregation) and Module 2 (identification).

## Structure

```
experiment_pipeline/
├── src/
│   ├── core/                  # Config, paths, data loading, NaN imputation
│   ├── disaggregation/        # Module 1: signal-level processing
│   │   ├── rectangle/         #   Core disaggregation engine
│   │   │   ├── detection/     #     Sharp, gradual, near-threshold, tail extension
│   │   │   ├── matching/      #     Stage 1 (clean), 2 (noisy), 3 (partial)
│   │   │   ├── segmentation/  #     Power extraction & evaluation
│   │   │   └── pipeline/      #     Orchestration steps (detection, matching, segmentation, recovery)
│   │   ├── wave_recovery/     #   AC wave pattern recovery
│   │   │   ├── detection/     #     Wave pattern detector
│   │   │   ├── matching/      #     Phase matcher for wave patterns
│   │   │   ├── segmentation/  #     Wave segmentor & validator
│   │   │   └── pipeline/      #     Wave recovery step, hole repair, I/O
│   │   ├── detection/         #   (shim - delegates to rectangle/detection)
│   │   ├── matching/          #   (shim - delegates to rectangle/matching)
│   │   ├── segmentation/      #   (shim - delegates to rectangle/segmentation)
│   │   └── pipeline/          #   (shim - delegates to rectangle/pipeline)
│   ├── identification/        # Module 2: session-level classification
│   │   ├── classifiers/       #   Device-specific classifiers
│   │   │   ├── boiler_classifier.py      # Boiler + three-phase detection
│   │   │   ├── ac_classifier.py          # AC cycling pattern detection
│   │   │   ├── central_ac_classifier.py  # Cross-phase AC overlap
│   │   │   ├── unknown_classifier.py     # Unknown confidence/reason
│   │   │   └── scoring_utils.py          # Shared scoring functions
│   │   ├── config.py          #   Constants (session gap, thresholds)
│   │   ├── session_builder.py #   Build sessions from grouped events
│   │   ├── session_grouper.py #   Load, filter spikes, group sessions
│   │   ├── session_classifier.py  # Classify-first orchestrator
│   │   ├── session_output.py  #   JSON output builder
│   │   ├── spike_stats.py     #   Spike filtering statistics
│   │   └── cleanup.py         #   Intermediate file cleanup
│   ├── pipeline/              # Orchestration
│   │   ├── runner.py          #   Main entry point (run_pipeline)
│   │   ├── pipeline_setup.py  #   Experiment setup and configuration
│   │   └── post_pipeline.py   #   Post-pipeline processing
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
# Run on a single house
python scripts/test_single_house.py --house_id 305

# With options:
python scripts/test_single_house.py --house_id 305 --skip_visualization --minimal_output

# Run on all houses in parallel
python scripts/test_array_of_houses.py --skip_visualization

# Standalone identification (M2 only, on existing M1 output):
python scripts/run_identification.py --experiment_dir OUTPUT/experiments/<experiment_folder> --house_id 305
```

## Module 1 - Disaggregation

The disaggregation module processes the aggregate power signal iteratively at decreasing detection thresholds. Each iteration detects and removes large device events, making smaller devices visible in the remaining signal.

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

### Recovery Passes (post-iteration)

After all threshold iterations complete, optional recovery passes extract additional device power:

- **Guided recovery**: Uses M2 classification hints to re-detect events that were missed during standard iterations
- **Wave recovery**: Detects AC cycling wave patterns directly from the remaining signal using pattern-based detection, phase matching, and specialized wave segmentation. Includes hole repair for gaps caused by partial extraction in earlier iterations.

## Module 2 - Identification

Runs once after all M1 iterations (and optional recovery passes) complete. Uses a **classify-first** approach: individual events are classified by device type before being grouped into sessions.

### Pipeline

1. **Load all matches** from all iterations (all `matches/*.pkl` files)
2. **Filter spikes** - Remove transient events (very short duration noise from devices not targeted by identification)
3. **Classify events** (priority order):
   - **Boiler**: High-power (>=1500W), long duration (>=15 min avg), isolated (no compressor cycles nearby). Phase exclusivity enforced as post-processing (one boiler per household = one phase)
   - **Three-phase device**: Boiler-candidate events that have simultaneous high-power events on other phases (likely EV charger or industrial equipment)
   - **Central AC**: AC cycling sessions that overlap across 2+ phases synchronized within 10 min
   - **Regular AC**: 800W+, cycling compressor pattern (>=4 cycles, first >=10 min), consistent magnitude (CV <=20%)
   - **Unknown**: Remaining events grouped by time proximity (30-min gap)
4. **Confidence scoring** - 0-1 score per session based on how well it matches classification criteria
5. **JSON output** - `device_sessions_{house_id}.json` + backward-compatible `device_activations_{house_id}.json`

After observing recurring patterns across houses, additional device categories were added. Three-phase device (likely EV charger) was identified from synchronized high-power patterns across all 3 phases. Future work includes clustering unknown sessions to discover more device types.

## Configuration

Detection parameters (thresholds, matching tolerances, session gaps) are configurable through experiment configurations defined in `src/core/config.py`. The pipeline supports multiple configurations and runs them with the appropriate settings.

## Output Structure

```
OUTPUT/experiments/<experiment_name>_{timestamp}/
├── experiment_metadata.json
├── run_0_th{threshold}/house_{id}/
│   ├── on_off/on_off_{id}_{MM}_{YYYY}.pkl
│   ├── matches/matches_{id}_{MM}_{YYYY}.pkl
│   ├── unmatched_on/, unmatched_off/
│   └── summarized/summarized_{id}_{MM}_{YYYY}.pkl
├── run_1_th{threshold}/house_{id}/...
├── ...
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

## Legacy

### Experiments Table (removed 2026-02-24)

Previously this section listed specific experiment names. Experiments are internal configuration details managed in `src/core/config.py`. The pipeline supports multiple experiment configurations with configurable thresholds, and legacy experiments are preserved for backward compatibility.

### Boiler Classification Criteria (updated 2026-02-24)

Previously documented a "<=2 events" constraint for boiler classification. This constraint has been removed. Boiler classification now uses: high power (>=1500W), long duration (>=15 min), and isolation from compressor cycling patterns.
