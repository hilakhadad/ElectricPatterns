# Experiment Pipeline

Core pipeline for energy consumption analysis. Contains both Module 1 (disaggregation) and Module 2 (identification).

## Structure

```
experiment_pipeline/
├── src/
│   ├── core/                  # Config, paths, data loading, NaN imputation
│   ├── disaggregation/        # Module 1: signal-level processing
│   │   ├── rectangle/         #   Core disaggregation engine
│   │   │   ├── detection/     #     Sharp, gradual, settling, near-threshold, tail extension
│   │   │   ├── matching/      #     Stage 1 (clean), 2 (noisy), 3 (partial), stack matcher
│   │   │   ├── segmentation/  #     Power extraction & evaluation
│   │   │   └── pipeline/      #     Orchestration steps (detection, matching, segmentation, recovery)
│   │   ├── wave_recovery/     #   AC wave pattern recovery (new in exp014+)
│   │   │   ├── detection/     #     Wave pattern detector (sharp rise -> gradual decay)
│   │   │   ├── matching/      #     Cross-phase wave matcher
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
│   │   │   ├── recurring_pattern_classifier.py  # DBSCAN clustering (requires scipy)
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
│   ├── test_single_house.py       # Run pipeline on one house
│   ├── run_local_batch.py         # Run on multiple houses (local, sequential)
│   ├── run_identification.py      # Standalone M2 on existing M1 output
│   ├── run_single_month.py        # Month-level HPC parallelism
│   └── sbatch_run_houses.sh       # SLURM batch submission (HPC)
├── tests/                     # Regression tests (pytest)
└── OUTPUT/experiments/        # Results (gitignored)
```

## Quick Start

```bash
# Run on a single house (default experiment: exp015_hole_repair)
python scripts/test_single_house.py --house_id 305

# With options:
python scripts/test_single_house.py --house_id 305 --skip_visualization --minimal_output

# Explicit experiment:
python scripts/test_single_house.py --house_id 305 --experiment_name exp015_hole_repair

# Run on multiple houses (local, sequential):
python scripts/run_local_batch.py --houses 305,1234,2008
python scripts/run_local_batch.py --shortest 10   # 10 smallest houses by month count

# HPC (SLURM): month-level parallel processing
bash scripts/sbatch_run_houses.sh

# Month-level parallelism (for manual SBATCH usage):
python scripts/run_single_month.py --house_id 305 --month_index 0
python scripts/run_single_month.py --house_id 305 --list_months

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
5. **Settling extension** (exp013+): Extends event boundaries through transient settling periods (e.g., power spikes that quickly settle to steady-state)
6. **Split-OFF merger** (exp013+): Merges split OFF events caused by measurement errors (gap <= 2 min)
7. **Expand + Merge**: Include adjacent changes, merge consecutive events

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

After all threshold iterations complete, optional recovery passes extract additional device power. These run between rectangle matching and M2 identification.

- **Guided recovery** (exp013+): Uses matched AC cycle templates to search for missed compressor cycles in the remaining signal at a lower threshold. Requires >= 3 existing matched cycles in a session to establish a template.
- **Wave recovery** (exp014+): Detects wave-shaped patterns (sharp rise followed by gradual decay) directly from the remaining signal. Uses pattern-based detection, cross-phase matching, and specialized wave segmentation. Located in `disaggregation/wave_recovery/`.
- **Hole repair** (exp015+): Identifies rectangle matches that incorrectly extracted wave-shaped events as flat rectangles (indicated by APPROX/LOOSE + EXTENDED tags and near-zero remaining during the event). Undoes the rectangle extraction, re-detects the wave, and re-extracts it properly.

Recovery output goes to `run_post/house_{id}/` so it is separated from rectangle runs but seamlessly picked up by M2's `session_grouper.load_all_matches()`.

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
   - **Recurring pattern**: DBSCAN clustering on unknown sessions with similar magnitude + duration (requires scipy). Each cluster gets a `pattern_id`
   - **Unknown**: Remaining events grouped by time proximity (30-min gap)
4. **Confidence scoring** - 0-1 score per session based on how well it matches classification criteria
5. **JSON output** - `device_sessions_{house_id}.json` + backward-compatible `device_activations_{house_id}.json`

After observing recurring patterns across houses, additional device categories were added. Three-phase device (likely EV charger) was identified from synchronized high-power patterns across all 3 phases. Future work includes clustering unknown sessions to discover more device types.

## Configuration

Detection parameters (thresholds, matching tolerances, session gaps) are configurable through experiment configurations defined in `src/core/config.py`. The pipeline supports multiple configurations and runs them with the appropriate settings.

The default experiment is `exp015_hole_repair`, which includes all features: NaN imputation, settling extension, split-OFF merger, guided recovery, wave recovery, and hole repair.

### Active Experiments

| Name | Features | Description |
|------|----------|-------------|
| `exp010_dynamic_threshold` | Base dynamic threshold | Threshold schedule [2000,1500,1100,800]W, gradual detection, tail extension |
| `exp012_nan_imputation` | exp010 + NaN filling | Runtime NaN gap filling (ffill<=5min, interp<=60min) |
| `exp013_settling_splitoff` | exp012 + settling + split-OFF + guided recovery | Fixes pits, split shutdowns, searches for missed AC cycles |
| `exp014_wave_recovery` | exp013 + wave recovery | Post-M1 wave pattern detection from remaining signal |
| **`exp015_hole_repair`** | **exp014 + hole repair** | **DEFAULT -- fixes wave-shaped events extracted as rectangles** |

Legacy experiments (exp000-exp008) are available via `get_experiment(name, include_legacy=True)`.

## Output Structure

```
OUTPUT/experiments/<experiment_name>_{timestamp}/
├── experiment_metadata.json
├── run_0/house_{id}/             # Iteration 0 (2000W)
│   ├── on_off/on_off_{id}_{MM}_{YYYY}.pkl
│   ├── matches/matches_{id}_{MM}_{YYYY}.pkl
│   ├── unmatched_on/, unmatched_off/
│   └── summarized/summarized_{id}_{MM}_{YYYY}.pkl
├── run_1/house_{id}/...          # Iteration 1 (1500W)
├── run_2/house_{id}/...          # Iteration 2 (1100W)
├── run_3/house_{id}/...          # Iteration 3 (800W)
├── run_post/house_{id}/          # Recovery passes (wave recovery, hole repair)
│   ├── matches/                  # Wave match records (M1-compatible format)
│   └── summarized/               # Updated remaining after wave extraction
├── device_sessions/device_sessions_{id}.json
├── device_activations/device_activations_{id}.json
├── evaluation/dynamic_evaluation_summary_{id}.csv
├── plots/{id}/
├── logs/test_{id}.log
├── house_timing.csv              # Timing log (batch mode)
└── reports/                      # Generated reports (batch mode)
```

## HPC (SLURM)

The pipeline supports two execution modes on HPC clusters:

### sbatch_run_houses.sh

Submits per-house SLURM jobs with automatic report generation. Two modes per house:

- **Sequential**: Houses listed in `completed_houses.txt` run as a single job (all months + iterations + identification in one process).
- **Monthly**: Houses NOT in the file run month-level parallel jobs (SBATCH array for M1 disaggregation + dependent M2 identification job).

Execution order:
1. **Phase 1** -- Per-house: pre-analysis reports, then pipeline (sequential or monthly parallel)
2. **Phase 2** -- Aggregate reports: house pre-analysis aggregate, segregation aggregate, identification aggregate

```bash
# Usage:
cd experiment_pipeline
bash scripts/sbatch_run_houses.sh

# Monitor:
squeue -u $USER

# Create completed_houses.txt from a previous run:
awk -F',' 'NR>1 && $7=="OK" {print $1}' house_timing.csv > completed_houses.txt
```

### run_single_month.py

Enables month-level parallelism: each month runs detection, matching, and segmentation for all threshold iterations independently. After all months complete, run identification separately.

```bash
# List available months:
python scripts/run_single_month.py --house_id 305 --list_months

# Run by month index (for SBATCH --array):
python scripts/run_single_month.py --house_id 305 --month_index 0

# Run by explicit month:
python scripts/run_single_month.py --house_id 305 --month 07_2021
```

## Tests

```bash
python -m pytest tests/ -v
```

## Legacy

### test_array_of_houses.py (replaced 2026-02-24)

Previously, `scripts/test_array_of_houses.py` was used for batch processing. This has been replaced by `scripts/run_local_batch.py` which replicates the SLURM batch script behavior locally with sequential house processing, automatic report generation, and timing CSV output.

### Boiler Classification Criteria (updated 2026-02-24)

Previously documented a "<=2 events" constraint for boiler classification. This constraint has been removed. Boiler classification now uses: high power (>=1500W), long duration (>=15 min), and isolation from compressor cycling patterns.
