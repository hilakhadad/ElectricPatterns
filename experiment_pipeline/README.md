# Experiment Pipeline

Core pipeline for energy consumption analysis - detecting, matching, and segmenting power events.

## Structure

```
experiment_pipeline/
├── src/                           # Source code (modular structure)
│   ├── core/                      # Configuration, paths, logging
│   ├── detection/                 # Event detection (sharp, gradual)
│   ├── matching/                  # Event matching (stage1, stage2)
│   ├── segmentation/              # Power segmentation & evaluation
│   ├── visualization/             # Interactive & static plots
│   ├── pipeline/                  # Orchestration (process_* functions)
│   └── legacy/                    # Backup of original code
├── scripts/                       # Execution scripts
│   ├── test_single_house.py       # Run pipeline on one house
│   └── test_array_of_houses.py    # Run on all houses (parallel)
├── tests/                         # Test suite
├── INPUT/HouseholdData/           # Input CSV files (gitignored)
└── OUTPUT/experiments/            # Results (gitignored)
```

## Quick Start

```bash
# Run on single house
python scripts/test_single_house.py

# Run on all houses in parallel
python scripts/test_array_of_houses.py
```

## Algorithm Overview

### Iteration Loop

The pipeline runs multiple iterations. Each iteration:
- **Iteration 0**: Processes raw power data
- **Iteration N**: Processes **remaining power** from previous iteration

Each iteration has 3 stages: **Detection** → **Matching** → **Segmentation**

```
┌─────────────────────────────────────────────────────────────────┐
│                         ITERATION 0                              │
│                                                                  │
│   RAW POWER DATA                                                │
│        ↓                                                        │
│   ┌─────────────┐                                               │
│   │  DETECTION  │  → Find ON/OFF events by threshold            │
│   └─────────────┘                                               │
│        ↓                                                        │
│   ┌─────────────┐                                               │
│   │  MATCHING   │  → Stage1 (clean) → Stage2 (noisy) → Stage3   │
│   └─────────────┘                                               │
│        ↓                                                        │
│   ┌──────────────┐                                              │
│   │ SEGMENTATION │  → Subtract device power                     │
│   └──────────────┘                                              │
│        ↓                                                        │
│   REMAINING POWER ───────────────┐                              │
│                                   ↓                              │
├─────────────────────────────────────────────────────────────────┤
│                         ITERATION 1                              │
│                                   │                              │
│   REMAINING POWER ←──────────────┘                              │
│        ↓                                                        │
│   Same threshold - finds events that were masked before         │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: Detection

**Goal**: Find when devices turned ON and OFF

**Algorithm**:
1. Calculate power difference: `diff = power[t] - power[t-1]`
2. Detect **ON**: diff >= threshold (e.g., 1500W)
3. Detect **OFF**: diff <= -threshold × 0.8
4. **Expand** events - include adjacent small changes in same direction
5. **Merge** consecutive ON/OFF events (multi-stage appliance turn-ons)
6. Filter events below minimum magnitude

**Output**: List of ON/OFF events with timestamp, magnitude, phase

### Stage 2: Matching

**Goal**: Pair each ON event with its corresponding OFF event

#### Stage 1 - Clean Matching (NON-M, SPIKE)
- Find OFF after ON on **same phase**
- **Progressive window**: Start small (15min), expand up to 6 hours
- Require **stable power** between ON and OFF (no other devices)
- Magnitude difference < 350W

#### Stage 2 - Noisy Matching (NOISY)
- Match events that failed Stage 1
- Allow **fluctuations** (other devices running in parallel)
- Condition: power never drops below baseline (device stays on)

#### Stage 3 - Partial Matching (PARTIAL)
- Handle cases where ON magnitude ≠ OFF magnitude (diff > 350W)
- **Match using smaller magnitude**
- **Create remainder event** - the difference returns to next iteration

**Validation**: Every match is checked - subtraction must not create negative values

### Stage 3: Segmentation

**Goal**: Separate device power from total consumption

**Algorithm** for each matched ON/OFF pair:
1. **ON segment**: on_start → on_end (gradual rise)
2. **Steady segment**: on_end → off_start (constant consumption)
3. **OFF segment**: off_start → off_end (gradual descent)

Subtract device power from total → **remaining power** for next iteration

**Output**:
- `summarized`: original, remaining, short/medium/long duration power
- Remaining power file: input for next iteration

---

## Pseudocode

```
ALGORITHM: Power Consumption Segmentation Pipeline

FOR iteration = 0 TO max_iterations:

    input_power = raw_power IF iteration == 0 ELSE remaining_power

    # =========== DETECTION ===========

    FOR each phase IN [w1, w2, w3]:
        diff = input_power[phase].diff()

        on_events  = WHERE diff >= threshold
        off_events = WHERE diff <= -threshold * 0.8

        expand_events()   # include adjacent changes
        merge_events()    # combine multi-stage turn-ons

    # =========== MATCHING ===========

    # Stage 1: Clean Matching
    FOR each on_event:
        FOR window IN [15min, 30min, 60min, 120min, 240min, 360min]:
            candidates = off_events WHERE:
                - same phase
                - off.start > on.end
                - time_diff <= window
                - |magnitude_diff| <= 350W

            FOR each candidate:
                IF stable_power_between AND valid_removal:
                    MATCH as "NON-M" or "SPIKE"
                    BREAK

    # Stage 2: Noisy Matching
    FOR each unmatched_on:
        FOR window IN [...]:
            candidates = off_events WHERE |magnitude_diff| <= 350W

            FOR each candidate:
                IF power_never_drops_below_baseline AND valid_removal:
                    MATCH as "NOISY"
                    BREAK

    # Stage 3: Partial Matching
    FOR each unmatched_on:
        FOR window IN [...]:
            candidates = off_events WHERE |magnitude_diff| > 350W

            FOR each candidate:
                match_magnitude = min(|on.mag|, |off.mag|)
                remainder = |on.mag - off.mag|

                IF valid_removal(match_magnitude):
                    MATCH as "PARTIAL"
                    CREATE remainder_event for next iteration
                    BREAK

    # =========== SEGMENTATION ===========

    remaining_power = copy(input_power)

    FOR each match:
        # Extract ON ramp
        on_power = cumsum(diff[on_start:on_end])
        remaining_power -= on_power

        # Extract steady state
        steady_power = magnitude (constant)
        remaining_power -= steady_power

        # Extract OFF ramp
        off_power = magnitude + cumsum(diff[off_start:off_end])
        remaining_power -= off_power

    # Categorize by duration
    short  = WHERE duration <= 2min
    medium = WHERE 3min <= duration <= 24min
    long   = WHERE duration >= 25min

    SAVE remaining_power for next iteration

RETURN matches, segmented_power, remaining_power
```

---

## Scripts

### test_single_house.py

Run the full pipeline on one house:

```python
# Configuration (edit in script)
HOUSE_ID = "125"                           # House to process
EXPERIMENT_NAME = "exp005_asymmetric_windows"  # Experiment config
MAX_ITERATIONS = 2                          # Number of iterations
```

Can also be imported:
```python
from test_single_house import run_pipeline_for_house

result = run_pipeline_for_house(
    house_id="125",
    experiment_name="exp005_asymmetric_windows",
    output_path="/path/to/output",
    max_iterations=2
)
```

### test_array_of_houses.py

Run pipeline on all houses in parallel:
- Auto-detects houses from `INPUT/HouseholdData/`
- Uses parallel workers
- Creates timestamped output directory

## Module Structure

### core/
Central configuration and utilities:
- `config.py` - ExperimentConfig dataclass, experiment definitions
- `paths.py` - Path constants, PathManager class
- `logging_setup.py` - Logging configuration

### detection/
Event detection algorithms:
- `sharp.py` - Single-minute power jumps
- `gradual.py` - Multi-minute power ramps
- `merger.py` - Merge overlapping events
- `expander.py` - Expand events to include adjacent changes

### matching/
Event matching algorithms:
- `stage1.py` - Clean matching (stable power between events)
- `stage2.py` - Noisy matching (tolerates interference)
- `stage3.py` - Partial matching (magnitude mismatch >350W)
- `stack_matcher.py` - Stack-based ON/OFF matching
- `validator.py` - Match validation

### segmentation/
Power segmentation:
- `processor.py` - Core segmentation logic
- `summarizer.py` - Duration category summarization
- `evaluation.py` - Metrics calculation

### visualization/
Plotting utilities:
- `interactive.py` - Plotly interactive plots
- `static.py` - Matplotlib static plots

### pipeline/
Orchestration functions that tie everything together:
- `process_detection()` - Run detection step
- `process_matching()` - Run matching step
- `process_segmentation()` - Run segmentation step
- `process_evaluation()` - Run evaluation step
- `process_visualization()` - Run visualization step

## Experiments

Defined in `src/core/config.py`:

| Name | Threshold | Gradual | Description |
|------|-----------|---------|-------------|
| exp000_baseline | 1600W | No | Original detection |
| exp001_gradual_detection | 1600W | Yes | Smart gradual detection |
| exp002_lower_TH | 1500W | Yes | Lower threshold |
| exp003_progressive_search | 1500W | Yes | Progressive window search |
| exp004_noisy_matching | 1500W | Yes | + Noisy event matching |
| exp005_asymmetric_windows | 1500W | Yes | Asymmetric time windows |

## Output Structure

```
OUTPUT/experiments/{exp_name}_{timestamp}/
├── experiment_metadata.json    # Configuration used
├── run_0/house_{id}/
│   ├── on_off_1500.csv         # Detected events
│   ├── matches_{id}.csv        # Matched events
│   ├── summarized_{id}.csv     # Segmented data
│   └── evaluation_history_{id}.csv
├── run_1/...
├── plots/{id}/                 # Interactive HTML plots
└── logs/test_{id}.log
```

## Usage Example

```python
from pipeline import (
    process_detection,
    process_matching,
    process_segmentation,
    process_evaluation,
    process_visualization
)
from core import get_experiment

# Get experiment configuration
config = get_experiment("exp005_asymmetric_windows")

# Run pipeline steps
process_detection(house_id="125", run_number=0, threshold=config.threshold, config=config)
process_matching(house_id="125", run_number=0, threshold=config.threshold)
process_segmentation(house_id="125", run_number=0)
process_evaluation(house_id="125", run_number=0, threshold=config.threshold)
process_visualization(house_id="125", run_number=0, threshold=config.threshold)
```
