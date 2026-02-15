# Device Activations Visualization - Dynamic Threshold

## Quick Start

### 1. Upload to Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `device_plots_dynamic_threshold.ipynb`
3. Mount your Google Drive (first cell)
4. Upload the experiment directory to your Drive

### 2. Configure
Edit these variables in cell "2. Configuration":
```python
HOUSE_ID = "305"
EXPERIMENT_DIR = "/content/drive/MyDrive/experiments/exp010_20260215_120000"
```

### 3. Run All Cells
- Click Runtime → Run all
- Or run cells one by one

## What You'll See

### Timeline Overview
Interactive timeline showing all activations across 3 phases (w1, w2, w3):
- **Matched events**: Colored by device type (boiler=red, AC=blue/green)
- **Unmatched events**: Gray
- Hover to see details

### Individual Plots
For each activation:
- **Green line**: ON event (start/end + magnitude)
- **Red line**: OFF event (start/end + magnitude)
- **Blue area**: Per-minute device power consumption

## Filtering Options

```python
# Show only boilers
FILTER_DEVICE_TYPE = "boiler"

# Show only first iteration (2000W threshold)
FILTER_ITERATION = 0

# Show only phase w1
FILTER_PHASE = "w1"

# Show only matched events
FILTER_MATCH_TYPE = "matched"

# Date range
FILTER_START_DATE = "2024-01-01"
FILTER_END_DATE = "2024-01-31"
```

## Export to Files

Cell "9. Export to Files" saves:
- **HTML files**: Interactive plots (download to view)
- **PNG files**: Static images (view directly in Drive)

Output location: `./plots/activations_{HOUSE_ID}/`

## Differences from Old Notebook

| Old (`device_plots_colab.ipynb`) | New (`device_plots_dynamic_threshold.ipynb`) |
|----------------------------------|---------------------------------------------|
| Loads from pkl files (summarized, matches, on_off) | Loads from unified JSON (`device_activations_{house_id}.json`) |
| Manual date entry (copy/paste) | Automatic - all activations loaded from JSON |
| 4-row plot (Original, Remaining, Segregated, Events) | Timeline overview + individual activation plots |
| Shows full 12-hour window around event | Shows exact activation duration + power profile |
| Needs to load large summarized files | Fast - loads only JSON file |

## When to Use Which Notebook

**Use the OLD notebook** (`device_plots_colab.ipynb`) when:
- You want to see the full pipeline output (Original → Remaining → Segregated)
- You need 12-hour context around specific events
- You're working with static threshold experiments (exp007)

**Use the NEW notebook** (`device_plots_dynamic_threshold.ipynb`) when:
- You want to explore all detected devices across thresholds
- You want to filter/search activations
- You want to see per-device power profiles
- You're working with dynamic threshold experiments (exp010)

## Troubleshooting

### "Device activations JSON not found"
- Check that `EXPERIMENT_DIR` is correct
- Make sure the experiment was run with the dynamic threshold pipeline (`test_dynamic_threshold.py`)
- The JSON file should be: `{EXPERIMENT_DIR}/device_activations_{HOUSE_ID}.json`

### No activations shown
- Check your filters - you may be filtering out everything
- Set all filters to `None` to see all activations
- Check the "Explore the Data" section to see what's available

### Too slow / hangs
- Reduce `MAX_ACTIVATIONS_TO_PLOT` (default 20)
- Add more restrictive filters

## Example Workflows

### Find all boilers in January 2024
```python
FILTER_DEVICE_TYPE = "boiler"
FILTER_START_DATE = "2024-01-01"
FILTER_END_DATE = "2024-01-31"
```

### Compare first vs last iteration
Run twice with:
```python
# First run
FILTER_ITERATION = 0  # 2000W

# Second run
FILTER_ITERATION = 3  # 800W
```

### Export all central AC activations
```python
FILTER_DEVICE_TYPE = "central_ac"
MAX_ACTIVATIONS_TO_PLOT = 100  # increase limit
```
Then run cell "9. Export to Files"
