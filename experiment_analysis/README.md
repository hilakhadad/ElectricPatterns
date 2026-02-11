# Experiment Analysis

Analyzes the results of household power segmentation experiments. Generates comprehensive reports with matching performance, segmentation quality, and device detection.

## Quick Start

```bash
cd scripts

# Analyze latest experiment (fast mode)
python run_analysis.py

# Full analysis with all metrics
python run_analysis.py --full

# Analyze specific houses
python run_analysis.py --houses 140,125,1001

# Analyze specific experiment
python run_analysis.py --experiment /path/to/experiment
```

## Structure

```
experiment_analysis/
├── scripts/
│   ├── run_analysis.py           # Main entry point
│   └── generate_pattern_plots.py # Detailed pattern visualizations
├── src/
│   ├── metrics/                  # Metric calculations
│   │   ├── matching.py           # Matching performance
│   │   ├── segmentation.py       # Segmentation quality
│   │   ├── events.py             # Event classification
│   │   ├── patterns.py           # Device detection (AC, boiler)
│   │   ├── iterations.py         # Iteration progression
│   │   └── monthly.py            # Monthly breakdown
│   ├── reports/
│   │   ├── experiment_report.py  # Single house report
│   │   └── aggregate_report.py   # Cross-house aggregation
│   └── visualization/
│       ├── html_report.py        # Interactive HTML reports
│       ├── charts.py             # Plotly charts
│       └── pattern_plots.py      # Pattern visualizations
└── OUTPUT/                       # Analysis results (gitignored)
```

## Output

```
OUTPUT/analysis_{timestamp}/
├── index.html                    # Main report with all houses
├── house_{id}.html               # Individual house reports
├── aggregate_metrics.json        # Aggregated statistics
└── plots/                        # Generated charts
```

## Metrics

### Matching Metrics
- **Matching Rate**: % of ON events successfully matched
- **Tag Distribution**: NON-M, SPIKE, NOISY, PARTIAL breakdown
- **Duration Distribution**: Short/Medium/Long events

### Segmentation Metrics
- **Power Segmentation Ratio**: Segmented power / Total power
- **Minutes Segmentation Ratio**: Matched minutes / Total minutes
- **High-Power Energy Explained**: % of minutes above threshold where remaining dropped below threshold
- **Negative Value Count**: Quality indicator

### Pre-Quality Score
- Loaded from house_analysis output (0-100 or 'faulty')
- Houses with faulty phases (>= 20% NaN) displayed as "Faulty" instead of numeric score

### Device Detection

#### Central AC
- Events synchronized across all 3 phases
- Within 10 minutes of each other

#### Regular AC
Strict criteria to reduce false positives:
- Power >= 800W
- Cycle duration: 3-30 minutes
- Session: 2+ cycles
- Total session: 30+ minutes
- Magnitude consistency: std < 20%

#### Boiler
- Single phase, high power
- Long duration (>30 min)
- Typically morning/evening usage

## Usage in Code

```python
from experiment_analysis.src.reports import analyze_experiment_house
from experiment_analysis.src.visualization import generate_html_report

# Analyze single house
analysis = analyze_experiment_house(
    experiment_dir="/path/to/experiment",
    house_id="140",
    run_number=0
)

# Generate HTML report
generate_html_report(analysis, output_path="house_140.html")
```

## Charts

The HTML reports include interactive Plotly charts:
- Matching Rate Distribution
- Segmentation Ratio Distribution
- Tag Breakdown (pie chart)
- Duration Distribution
- Device Detection Summary
