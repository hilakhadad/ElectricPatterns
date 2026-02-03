# House Analysis

Pre-analysis module for household power data. Assesses data quality, coverage, and temporal patterns before running the segmentation pipeline.

## Quick Start

```bash
cd scripts

# Analyze all houses
python run_analysis.py

# Analyze specific house
python run_analysis.py --house 140

# List available houses
python run_analysis.py --list
```

## Structure

```
house_analysis/
├── scripts/
│   └── run_analysis.py           # Main entry point
├── src/
│   ├── metrics/
│   │   ├── coverage.py           # Data availability metrics
│   │   ├── power_stats.py        # Power statistics
│   │   ├── temporal.py           # Temporal patterns
│   │   └── quality.py            # Quality scoring
│   ├── reports/
│   │   ├── house_report.py       # Per-house analysis
│   │   └── aggregate_report.py   # Cross-house summary
│   └── visualization/
│       ├── html_report.py        # HTML report generation
│       └── charts.py             # Plotly charts
└── OUTPUT/                       # Analysis results (gitignored)
```

## Output

```
OUTPUT/run_{timestamp}/
├── index.html                    # Aggregate report
├── house_{id}.html               # Individual house reports
└── summary.json                  # JSON metrics
```

## Metrics

### Coverage Metrics
- **Total Rows**: Number of data points
- **Date Range**: First to last timestamp
- **Coverage Ratio**: Actual rows / Expected rows
- **Missing Data**: Gaps in time series

### Power Statistics
- Mean, std, min, max per phase (w1, w2, w3)
- Total power distribution
- Phase balance analysis

### Temporal Patterns
- Flat segments (no change periods)
- Daily/weekly patterns
- Anomaly detection

### Quality Score
Composite score (0-100) based on:
- Coverage completeness
- Data consistency
- Phase balance
- Anomaly count

## Usage in Code

```python
from house_analysis.src.reports import analyze_single_house
from house_analysis.src.visualization import generate_single_house_html_report

# Analyze house
analysis = analyze_single_house(house_id="140")

# Generate report
generate_single_house_html_report(analysis, output_path="house_140.html")
```

## Use Cases

1. **Before pipeline**: Check data quality before running segmentation
2. **Data validation**: Identify problematic houses
3. **Coverage check**: Ensure sufficient data for analysis
