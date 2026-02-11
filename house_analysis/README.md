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
- **Coverage Ratio**: Actual rows / Expected rows (% of expected minutes present)
- **NaN %**: Average missing values across phases
- **Gap Analysis**: Max gap, % of gaps over 2 minutes
- **Duplicate Timestamps**: Detection of repeated timestamps

### Power Statistics
- Mean power per phase (w1, w2, w3)
- Total power (mean, max)
- Phase balance ratio (max/min)
- Power range distribution (0-100W, 100-500W, 500-1000W, 1000-2000W, 2000W+)

### Temporal Patterns
- Day/Night power ratio (per phase and total)
- Hourly, weekly, monthly patterns (total power)
- Power consumption heatmap (hour x day-of-week)
- Flat segment detection (% of readings with no change)
- Yearly breakdown with monthly detail

### Quality Score (0-100)
Composite score based on 5 components:

| Component | Max Points | Description |
|-----------|-----------|-------------|
| Completeness | 30 | Coverage ratio × 30 |
| Gap Quality | 20 | Deductions for large gaps, high gap %, NaN values |
| Phase Balance | 15 | Based on max/min phase ratio (1-2=15, 2-3=10, 3-5=5, >5=0) |
| Monthly Balance | 20 | Even coverage across months (low std = high score) |
| Low Noise | 15 | Reasonable hourly variability (CV 0.3-0.8 = optimal) |

### Faulty Phase Detection
- Phase with >= 20% NaN values is marked as **faulty** (תקולה)
- Houses with faulty phases get `quality_label='faulty'` instead of a numeric score
- Dead phase detection: phase with < 1% of max phase power

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
