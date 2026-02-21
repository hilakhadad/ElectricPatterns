# House Analysis

Pre-analysis module for household power data. Assesses data quality, coverage, and temporal patterns **before** running the segmentation pipeline. Generates per-house and aggregate HTML reports with quality scores.

## Quick Start

```bash
cd house_analysis/scripts

python run_analysis.py                 # Analyze all houses
python run_analysis.py --house 140     # Analyze specific house
python run_analysis.py --list          # List available houses
```

## Structure

```
house_analysis/
├── scripts/
│   └── run_analysis.py          # Main entry point
├── src/
│   ├── metrics/
│   │   ├── coverage.py          # Data availability, gaps, duplicates
│   │   ├── power_stats.py       # Power statistics per phase
│   │   ├── temporal.py          # Day/night, hourly, weekly patterns
│   │   └── quality.py           # Composite quality scoring (0-100)
│   ├── reports/
│   │   ├── house_report.py      # Per-house analysis
│   │   └── aggregate_report.py  # Cross-house summary
│   └── visualization/
│       ├── html_report.py       # Interactive HTML reports
│       └── charts.py            # Plotly charts
└── OUTPUT/                      # Analysis results (gitignored)
```

## Quality Score (0-100)

Composite score based on 5 components:

| Component | Max Points | Description |
|-----------|-----------|-------------|
| Completeness | 30 | Coverage ratio (actual rows / expected rows) |
| Gap Quality | 20 | Deductions for large gaps, high gap %, NaN values |
| Phase Balance | 15 | Based on max/min phase power ratio |
| Monthly Balance | 20 | Even coverage across months (low std = high score) |
| Low Noise | 15 | Reasonable hourly variability (CV 0.3-0.8 = optimal) |

### Faulty Phase Detection
- Phase with >=20% NaN values -> marked as **faulty**
- Dead phase: < 1% of max phase power
- Houses with faulty phases get `quality_label='faulty'` instead of a numeric score

## Output

```
OUTPUT/run_{timestamp}/
├── index.html            # Aggregate report with filters and score boxes
├── house_{id}.html       # Individual house reports
└── summary.json          # JSON metrics for downstream use
```

Quality scores from this module are used by `disaggregation_analysis` to contextualize pipeline results.
