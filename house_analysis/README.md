# House Analysis

Pre-analysis module for household power data. Assesses data quality, coverage, and temporal patterns **before** running the segmentation pipeline. Generates per-house and aggregate HTML reports with quality scores.

## Quick Start

```bash
cd house_analysis/scripts

python run_analysis.py                        # Analyze all houses
python run_analysis.py --house 140            # Analyze specific house
python run_analysis.py --houses 140,305,2008  # Analyze multiple houses
python run_analysis.py --list                 # List available houses
python run_analysis.py --input-dir <path>     # Custom input directory
python run_analysis.py --publish house --output-dir reports/  # Publish mode
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
│   │   ├── quality.py           # Data quality metrics and anomaly detection
│   │   ├── quality_scoring.py   # Composite quality scoring (0-100)
│   │   └── wave_behavior.py     # Wave/cycling behavior analysis per phase
│   ├── reports/
│   │   ├── house_report.py      # Per-house analysis
│   │   └── aggregate_report.py  # Cross-house aggregate summary
│   └── visualization/
│       ├── html_report.py       # Main HTML report module
│       ├── html_report_single.py    # Single-house HTML report
│       ├── html_report_aggregate.py # Aggregate HTML report
│       ├── html_report_template.py  # HTML template components
│       └── charts.py            # Plotly charts
└── OUTPUT/                      # Analysis results (gitignored)
```

## Quality Score (0-100)

Composite score based on 6 components:

| Component | Max Points | Description |
|-----------|-----------|-------------|
| Sharp Entry Rate | 20 | Fraction of threshold crossings from single-minute jumps |
| Device Signature | 15 | Presence of clear device ON/OFF patterns |
| Power Profile | 20 | Reasonable power distribution and phase balance |
| Variability | 20 | Appropriate hourly variability (not too flat, not too noisy) |
| Data Volume | 15 | Sufficient data coverage across months |
| Data Integrity | 10 | Low NaN rate, few duplicates, consistent timestamps |

After the base score, anomaly penalties are applied for detected issues.

### Faulty Phase Detection
- Phase with >=20% NaN values -> marked as **faulty**
- Dead phase: < 1% of max phase power
- Houses with faulty phases get `quality_label='faulty'` instead of a numeric score

## CLI Arguments

| Argument | Description |
|----------|-------------|
| `--house <id>` | Analyze a specific house |
| `--houses <id1,id2,...>` | Analyze multiple houses (comma-separated) |
| `--list` | List available houses |
| `--input-dir <path>` | Input directory with data files |
| `--output-dir <path>` | Output directory for reports |
| `--publish <name>` | Publish mode: generates `{name}_report.html` + `{name}_reports/` (requires `--output-dir`) |

## Output

```
OUTPUT/run_{timestamp}/
├── index.html            # Aggregate report with filters and score boxes
├── house_{id}.html       # Individual house reports
└── summary.json          # JSON metrics for downstream use
```

Quality scores from this module are used by `disaggregation_analysis` and `identification_analysis` to contextualize pipeline results.
