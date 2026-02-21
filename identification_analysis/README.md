# Identification Analysis

Post-run analysis module for Module 2 (identification) results. Generates interactive HTML reports showing device classification quality, confidence scores, spike filtering statistics, and session summaries.

## Quick Start

```bash
# Latest exp010 experiment
python scripts/run_identification_report.py

# Specific experiment directory
python scripts/run_identification_report.py --experiment <path>

# Specific houses only
python scripts/run_identification_report.py --houses 305,1234

# Resume (only process new houses)
python scripts/run_identification_report.py --resume <dir>

# Faster (skip activation detail)
python scripts/run_identification_report.py --skip-activations
```

## Structure

```
identification_analysis/
├── scripts/
│   └── run_identification_report.py    # Generate HTML reports
├── src/
│   ├── metrics/
│   │   ├── classification.py           # Session-level classification metrics
│   │   ├── classification_quality.py   # Quality flags (phase consistency, magnitude, duration)
│   │   ├── confidence_scoring.py       # Confidence score analysis and distribution
│   │   └── population_statistics.py    # Cross-house statistics and outlier detection
│   └── visualization/
│       ├── identification_html_report.py   # Main HTML report generator
│       ├── identification_charts.py        # Plotly charts (sessions, spikes, magnitude, timeline)
│       └── classification_charts.py        # Quality section charts
├── tests/
│   ├── test_classification_quality.py  # Quality metric tests
│   └── test_population_statistics.py   # Population statistics tests
└── OUTPUT/                             # Generated reports (gitignored)
```

## Output

```
OUTPUT/identification_{experiment_name}_{timestamp}/
├── identification_report_aggregate.html    # Aggregate report (all houses)
└── house_reports/
    ├── identification_report_305.html      # Per-house report
    └── ...
```

## Report Sections

Each per-house report includes:

1. **Spike Filter** - How many transient events (<3 min) were filtered and why
2. **Session Overview** - Total sessions, breakdown by device type, average duration
3. **Device Timeline** - Visual timeline of all classified sessions
4. **Magnitude Analysis** - Power distribution per device type
5. **Confidence Scoring** - Distribution of confidence scores, breakdown by criteria
6. **Classification Quality** - Quality flags and potential issues

## Tests

```bash
python -m pytest tests/ -v
```

40 tests covering classification quality metrics and population statistics.
