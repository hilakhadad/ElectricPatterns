# Disaggregation Analysis

Post-run analysis module for Module 1 (disaggregation) results. Generates interactive HTML reports with matching performance, segmentation quality, device detection patterns, and cross-house comparisons.

Supports both dynamic threshold experiments (exp010, exp012) and static experiments (exp000-exp008).

## Quick Start

```bash
# Dynamic threshold report (exp010) - most common
python scripts/run_dynamic_report.py

# Specific experiment directory
python scripts/run_dynamic_report.py --experiment <path>

# Specific houses only
python scripts/run_dynamic_report.py --houses 305,1234

# Include pre-analysis quality scores
python scripts/run_dynamic_report.py --pre-analysis <house_analysis_output_path>

# Resume (only process new houses)
python scripts/run_dynamic_report.py --resume <analysis_dir>

# Static experiment analysis (legacy)
python scripts/run_analysis.py --experiment <path>
python scripts/run_analysis.py --full   # Full mode with pattern analysis
```

## Structure

```
disaggregation_analysis/
├── scripts/
│   ├── run_dynamic_report.py       # Dynamic threshold HTML reports
│   ├── run_analysis.py             # General analysis (static + dynamic)
│   ├── regenerate_html.py          # Regenerate HTML from existing metrics
│   └── generate_pattern_plots.py   # Generate pattern analysis plots
├── src/
│   ├── metrics/
│   │   ├── matching.py             # Matching rate, tag distribution, duration
│   │   ├── segmentation.py         # Power/minute ratios, negative values
│   │   ├── events.py               # Event count, magnitude distribution
│   │   ├── patterns.py             # Device detection (boiler, AC)
│   │   ├── iterations.py           # Per-iteration progression
│   │   ├── monthly.py              # Monthly breakdown
│   │   ├── classification.py       # Device classification metrics
│   │   └── dynamic_report_metrics.py   # Metrics for dynamic threshold mode
│   ├── reports/
│   │   ├── experiment_report.py    # Per-house report generation
│   │   └── aggregate_report.py     # Cross-house aggregate report
│   └── visualization/
│       ├── dynamic_html_report.py  # HTML report generator (dynamic mode)
│       ├── dynamic_report_charts.py    # Plotly charts for dynamic reports
│       ├── html_report.py          # HTML report generator (static mode)
│       └── charts.py               # Plotly charts (static mode)
└── OUTPUT/                         # Generated reports (gitignored)
```

## Output

```
OUTPUT/analysis_{experiment_name}_{timestamp}/
├── dynamic_report_aggregate.html       # Aggregate report (all houses)
└── house_reports/
    ├── dynamic_report_305.html         # Per-house report
    └── ...
```

## Metrics

- **Matching**: Matching rate %, tag distribution (EXACT/CLOSE/APPROX/LOOSE), duration breakdown
- **Segmentation**: Power segmentation ratio, minutes explained %, negative remaining minutes
- **Iterations**: Per-iteration contribution, cumulative explained power
- **Monthly**: Month-by-month breakdown, seasonal patterns
- **Device detection**: Boiler/AC identification criteria (from patterns module)
