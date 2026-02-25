# Disaggregation Analysis

Post-run analysis module for Module 1 (disaggregation) results. Generates interactive HTML reports with matching performance, segmentation quality, device detection patterns, and cross-house comparisons.

## Quick Start

```bash
# Dynamic threshold report — most common
python scripts/run_dynamic_report.py

# Specific experiment directory
python scripts/run_dynamic_report.py --experiment <path>

# Specific houses only
python scripts/run_dynamic_report.py --houses 305,1234

# Include pre-analysis quality scores
python scripts/run_dynamic_report.py --pre-analysis <house_analysis_output_path>

# Resume (only process new houses)
python scripts/run_dynamic_report.py --resume <analysis_dir>

# Publish mode (for HPC batch, outputs to shared reports directory)
python scripts/run_dynamic_report.py --experiment <path> --output-dir <reports_dir> --publish segregation
```

## Structure

```
disaggregation_analysis/
├── scripts/
│   ├── run_dynamic_report.py       # Main entry point — dynamic threshold HTML reports
│   ├── regenerate_html.py          # Regenerate HTML from existing metrics
│   ├── generate_pattern_plots.py   # Generate pattern analysis plots
│   └── analyze_logs.py             # Analyze pipeline log files
├── src/
│   ├── metrics/
│   │   ├── matching.py             # Matching rate, tag distribution, duration
│   │   ├── segmentation.py         # Power/minute ratios, negative values
│   │   ├── events.py               # Event count, magnitude distribution
│   │   ├── remaining_events.py     # Analysis of events remaining after pipeline
│   │   ├── patterns.py             # Device detection (boiler, AC)
│   │   ├── pattern_detection.py    # Pattern detection logic
│   │   ├── pattern_ac.py           # AC-specific pattern metrics
│   │   ├── pattern_boiler.py       # Boiler-specific pattern metrics
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
│       ├── html_report.py          # HTML report generator (shared utilities)
│       ├── html_report_single.py   # Single-house HTML report
│       ├── html_report_aggregate.py    # Aggregate HTML report
│       ├── html_templates.py       # HTML template components
│       ├── pattern_plots.py        # Pattern visualization plots
│       └── charts.py              # Plotly charts (shared)
├── legacy/
│   └── run_analysis.py            # Legacy static experiment analysis
└── OUTPUT/                        # Generated reports (gitignored)
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
- **Remaining events**: Analysis of what remains after all pipeline iterations

## Legacy

### run_analysis.py (moved 2026-02-24)

The original `scripts/run_analysis.py` entry point for static experiment analysis has been moved to `legacy/run_analysis.py`. Use `scripts/run_dynamic_report.py` for all current analysis.
