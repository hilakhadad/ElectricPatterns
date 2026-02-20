# Experiment Analysis

Post-run analysis module. Takes the output of `experiment_pipeline` and generates interactive HTML reports with matching performance, segmentation quality, device detection, and cross-house comparisons.

Supports both static experiments (exp000–exp008) and dynamic threshold experiments (exp010, exp012).

## Quick Start

```bash
cd experiment_analysis/scripts

# Static experiments — analyze latest run
python run_analysis.py                              # Fast mode
python run_analysis.py --full                       # Full mode with pattern analysis
python run_analysis.py --houses 140,125,1001        # Specific houses only
python run_analysis.py --experiment /path/to/exp    # Specific experiment directory

# Dynamic threshold experiments (exp010/exp012)
python run_dynamic_report.py                                    # Latest exp010 experiment
python run_dynamic_report.py --experiment /path/to/exp010       # Specific experiment
python run_dynamic_report.py --houses 305,1234                  # Specific houses
python run_dynamic_report.py --resume /path/to/analysis_dir     # Resume: only new houses
python run_dynamic_report.py --pre-analysis /path/to/quality    # Include quality scores
```

## Structure

```
experiment_analysis/
├── scripts/
│   ├── run_analysis.py             # Static experiment analysis
│   ├── run_dynamic_report.py       # Dynamic threshold report generation
│   ├── generate_pattern_plots.py   # Detailed pattern visualizations
│   ├── regenerate_html.py          # Regenerate HTML from cached metrics
│   └── analyze_logs.py             # Analyze pipeline log files
├── src/
│   ├── metrics/
│   │   ├── matching.py             # Matching rate, tag distribution
│   │   ├── segmentation.py         # Segmentation ratio, power explained
│   │   ├── events.py               # Event classification
│   │   ├── patterns.py             # Device detection (AC, boiler)
│   │   ├── classification.py       # Device classification quality metrics
│   │   ├── iterations.py           # Iteration progression analysis
│   │   ├── monthly.py              # Monthly breakdown
│   │   └── dynamic_report_metrics.py  # Metrics for dynamic threshold reports
│   ├── reports/
│   │   ├── experiment_report.py    # Single house report generation
│   │   └── aggregate_report.py     # Cross-house aggregation
│   └── visualization/
│       ├── html_report.py          # Static experiment HTML reports
│       ├── charts.py               # Plotly charts for static reports
│       ├── dynamic_html_report.py  # Dynamic threshold HTML reports
│       ├── dynamic_report_charts.py # Plotly charts for dynamic reports
│       └── pattern_plots.py        # Pattern visualizations
└── OUTPUT/                         # Analysis results (gitignored)
```

## Output

```
OUTPUT/analysis_{experiment_name}_{timestamp}/
├── index.html                          # Aggregate report (all houses)
├── house_{id}.html                     # Individual house reports
├── house_reports/                      # Dynamic reports per house
│   └── dynamic_report_{id}.html
├── dynamic_report_aggregate.html       # Dynamic aggregate report
├── aggregate_metrics.json              # Aggregated statistics
└── plots/                              # Generated charts
```

## Metrics

### Matching Metrics
- **Matching Rate**: % of ON events successfully matched to an OFF event
- **Tag Distribution**: Breakdown by match quality (EXACT/CLOSE/APPROX/LOOSE) and type (clean/NOISY/PARTIAL)
- **Duration Distribution**: Short (≤2min) / Medium (3–24min) / Long (≥25min)

### Segmentation Metrics
- **Power Segmentation Ratio**: Segmented power / Total power
- **Minutes Segmentation Ratio**: Matched minutes / Total minutes
- **High-Power Energy Explained**: % of above-threshold minutes where remaining dropped below threshold
- **Negative Value Count**: Quality indicator for segmentation errors

### Device Detection
- **Boiler**: Single-phase, ≥1500W, ≥25min, isolated (no compressor cycles nearby)
- **Central AC**: Events synchronized across 2+ phases within 10 minutes
- **Regular AC**: 800W+, compressor cycling (3-30min cycles, 4+ cycles per session)

### Pre-Quality Score
Loaded from `house_analysis` output (0–100 or "faulty"). Houses with faulty phases (≥20% NaN) displayed as "Faulty" instead of a numeric score.
