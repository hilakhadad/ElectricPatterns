# Identification Analysis

Post-run analysis module for Module 2 (identification) results. Generates interactive HTML reports showing device classification quality, confidence scores, spike filtering statistics, and session summaries.

## Quick Start

```bash
# Latest experiment
python scripts/run_identification_report.py

# Specific experiment directory
python scripts/run_identification_report.py --experiment <path>

# Specific houses only
python scripts/run_identification_report.py --houses 305,1234

# Include pre-analysis quality scores
python scripts/run_identification_report.py --pre-analysis <house_analysis_output_path>

# Resume (only process new houses)
python scripts/run_identification_report.py --resume <dir>

# Faster (skip activation detail)
python scripts/run_identification_report.py --skip-activations

# Publish mode (for HPC batch, outputs to shared reports directory)
python scripts/run_identification_report.py --experiment <path> --output-dir <reports_dir> --publish identification
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
│   │   ├── cross_house_patterns.py     # Cross-house recurring pattern matching
│   │   └── population_statistics.py    # Cross-house statistics and outlier detection
│   ├── reports/                        # Report generation utilities
│   └── visualization/
│       ├── identification_html_report.py   # Main HTML report generator
│       ├── identification_html_single.py   # Single-house HTML report
│       ├── identification_html_aggregate.py    # Aggregate HTML report
│       ├── identification_charts.py        # Plotly charts (main module)
│       ├── charts_device.py                # Device-type charts
│       ├── charts_session.py               # Session charts
│       ├── charts_confidence.py            # Confidence distribution charts
│       ├── charts_spike.py                 # Spike filtering charts
│       └── classification_charts.py        # Quality section charts
├── tests/
│   ├── test_classification_quality.py  # Quality metric tests
│   ├── test_cross_house_patterns.py    # Cross-house pattern matching tests
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

1. **Spike Filter** - How many transient events were filtered and why
2. **Session Overview** - Total sessions, breakdown by device type (boiler, three-phase device, central AC, regular AC, unknown), average duration
3. **Device Timeline** - Visual timeline of all classified sessions
4. **Magnitude Analysis** - Power distribution per device type
5. **Confidence Scoring** - Distribution of confidence scores, breakdown by criteria
6. **Classification Quality** - Quality flags and potential issues

### Device Classification Categories

| Category | Description |
|----------|-------------|
| **Boiler** | Single-phase high-power heating element |
| **Three-phase device** | Synchronized events across all 3 phases (likely EV charger) |
| **Central AC** | Multi-phase synchronized AC compressor cycles |
| **Regular AC** | Single-phase cycling compressor pattern |
| **Recurring pattern** | DBSCAN-discovered clusters of sessions with similar magnitude + duration |
| **Unknown** | Does not match any known device signature |

### Cross-House Pattern Matching

When generating reports for multiple houses, the report pipeline runs cross-house pattern matching as an intermediate phase:

1. **Phase 1** -- Per-house reports (classification, confidence, quality)
2. **Phase 1.5** -- Cross-house pattern matching: extracts recurring pattern signatures (avg magnitude + duration) from each house, matches across houses using relative tolerance (20% magnitude, 30% duration), groups via connected components (BFS), and assigns global names (Device A, B, C... sorted by magnitude descending)
3. **Phase 2** -- Aggregate HTML report with cross-house device comparisons

Output: `cross_house_patterns.json` + per-house JSONs updated with `global_pattern_name`.

## Tests

```bash
python -m pytest tests/ -v
```
