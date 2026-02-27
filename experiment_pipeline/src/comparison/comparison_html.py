"""
HTML comparison report generator.

Produces a standalone HTML file with:
1. Configuration comparison table (feature flags per experiment)
2. Aggregate performance summary
3. Per-house heatmap (houses x experiments → explained_pct)
4. Device type distribution table
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Import shared CSS
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root / 'shared'))
try:
    from html_utils import get_unified_css, EP_TEXT_SECONDARY
except ImportError:
    # Fallback if shared not on path
    def get_unified_css():
        return ""
    EP_TEXT_SECONDARY = '#7D7D92'


def _pct_to_color(value: float) -> str:
    """Map a percentage (0-100) to a green←→red color.

    100% = deep green, 0% = light red. NaN = gray.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '#f0f0f0'

    value = max(0, min(100, value))
    # Interpolate hue from red (0) to green (120) in HSL
    # Low value → warm (red/orange), high value → cool (green)
    hue = int(value * 1.2)  # 0→0 (red), 100→120 (green)
    return f'hsl({hue}, 55%, 85%)'


def _format_val(val, fmt: str = '.1f') -> str:
    """Format a value for display, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '<span style="color:#ccc;">-</span>'
    if isinstance(val, float):
        return f'{val:{fmt}}'
    return str(val)


def generate_comparison_html(
    per_house_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    output_path: str,
    experiments_data: List[dict],
) -> None:
    """Generate a standalone HTML comparison report.

    Args:
        per_house_df: Per-house metrics DataFrame.
        aggregate_df: Aggregate metrics DataFrame.
        output_path: Path to write the HTML file.
        experiments_data: List of experiment data dicts (from load_experiment_data).
    """
    css = get_unified_css()
    exp_names = aggregate_df['exp_id'].tolist() if not aggregate_df.empty else []
    n_experiments = len(exp_names)

    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Comparison</title>
    <style>
        {css}
        .heatmap-cell {{
            text-align: center;
            font-weight: 600;
            font-size: 0.85em;
            padding: 6px 10px !important;
        }}
        .config-check {{ color: #28a745; font-weight: bold; }}
        .config-cross {{ color: #ccc; }}
        .best-value {{ font-weight: 700; text-decoration: underline; }}
        .delta-positive {{ color: #28a745; font-size: 0.8em; }}
        .delta-negative {{ color: #dc3545; font-size: 0.8em; }}
    </style>
</head>
<body>
<div class="container">

<header>
    <h1>Experiment Comparison</h1>
    <div class="subtitle">Side-by-side comparison of {n_experiments} experiments</div>
    <div class="info-bar">
        <span class="info-item">Generated: <strong>{now}</strong></span>
        <span class="info-item">Experiments: <strong>{', '.join(exp_names)}</strong></span>
    </div>
</header>
"""]

    # --- Section 1: Configuration Comparison ---
    html_parts.append(_build_config_section(experiments_data))

    # --- Section 2: Aggregate Summary ---
    html_parts.append(_build_aggregate_section(aggregate_df))

    # --- Section 3: Per-house Heatmap ---
    if not per_house_df.empty:
        html_parts.append(_build_heatmap_section(per_house_df, exp_names))

    # --- Section 4: Device Distribution ---
    if not aggregate_df.empty:
        html_parts.append(_build_device_section(aggregate_df))

    html_parts.append(f"""
<footer>
    ElectricPatterns &mdash; Experiment Comparison &mdash; {now}
</footer>

</div>
</body>
</html>""")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_config_section(experiments_data: List[dict]) -> str:
    """Build the configuration comparison table."""
    if not experiments_data:
        return ''

    features_to_show = [
        ('use_nan_imputation', 'NaN Imputation'),
        ('use_settling_extension', 'Settling Extension'),
        ('use_split_off_merger', 'Split-OFF Merger'),
        ('use_guided_recovery', 'Guided Recovery'),
        ('use_wave_recovery', 'Wave Recovery'),
        ('use_normalization', 'Normalization'),
    ]

    # Header row
    header_cells = '<th>Feature</th>'
    for exp in experiments_data:
        header_cells += f'<th>{exp["exp_id"]}</th>'

    # Feature rows
    rows_html = ''
    for flag, label in features_to_show:
        cells = f'<td style="font-weight:600;">{label}</td>'
        for exp in experiments_data:
            cfg = exp['metadata'].get('experiment', {})
            val = cfg.get(flag, False)
            if flag == 'use_normalization' and val:
                method = cfg.get('normalization_method', '?')
                cells += f'<td class="config-check">{method}</td>'
            elif val:
                cells += '<td class="config-check">&#10003;</td>'
            else:
                cells += '<td class="config-cross">&#10007;</td>'
        rows_html += f'<tr>{cells}</tr>\n'

    # Threshold schedule row
    cells = '<td style="font-weight:600;">Threshold Schedule</td>'
    for exp in experiments_data:
        cfg = exp['metadata'].get('experiment', {})
        schedule = cfg.get('threshold_schedule', [])
        cells += f'<td>{schedule}</td>'
    rows_html += f'<tr>{cells}</tr>\n'

    return f"""
<section>
    <h2>Configuration Comparison</h2>
    <table class="data-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</section>"""


def _build_aggregate_section(aggregate_df: pd.DataFrame) -> str:
    """Build the aggregate performance summary table."""
    if aggregate_df.empty:
        return ''

    header = """<th>Experiment</th><th>Houses</th>
        <th>Mean Explained %</th><th>Median</th><th>P25</th><th>P75</th>
        <th>Classified Rate</th><th>Mean Time (s)</th><th>Key Features</th>"""

    rows_html = ''
    # Find best values for highlighting
    best_mean = aggregate_df['mean_explained_pct'].max()
    best_classified = aggregate_df['mean_classified_rate'].max()

    for _, row in aggregate_df.iterrows():
        mean_cls = ' class="best-value"' if row['mean_explained_pct'] == best_mean else ''
        cr_cls = ' class="best-value"' if row['mean_classified_rate'] == best_classified else ''

        rows_html += f"""<tr>
            <td><strong>{row['exp_id']}</strong><br>
                <span style="font-size:0.75em;color:{EP_TEXT_SECONDARY};">{row.get('description', '')[:60]}</span></td>
            <td>{row['n_houses']}</td>
            <td{mean_cls}>{_format_val(row['mean_explained_pct'])}</td>
            <td>{_format_val(row['median_explained_pct'])}</td>
            <td>{_format_val(row['p25_explained_pct'])}</td>
            <td>{_format_val(row['p75_explained_pct'])}</td>
            <td{cr_cls}>{_format_val(row['mean_classified_rate'], '.1%')}</td>
            <td>{_format_val(row['mean_elapsed_sec'], '.0f')}</td>
            <td style="font-size:0.8em;">{row.get('key_features', '')}</td>
        </tr>\n"""

    return f"""
<section>
    <h2>Aggregate Performance Summary</h2>
    <table class="data-table">
        <thead><tr>{header}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</section>"""


def _build_heatmap_section(per_house_df: pd.DataFrame, exp_names: List[str]) -> str:
    """Build per-house heatmap: rows=houses, columns=experiments, cells=explained_pct."""
    if per_house_df.empty:
        return ''

    houses = sorted(per_house_df['house_id'].unique())

    # Header
    header_cells = '<th>House</th>'
    for exp in exp_names:
        header_cells += f'<th>{exp}</th>'

    # Pivot for easy lookup
    pivot = per_house_df.pivot_table(
        index='house_id', columns='exp_id', values='avg_explained_pct',
    )

    rows_html = ''
    for house in houses:
        cells = f'<td style="font-weight:600;">{house}</td>'
        values = []
        for exp in exp_names:
            val = pivot.loc[house, exp] if (house in pivot.index and exp in pivot.columns) else np.nan
            values.append(val)

        # Find best value for this house
        valid_values = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        best_val = max(valid_values) if valid_values else None

        for val in values:
            bg = _pct_to_color(val)
            is_best = (best_val is not None and val == best_val and len(valid_values) > 1)
            bold = ' best-value' if is_best else ''
            cells += f'<td class="heatmap-cell{bold}" style="background:{bg};">{_format_val(val)}</td>'

        rows_html += f'<tr>{cells}</tr>\n'

    # Summary row
    cells = '<td style="font-weight:700;">Mean</td>'
    for exp in exp_names:
        if exp in pivot.columns:
            mean_val = pivot[exp].mean()
            bg = _pct_to_color(mean_val)
            cells += f'<td class="heatmap-cell" style="background:{bg};font-weight:700;">{_format_val(mean_val)}</td>'
        else:
            cells += '<td class="heatmap-cell" style="background:#f0f0f0;">-</td>'
    rows_html += f'<tr style="border-top:2px solid #666;">{cells}</tr>\n'

    return f"""
<section>
    <h2>Per-House Segregation Heatmap</h2>
    <p style="color:{EP_TEXT_SECONDARY};font-size:0.85em;margin-bottom:12px;">
        Weighted average explained % across phases. Best value per house is <u>underlined</u>.
    </p>
    <div style="overflow-x:auto;">
    <table class="data-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
</section>"""


def _build_device_section(aggregate_df: pd.DataFrame) -> str:
    """Build device type distribution comparison."""
    device_cols = [
        ('total_boiler', 'Boiler'),
        ('total_regular_ac', 'Regular AC'),
        ('total_central_ac', 'Central AC'),
        ('total_three_phase_device', '3-Phase'),
        ('total_recurring_pattern', 'Recurring'),
        ('total_unknown', 'Unknown'),
    ]

    header = '<th>Experiment</th>'
    for _, label in device_cols:
        header += f'<th>{label}</th>'
    header += '<th>Total Sessions</th>'

    rows_html = ''
    for _, row in aggregate_df.iterrows():
        cells = f'<td style="font-weight:600;">{row["exp_id"]}</td>'
        total = 0
        for col, _ in device_cols:
            val = row.get(col, 0)
            val = int(val) if not (isinstance(val, float) and np.isnan(val)) else 0
            total += val
            cells += f'<td>{val}</td>'
        cells += f'<td style="font-weight:600;">{total}</td>'
        rows_html += f'<tr>{cells}</tr>\n'

    return f"""
<section>
    <h2>Device Type Distribution</h2>
    <table class="data-table">
        <thead><tr>{header}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
</section>"""
