"""
HTML report generator for dynamic threshold experiments.

Generates stand-alone HTML reports per house and aggregate reports.
Follows the same patterns as html_report.py (inline CSS, Plotly CDN).

Color scheme: Green (explained), Gray (background), Orange (improvable).
No red - avoids false impression of failure.
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from metrics.dynamic_report_metrics import calculate_dynamic_report_metrics
from visualization.dynamic_report_charts import (
    create_summary_boxes,
    create_power_breakdown_bar,
    create_efficiency_gauge,
    create_threshold_waterfall,
    create_remaining_analysis_chart,
    create_device_summary_table,
    create_device_activations_detail,
)

logger = logging.getLogger(__name__)


def _assign_tier(pre_quality) -> str:
    """Assign quality tier based on pre-analysis quality score."""
    if pre_quality == 'faulty':
        return 'faulty'
    elif pre_quality is None:
        return 'unknown'
    elif pre_quality >= 90:
        return 'excellent'
    elif pre_quality >= 75:
        return 'good'
    elif pre_quality >= 50:
        return 'fair'
    else:
        return 'poor'


def _format_pre_quality(pre_quality) -> str:
    """Format pre-quality score as colored HTML."""
    if pre_quality == 'faulty':
        return '<span style="color:#dc3545;font-weight:bold;" title="Phase with >=20% NaN values">Faulty</span>'
    elif pre_quality is None:
        return '<span style="color:#999;">-</span>'
    else:
        if pre_quality >= 90:
            color = '#28a745'
        elif pre_quality >= 75:
            color = '#007bff'
        elif pre_quality >= 50:
            color = '#ffc107'
        else:
            color = '#dc3545'
        return f'<span style="color:{color};font-weight:bold;">{pre_quality:.0f}</span>'


def generate_dynamic_house_report(
    experiment_dir: str,
    house_id: str,
    output_path: Optional[str] = None,
    pre_quality=None,
) -> str:
    """
    Generate dynamic threshold HTML report for a single house.

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID
        output_path: Where to save the HTML file (optional, auto-generated if None)
        pre_quality: Pre-analysis quality score (float, 'faulty', or None)

    Returns:
        Path to generated HTML file
    """
    experiment_dir = Path(experiment_dir)

    # Calculate metrics
    metrics = calculate_dynamic_report_metrics(experiment_dir, house_id)

    # Generate chart sections
    summary_html = create_summary_boxes(metrics)
    breakdown_html = create_power_breakdown_bar(metrics)
    efficiency_html = create_efficiency_gauge(metrics)
    waterfall_html = create_threshold_waterfall(metrics)
    remaining_html = create_remaining_analysis_chart(metrics)
    devices_html = create_device_summary_table(metrics)

    # Load device activations for detailed table
    activations = _load_device_activations(experiment_dir, house_id)
    activations_detail_html = create_device_activations_detail(activations)

    # Build HTML document
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    period = metrics.get('data_period', {})
    threshold_schedule = metrics.get('threshold_schedule', [])

    html = _build_house_html(
        house_id=house_id,
        generated_at=generated_at,
        period=period,
        threshold_schedule=threshold_schedule,
        summary_html=summary_html,
        breakdown_html=breakdown_html,
        efficiency_html=efficiency_html,
        waterfall_html=waterfall_html,
        remaining_html=remaining_html,
        devices_html=devices_html,
        activations_detail_html=activations_detail_html,
        metrics=metrics,
        pre_quality=pre_quality,
    )

    # Save
    if output_path is None:
        output_path = str(experiment_dir / f"dynamic_report_{house_id}.html")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Report saved to {output_path}")
    return output_path


def _load_device_activations(experiment_dir: Path, house_id: str) -> list:
    """Load device activations list from JSON."""
    json_path = experiment_dir / "device_activations" / f"device_activations_{house_id}.json"
    if not json_path.exists():
        json_path = experiment_dir / f"device_activations_{house_id}.json"

    if not json_path.exists():
        return []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('activations', [])
    except Exception as e:
        logger.warning(f"Failed to load device activations JSON: {e}")
        return []


def generate_dynamic_aggregate_report(
    experiment_dir: str,
    house_ids: List[str],
    output_path: Optional[str] = None,
    pre_analysis_scores: Optional[Dict[str, Any]] = None,
    house_reports_subdir: Optional[str] = None,
) -> str:
    """
    Generate aggregate report across multiple houses.

    Args:
        experiment_dir: Root experiment output directory
        house_ids: List of house IDs
        output_path: Where to save (optional)
        pre_analysis_scores: Dict mapping house_id -> quality score
        house_reports_subdir: Subdirectory name for per-house report links

    Returns:
        Path to generated HTML file
    """
    experiment_dir = Path(experiment_dir)

    all_metrics = []
    for house_id in house_ids:
        metrics = calculate_dynamic_report_metrics(experiment_dir, house_id)
        if pre_analysis_scores:
            metrics['pre_quality'] = pre_analysis_scores.get(house_id)
        all_metrics.append(metrics)

    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = _build_aggregate_html(
        all_metrics=all_metrics,
        generated_at=generated_at,
        experiment_dir=str(experiment_dir),
        house_reports_subdir=house_reports_subdir,
    )

    if output_path is None:
        output_path = str(experiment_dir / "dynamic_report_aggregate.html")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Aggregate report saved to {output_path}")
    return output_path


def _build_house_html(
    house_id: str,
    generated_at: str,
    period: Dict,
    threshold_schedule: List[int],
    summary_html: str,
    breakdown_html: str,
    efficiency_html: str,
    waterfall_html: str,
    remaining_html: str,
    devices_html: str,
    activations_detail_html: str,
    metrics: Dict[str, Any],
    pre_quality=None,
) -> str:
    """Build complete HTML document for a single house."""
    th_str = ' -> '.join(f'{t}W' for t in threshold_schedule)
    period_str = f"{period.get('start', 'N/A')} to {period.get('end', 'N/A')} ({period.get('days', 0)} days)"

    # Pre-quality display
    pre_quality_html = ''
    if pre_quality is not None:
        pq_display = _format_pre_quality(pre_quality)
        pre_quality_html = f'<div class="info-item"><strong>Pre-Quality:</strong> {pq_display}</div>'

    # Phase detail table
    phase_detail = _build_phase_detail_table(metrics.get('phases', {}))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic Threshold Report - House {house_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}

        header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}

        header .subtitle {{
            opacity: 0.9;
        }}

        .info-bar {{
            display: flex;
            gap: 30px;
            margin-top: 15px;
            flex-wrap: wrap;
        }}

        .info-item {{
            font-size: 0.9em;
            opacity: 0.85;
        }}

        .info-item strong {{
            opacity: 1;
        }}

        section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        section h2 {{
            color: #444;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            header {{ padding: 20px; }}
            section {{ padding: 15px; }}
            .charts-grid {{ grid-template-columns: 1fr; }}
            .info-bar {{ flex-direction: column; gap: 5px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Dynamic Threshold Analysis - House {house_id}</h1>
            <div class="subtitle">Generated: {generated_at}</div>
            <div class="info-bar">
                <div class="info-item"><strong>Period:</strong> {period_str}</div>
                <div class="info-item"><strong>Thresholds:</strong> {th_str}</div>
                {pre_quality_html}
            </div>
        </header>

        <section>
            <h2>Power Decomposition Summary</h2>
            {summary_html}
        </section>

        <section>
            <h2>Power Breakdown by Phase</h2>
            <p style="color: #555; margin-bottom: 15px; font-size: 0.9em; line-height: 1.6;">
                Israeli households use a 3-phase electrical system (w1, w2, w3).
                Each phase carries part of the total power. The chart below shows how each phase's
                power is distributed between Explained, Background, and Improvable.
            </p>
            {breakdown_html}
            {phase_detail}
        </section>

        <section>
            <h2>Detection Efficiency by Phase</h2>
            <p style="color: #555; margin-bottom: 15px; font-size: 0.9em; line-height: 1.6;">
                Per-phase efficiency gauges: what fraction of targetable power (total minus background)
                was successfully matched to device events. A low-efficiency phase may have undetected devices
                or noisy data.
            </p>
            {efficiency_html}
        </section>

        <section>
            <h2>Threshold Contribution</h2>
            <p style="color: #555; margin-bottom: 15px; font-size: 0.9em; line-height: 1.6;">
                The algorithm runs multiple iterations with decreasing thresholds ({th_str}).
                Each iteration detects smaller devices in the remaining (unexplained) power.
                The waterfall chart shows how much each threshold level contributed to the total explained power.
            </p>
            {waterfall_html}
        </section>

        <div class="charts-grid">
            <section>
                <h2>Remaining Power Analysis</h2>
                <p style="color: #555; margin-bottom: 10px; font-size: 0.85em; line-height: 1.5;">
                    Classification of power that was <em>not</em> explained, by minute-level magnitude above the background baseline:
                    <strong>Noise</strong> (&lt;200W, likely measurement noise or very small loads),
                    <strong>Small Events</strong> (200-800W, potential undetected devices),
                    <strong>Large Unmatched</strong> (&gt;800W, significant unmatched consumption).
                </p>
                {remaining_html}
            </section>

            <section>
                <h2>Device Summary</h2>
                <p style="color: #555; margin-bottom: 10px; font-size: 0.85em; line-height: 1.5;">
                    Detected device types with their average characteristics.
                    Classification is based on power magnitude, duration, and phase patterns.
                </p>
                {devices_html}
            </section>
        </div>

        <section>
            <h2>Device Activations Detail</h2>
            <p style="color: #555; margin-bottom: 15px; font-size: 0.9em; line-height: 1.6;">
                Every individual device activation detected by the algorithm, grouped by device type.
                Each row is one ON&rarr;OFF event with its date, time range, duration, and power.
                Use "Copy Dates" to export timestamps for external visualization tools.
            </p>
            {activations_detail_html}
        </section>

        <footer>
            ElectricPatterns - Dynamic Threshold Experiment Report
        </footer>
    </div>
</body>
</html>"""


def _build_phase_detail_table(phases: Dict[str, Dict]) -> str:
    """Build a detailed comparison table for all 3 phases."""
    rows = ''
    for phase in ['w1', 'w2', 'w3']:
        ph = phases.get(phase, {})
        rows += f'''
        <tr>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; font-weight: 600;">{phase}</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right;">{ph.get('total_kwh', 0)} kWh</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #28a745;">{ph.get('explained_kwh', 0)} kWh ({ph.get('explained_pct', 0):.1f}%)</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #6c757d;">{ph.get('background_kwh', 0)} kWh ({ph.get('background_pct', 0):.1f}%)</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #fd7e14;">{ph.get('improvable_kwh', 0)} kWh ({ph.get('improvable_pct', 0):.1f}%)</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right;">{ph.get('efficiency', 0):.1f}%</td>
        </tr>
        '''

    return f'''
    <table style="width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.9em;">
        <thead>
            <tr style="background: #f8f9fa;">
                <th style="padding: 10px 15px; text-align: left;">Phase</th>
                <th style="padding: 10px 15px; text-align: right;" title="Total original power consumption">Total</th>
                <th style="padding: 10px 15px; text-align: right; color: #28a745;" title="Power attributed to detected device activations">Explained</th>
                <th style="padding: 10px 15px; text-align: right; color: #6c757d;" title="Baseline power (5th percentile) - always-on devices like fridge, standby">Background</th>
                <th style="padding: 10px 15px; text-align: right; color: #fd7e14;" title="Remaining power above background - potential undetected devices">Improvable</th>
                <th style="padding: 10px 15px; text-align: right;" title="Explained / (Total - Background) - how well we detect non-background devices">Efficiency</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    '''


def _build_aggregate_html(
    all_metrics: List[Dict[str, Any]],
    generated_at: str,
    experiment_dir: str,
    house_reports_subdir: Optional[str] = None,
) -> str:
    """Build aggregate report for multiple houses."""
    # Compute aggregate stats
    valid = [m for m in all_metrics if m.get('totals', {}).get('total_power', 0) > 0]
    n_houses = len(valid)

    if n_houses == 0:
        return _build_empty_aggregate_html(generated_at, experiment_dir, len(all_metrics))

    efficiencies = [m['totals']['efficiency'] for m in valid]
    explained_pcts = [m['totals']['explained_pct'] for m in valid]
    background_pcts = [m['totals']['background_pct'] for m in valid]
    improvable_pcts = [m['totals']['improvable_pct'] for m in valid]

    import numpy as np
    avg_eff = np.mean(efficiencies)
    avg_explained = np.mean(explained_pcts)
    avg_background = np.mean(background_pcts)
    avg_improvable = np.mean(improvable_pcts)

    # Link prefix for per-house reports
    link_prefix = f"{house_reports_subdir}/" if house_reports_subdir else ""

    # House rows with tier data
    house_rows = ''
    for m in sorted(valid, key=lambda x: x['totals']['efficiency'], reverse=True):
        hid = m['house_id']
        t = m['totals']
        pre_quality = m.get('pre_quality')
        tier = _assign_tier(pre_quality)

        eff = t.get('efficiency', 0)
        if eff >= 70:
            eff_color = '#28a745'
        elif eff >= 50:
            eff_color = '#ffc107'
        else:
            eff_color = '#fd7e14'

        pq_html = _format_pre_quality(pre_quality)

        house_rows += f'''
        <tr data-tier="{tier}">
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee;">
                <a href="{link_prefix}dynamic_report_{hid}.html" style="color: #667eea; text-decoration: none;">{hid}</a>
            </td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: center;">{pq_html}</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right;" data-value="{t.get('total_kwh', 0)}">{t.get('total_kwh', 0)} kWh</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #28a745;" data-value="{t.get('explained_pct', 0):.1f}">{t.get('explained_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #6c757d;" data-value="{t.get('background_pct', 0):.1f}">{t.get('background_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #fd7e14;" data-value="{t.get('improvable_pct', 0):.1f}">{t.get('improvable_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold; color: {eff_color};" data-value="{eff:.1f}">{eff:.1f}%</td>
        </tr>
        '''

    # Count tiers
    tier_counts = {}
    for m in valid:
        tier = _assign_tier(m.get('pre_quality'))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    # Distribution chart for efficiency
    chart_id = 'agg-efficiency-dist'
    bins = {'<30%': 0, '30-50%': 0, '50-70%': 0, '70-85%': 0, '>85%': 0}
    for eff in efficiencies:
        if eff < 30:
            bins['<30%'] += 1
        elif eff < 50:
            bins['30-50%'] += 1
        elif eff < 70:
            bins['50-70%'] += 1
        elif eff < 85:
            bins['70-85%'] += 1
        else:
            bins['>85%'] += 1

    dist_data = json.dumps({
        'x': list(bins.keys()),
        'y': list(bins.values()),
        'type': 'bar',
        'marker': {'color': ['#fd7e14', '#fd7e14', '#ffc107', '#28a745', '#28a745']},
    })
    dist_layout = json.dumps({
        'title': f'Detection Efficiency Distribution<br><sub>Avg: {avg_eff:.1f}% across {n_houses} houses</sub>',
        'xaxis': {'title': 'Efficiency Range'},
        'yaxis': {'title': 'Number of Houses'},
        'showlegend': False,
        'height': 350,
    })

    # Build filter bar
    filter_bar = _build_filter_bar(tier_counts)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic Threshold - Aggregate Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        header .subtitle {{ opacity: 0.9; }}
        section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        section h2 {{
            color: #444;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }}
        .summary-number {{ font-size: 2.2em; font-weight: bold; color: #2d3748; }}
        .summary-label {{ color: #666; margin-top: 5px; font-size: 0.9em; }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}
        .data-table th {{
            background: #2d3748;
            color: white;
            padding: 12px 15px;
            text-align: left;
            cursor: pointer;
            user-select: none;
            position: relative;
        }}
        .data-table th:hover {{ background: #4a5568; }}
        .data-table th .sort-arrow {{ margin-left: 5px; opacity: 0.5; }}
        .data-table th.sorted-asc .sort-arrow {{ opacity: 1; }}
        .data-table th.sorted-desc .sort-arrow {{ opacity: 1; }}
        .data-table td {{
            padding: 10px 15px;
            border-bottom: 1px solid #eee;
        }}
        .data-table tr:hover {{ background: #f8f9fa; }}
        .data-table tr.hidden {{ display: none; }}
        .filter-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            align-items: center;
            margin-bottom: 15px;
            padding: 12px 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .filter-bar label {{ font-weight: 600; color: #555; margin-right: 5px; }}
        .filter-checkbox {{ display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 4px; font-size: 0.85em; }}
        .filter-checkbox input {{ cursor: pointer; }}
        .tier-excellent {{ background: #d4edda; color: #155724; }}
        .tier-good {{ background: #cce5ff; color: #004085; }}
        .tier-fair {{ background: #fff3cd; color: #856404; }}
        .tier-poor {{ background: #fde2d4; color: #813e1a; }}
        .tier-faulty {{ background: #f8d7da; color: #721c24; }}
        .tier-unknown {{ background: #e9ecef; color: #495057; }}
        .filter-btn {{ padding: 4px 12px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer; font-size: 0.85em; }}
        .filter-btn:hover {{ background: #e9ecef; }}
        .filter-status {{ font-size: 0.85em; color: #888; margin-left: auto; }}
        .tooltip {{ position: relative; cursor: help; border-bottom: 1px dotted #999; }}
        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            header {{ padding: 20px; }}
            section {{ padding: 15px; }}
            .data-table {{ font-size: 0.8em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Dynamic Threshold - Aggregate Report</h1>
            <div class="subtitle">Generated: {generated_at} | {n_houses} houses analyzed</div>
        </header>

        <section>
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-card" title="Total number of houses with valid data">
                    <div class="summary-number">{n_houses}</div>
                    <div class="summary-label">Houses</div>
                </div>
                <div class="summary-card" title="Average % of total power attributed to detected device activations">
                    <div class="summary-number" style="color: #28a745;">{avg_explained:.1f}%</div>
                    <div class="summary-label">Avg Explained</div>
                </div>
                <div class="summary-card" title="Average baseline power (5th percentile) - always-on devices like fridge, standby, etc.">
                    <div class="summary-number" style="color: #6c757d;">{avg_background:.1f}%</div>
                    <div class="summary-label">Avg Background</div>
                </div>
                <div class="summary-card" title="Average remaining power above background that could potentially be detected with algorithm improvements">
                    <div class="summary-number" style="color: #fd7e14;">{avg_improvable:.1f}%</div>
                    <div class="summary-label">Avg Improvable</div>
                </div>
                <div class="summary-card" title="Explained / (Total - Background) - measures how well the algorithm detects non-background device activations">
                    <div class="summary-number" style="color: #667eea;">{avg_eff:.1f}%</div>
                    <div class="summary-label">Avg Efficiency</div>
                </div>
            </div>
        </section>

        <section>
            <h2>Efficiency Distribution</h2>
            <div id="{chart_id}" style="width:100%;height:350px;"></div>
            <script>
                Plotly.newPlot('{chart_id}', [{dist_data}], {dist_layout});
            </script>
        </section>

        <section>
            <h2>Per-House Results</h2>
            {filter_bar}
            <table class="data-table" id="houses-table">
                <thead>
                    <tr>
                        <th onclick="sortTable(0, 'str')" title="House identifier">House ID <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(1, 'quality')" style="text-align: center;" title="Pre-analysis data quality score (0-100). Based on sharp entry rate, device signatures, power profile, variability, data volume, and data integrity.">Pre-Quality <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(2, 'num')" style="text-align: right;" title="Total original power consumption across all phases">Total <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(3, 'num')" style="text-align: right;" title="% of total power attributed to detected ON/OFF device activations">Explained <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(4, 'num')" style="text-align: right;" title="% of total power that is baseline (5th percentile) - always-on loads">Background <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(5, 'num')" style="text-align: right;" title="% of power above background not yet explained - potential for improvement">Improvable <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(6, 'num')" style="text-align: right;" title="Explained / (Total - Background) - detection rate excluding always-on baseline">Efficiency <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                    </tr>
                </thead>
                <tbody>
                    {house_rows}
                </tbody>
            </table>
        </section>

        <footer>
            ElectricPatterns - Dynamic Threshold Aggregate Report
        </footer>
    </div>

    <script>
    // ── Column Sorting ──────────────────────────────────────────
    var currentSort = {{ col: -1, asc: true }};

    function sortTable(colIdx, type) {{
        var table = document.getElementById('houses-table');
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var headers = table.querySelectorAll('th');

        // Toggle direction
        if (currentSort.col === colIdx) {{
            currentSort.asc = !currentSort.asc;
        }} else {{
            currentSort.col = colIdx;
            currentSort.asc = true;
        }}

        // Clear sort indicators
        headers.forEach(function(h) {{ h.classList.remove('sorted-asc', 'sorted-desc'); }});
        headers[colIdx].classList.add(currentSort.asc ? 'sorted-asc' : 'sorted-desc');

        rows.sort(function(a, b) {{
            var cellA = a.cells[colIdx];
            var cellB = b.cells[colIdx];
            var valA, valB;

            if (type === 'num') {{
                valA = parseFloat(cellA.getAttribute('data-value') || cellA.textContent.replace(/[^0-9.-]/g, '')) || 0;
                valB = parseFloat(cellB.getAttribute('data-value') || cellB.textContent.replace(/[^0-9.-]/g, '')) || 0;
            }} else if (type === 'quality') {{
                var txtA = cellA.textContent.trim();
                var txtB = cellB.textContent.trim();
                valA = txtA === 'Faulty' ? -1 : txtA === '-' ? -2 : parseFloat(txtA) || 0;
                valB = txtB === 'Faulty' ? -1 : txtB === '-' ? -2 : parseFloat(txtB) || 0;
            }} else {{
                valA = cellA.textContent.trim().toLowerCase();
                valB = cellB.textContent.trim().toLowerCase();
                if (valA < valB) return currentSort.asc ? -1 : 1;
                if (valA > valB) return currentSort.asc ? 1 : -1;
                return 0;
            }}

            return currentSort.asc ? (valA - valB) : (valB - valA);
        }});

        rows.forEach(function(row) {{ tbody.appendChild(row); }});
    }}

    // ── Tier Filtering ──────────────────────────────────────────
    function getCheckedTiers() {{
        var checkboxes = document.querySelectorAll('.filter-checkbox input[type=checkbox]');
        var tiers = [];
        checkboxes.forEach(function(cb) {{ if (cb.checked) tiers.push(cb.value); }});
        return tiers;
    }}

    function updateFilter() {{
        var selected = getCheckedTiers();
        var table = document.getElementById('houses-table');
        var rows = table.querySelectorAll('tbody tr');
        var visible = 0;

        rows.forEach(function(row) {{
            var tier = row.getAttribute('data-tier');
            if (selected.indexOf(tier) !== -1) {{
                row.classList.remove('hidden');
                visible++;
            }} else {{
                row.classList.add('hidden');
            }}
        }});

        var status = document.getElementById('filter-status');
        if (status) status.textContent = 'Showing ' + visible + ' / ' + rows.length + ' houses';
    }}

    function allExceptFaulty() {{
        var checkboxes = document.querySelectorAll('.filter-checkbox input[type=checkbox]');
        checkboxes.forEach(function(cb) {{ cb.checked = (cb.value !== 'faulty'); }});
        updateFilter();
    }}

    function selectAll() {{
        var checkboxes = document.querySelectorAll('.filter-checkbox input[type=checkbox]');
        checkboxes.forEach(function(cb) {{ cb.checked = true; }});
        updateFilter();
    }}
    </script>
</body>
</html>"""


def _build_filter_bar(tier_counts: Dict[str, int]) -> str:
    """Build the tier filter bar HTML."""
    tiers = [
        ('excellent', 'Excellent', 'tier-excellent'),
        ('good', 'Good', 'tier-good'),
        ('fair', 'Fair', 'tier-fair'),
        ('poor', 'Poor', 'tier-poor'),
        ('faulty', 'Faulty', 'tier-faulty'),
        ('unknown', 'Unknown', 'tier-unknown'),
    ]

    checkboxes = ''
    for value, label, css_class in tiers:
        count = tier_counts.get(value, 0)
        if count == 0:
            continue
        checkboxes += f'''
            <span class="filter-checkbox {css_class}">
                <input type="checkbox" value="{value}" checked onchange="updateFilter()">
                {label} ({count})
            </span>'''

    if not checkboxes:
        return ''

    return f'''
    <div class="filter-bar">
        <label>Filter by Pre-Quality:</label>
        {checkboxes}
        <button class="filter-btn" onclick="allExceptFaulty()" style="font-weight:bold;">All except Faulty</button>
        <button class="filter-btn" onclick="selectAll()">Show All</button>
        <span class="filter-status" id="filter-status"></span>
    </div>'''


def _build_empty_aggregate_html(generated_at: str, experiment_dir: str, total: int) -> str:
    """Build aggregate report when no valid data is found."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dynamic Threshold - Aggregate Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>No Valid Data</h1>
        <p style="margin-top: 15px; color: #888;">
            Attempted to analyze {total} houses but no valid summarized data was found.
        </p>
        <p style="margin-top: 10px; color: #888; font-size: 0.9em;">
            Directory: {experiment_dir}<br>
            Generated: {generated_at}
        </p>
    </div>
</body>
</html>"""
