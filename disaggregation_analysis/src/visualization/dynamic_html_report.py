"""
HTML report generator for dynamic threshold experiments (Module 1 — Disaggregation).

Generates stand-alone HTML reports per house and aggregate reports.
Follows the same patterns as html_report.py (inline CSS, Plotly CDN).

Color scheme: Green (explained), Gray (background), Orange (unmatched).
No red - avoids false impression of failure.

Note: Device identification (Module 2) content has been moved to
identification_analysis/src/visualization/identification_html_report.py
"""
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from metrics.dynamic_report_metrics import calculate_dynamic_report_metrics

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False
from visualization.dynamic_report_charts import (
    create_summary_boxes,
    create_power_breakdown_bar,
    create_efficiency_gauge,
    create_threshold_waterfall,
    create_remaining_analysis_chart,
)

logger = logging.getLogger(__name__)


def _format_class_quality(value) -> str:
    """Format classification quality score as colored HTML."""
    if value is None:
        return '<span style="color: #aaa;">-</span>'
    if value >= 0.7:
        color = '#28a745'
    elif value >= 0.4:
        color = '#eab308'
    else:
        color = '#e67e22'
    return f'<span style="color: {color}; font-weight: 600;">{value:.2f}</span>'


def _format_avg_confidence(value) -> str:
    """Format average confidence score as colored HTML."""
    if value is None:
        return '<span style="color: #aaa;">-</span>'
    if value >= 0.7:
        color = '#28a745'
    elif value >= 0.4:
        color = '#eab308'
    else:
        color = '#e67e22'
    return f'<span style="color: {color}; font-weight: 600;">{value:.2f}</span>'


def _build_population_stats_section(population_stats: Dict[str, Any]) -> str:
    """Build HTML section for population-level classification quality statistics."""
    if not population_stats or population_stats.get('houses_analyzed', 0) == 0:
        return ''

    n = population_stats['houses_analyzed']
    q_dist = population_stats.get('quality_distribution', {})
    c_dist = population_stats.get('confidence_distribution', {})
    outliers = population_stats.get('outlier_houses', [])
    rates = population_stats.get('classification_rates', {})
    per_device = population_stats.get('per_device_type', {})

    # Quality score stats
    q_median = q_dist.get('median', 0)
    q_min = q_dist.get('min', 0)
    q_max = q_dist.get('max', 0)

    # Confidence stats
    c_median = c_dist.get('median', 0)
    c_min = c_dist.get('min', 0)
    c_max = c_dist.get('max', 0)

    # High-confidence rates
    high_conf = rates.get('high_conf_rates', {})
    weighted = rates.get('weighted_rates', {})

    # Device type cards
    device_cards = ''
    device_labels = {'boiler': 'Boiler', 'central_ac': 'Central AC', 'regular_ac': 'Regular AC'}
    for dtype in ['boiler', 'central_ac', 'regular_ac']:
        if dtype not in per_device:
            continue
        d = per_device[dtype]
        count_dist = d.get('count', {})
        cv_dist = d.get('cv', {})
        if count_dist.get('n', 0) == 0:
            continue
        device_cards += f'''
            <div style="background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center;">
                <div style="font-weight: 600; margin-bottom: 8px;">{device_labels.get(dtype, dtype)}</div>
                <div style="font-size: 0.85em; color: #555;">
                    Detected in <b>{count_dist['n']}</b> houses<br>
                    Median count: <b>{count_dist.get('median', 0):.0f}</b><br>
                    Magnitude CV: <b>{cv_dist.get('median', 0):.2f}</b>
                </div>
            </div>'''

    # Outlier list
    outlier_html = ''
    if outliers:
        items = ', '.join(f"<b>{o['house_id']}</b>" for o in outliers[:10])
        if len(outliers) > 10:
            items += f', +{len(outliers) - 10} more'
        outlier_html = f'''
            <div style="margin-top: 15px; padding: 12px 15px; background: #fff3cd; border-radius: 8px; font-size: 0.9em;">
                <b>Outlier houses ({len(outliers)}):</b> {items}
                <div style="color: #856404; font-size: 0.85em; margin-top: 4px;">
                    Houses with unusual classification patterns (|z-score| &gt; 3.0 or 3+ warnings)
                </div>
            </div>'''

    return f'''
        <section>
            <h2>Classification Quality Overview</h2>
            <div class="summary-grid" style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));">
                <div class="summary-card">
                    <div class="summary-number" style="color: #667eea;">{n}</div>
                    <div class="summary-label">Houses Analyzed</div>
                </div>
                <div class="summary-card">
                    <div class="summary-number" style="color: #28a745;">{q_median:.2f}</div>
                    <div class="summary-label">Median Quality Score</div>
                    <div style="font-size: 0.78em; color: #888; margin-top: 4px;">Range: {q_min:.2f} — {q_max:.2f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-number" style="color: #17a2b8;">{c_median:.2f}</div>
                    <div class="summary-label">Median Confidence</div>
                    <div style="font-size: 0.78em; color: #888; margin-top: 4px;">Range: {c_min:.2f} — {c_max:.2f}</div>
                </div>
                <div class="summary-card">
                    <div class="summary-number" style="color: {'#e67e22' if len(outliers) > 0 else '#28a745'};">{len(outliers)}</div>
                    <div class="summary-label">Outlier Houses</div>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <h3 style="font-size: 1em; color: #555; margin-bottom: 12px;">Device Type Summary</h3>
                <div class="summary-grid" style="grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));">
                    {device_cards}
                </div>
            </div>
            {outlier_html}
        </section>'''


def _assign_tier(pre_quality) -> str:
    """Assign quality tier based on pre-analysis quality score."""
    if isinstance(pre_quality, str) and pre_quality.startswith('faulty'):
        return pre_quality  # 'faulty_dead_phase', 'faulty_high_nan', or 'faulty_both'
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
    if isinstance(pre_quality, str) and pre_quality.startswith('faulty'):
        _faulty_labels = {
            'faulty_dead_phase': ('Dead Phase', 'Phase with <2% of sisters avg'),
            'faulty_high_nan': ('High NaN', 'Phase with >=10% NaN values'),
            'faulty_both': ('Both', 'Dead phase + high NaN on other phases'),
        }
        _fl, _ft = _faulty_labels.get(pre_quality, ('Faulty', ''))
        return f'<span style="color:#6f42c1;font-weight:bold;" title="{_ft}">{_fl}</span>'
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
    skip_activations_detail: bool = False,
) -> str:
    """
    Generate dynamic threshold HTML report for a single house (M1 disaggregation only).

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID
        output_path: Where to save the HTML file (optional, auto-generated if None)
        pre_quality: Pre-analysis quality score (float, 'faulty', or None)
        skip_activations_detail: Unused (kept for backward compatibility)

    Returns:
        Path to generated HTML file
    """
    experiment_dir = Path(experiment_dir)

    # Calculate M1 disaggregation metrics
    metrics = calculate_dynamic_report_metrics(experiment_dir, house_id)

    # Generate M1 chart sections
    summary_html = create_summary_boxes(metrics)
    breakdown_html = create_power_breakdown_bar(metrics)
    efficiency_html = create_efficiency_gauge(metrics)
    waterfall_html = create_threshold_waterfall(metrics)
    remaining_html = create_remaining_analysis_chart(metrics)

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
    show_progress: bool = False,
) -> str:
    """
    Generate aggregate report across multiple houses.

    Args:
        experiment_dir: Root experiment output directory
        house_ids: List of house IDs
        output_path: Where to save (optional)
        pre_analysis_scores: Dict mapping house_id -> quality score
        house_reports_subdir: Subdirectory name for per-house report links
        show_progress: Show tqdm progress bar for metrics calculation

    Returns:
        Path to generated HTML file
    """
    experiment_dir = Path(experiment_dir)

    all_metrics = []
    houses_iter = house_ids
    use_tqdm = show_progress and _HAS_TQDM
    if use_tqdm:
        houses_iter = _tqdm(house_ids, desc="Aggregate metrics", unit="house")

    # Track cumulative timing per step (for summary)
    cumulative_timing = {}

    for house_id in houses_iter:
        metrics = calculate_dynamic_report_metrics(experiment_dir, house_id)

        # Update tqdm postfix with per-house timing breakdown
        timing = metrics.pop('_timing', None)
        if timing:
            for step, secs in timing.items():
                cumulative_timing[step] = cumulative_timing.get(step, 0) + secs
            if use_tqdm:
                slowest = max(timing, key=timing.get)
                houses_iter.set_postfix(
                    house=house_id,
                    slowest=f"{slowest}={timing[slowest]:.1f}s",
                )

        if pre_analysis_scores:
            house_pre = pre_analysis_scores.get(house_id, {})
            if isinstance(house_pre, dict):
                metrics['pre_quality'] = house_pre.get('quality_score')
                metrics['nan_continuity'] = house_pre.get('nan_continuity', 'unknown')
                metrics['max_nan_pct'] = house_pre.get('max_nan_pct', 0)
            else:
                metrics['pre_quality'] = house_pre
                metrics['nan_continuity'] = 'unknown'
                metrics['max_nan_pct'] = 0

        all_metrics.append(metrics)

    # Print timing summary
    if show_progress and cumulative_timing:
        total_t = sum(cumulative_timing.values())
        print(f"  Metrics timing breakdown ({total_t:.1f}s total):", flush=True)
        for step, secs in sorted(cumulative_timing.items(), key=lambda x: -x[1]):
            pct = secs / total_t * 100 if total_t > 0 else 0
            print(f"    {step:20s} {secs:7.1f}s  ({pct:.0f}%)", flush=True)

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
    metrics: Dict[str, Any] = None,
    pre_quality=None,
) -> str:
    """Build complete HTML document for a single house (M1 disaggregation only)."""
    th_str = ' -> '.join(f'{t}W' for t in threshold_schedule)
    period_str = f"{period.get('start', 'N/A')} to {period.get('end', 'N/A')} ({period.get('days', 0)} days)"

    # Pre-quality display
    pre_quality_html = ''
    if pre_quality is not None:
        pq_display = _format_pre_quality(pre_quality)
        pre_quality_html = f'<div class="info-item"><strong>Pre-Quality:</strong> {pq_display}</div>'

    # Phase detail table
    phase_detail = _build_phase_detail_table(metrics.get('phases', {}))

    # NaN info: show per-phase NaN minutes if any exist
    nan_info_html = ''
    phases_data = metrics.get('phases', {})
    total_nan = sum(phases_data.get(ph, {}).get('nan_minutes', 0) for ph in ['w1', 'w2', 'w3'])
    if total_nan > 0:
        nan_parts = []
        for ph in ['w1', 'w2', 'w3']:
            ph_data = phases_data.get(ph, {})
            nan_min = ph_data.get('nan_minutes', 0)
            all_min = ph_data.get('all_minutes', 0)
            if nan_min > 0 and all_min > 0:
                nan_pct = nan_min / all_min * 100
                nan_parts.append(f'{ph}: {nan_min:,} min ({nan_pct:.0f}%)')
        nan_info_html = (
            f'<div style="background:#e8daf0;border:1px solid #6f42c1;border-radius:6px;'
            f'padding:8px 15px;margin-bottom:12px;font-size:0.82em;color:#4a0e6b;">'
            f'NaN minutes (shown as "No Data"): {", ".join(nan_parts)}'
            f'</div>'
        )

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

        {nan_info_html}

        <section>
            <h2>Power Decomposition Summary</h2>
            {summary_html}
        </section>

        <section>
            <h2>Power Breakdown by Phase</h2>
            <p style="color: #666; margin-bottom: 10px; font-size: 0.85em;">
                Per-phase (w1, w2, w3) decomposition into Explained, Unmatched, Sub-threshold, Background, and No Data.
            </p>
            {breakdown_html}
            {phase_detail}
        </section>

        <section>
            <h2>Detection Efficiency by Phase</h2>
            <p style="color: #666; margin-bottom: 10px; font-size: 0.85em;">
                Fraction of detectable power (Explained + Unmatched) matched to device events per phase.
            </p>
            {efficiency_html}
        </section>

        <section>
            <h2>Threshold Contribution</h2>
            <p style="color: #666; margin-bottom: 10px; font-size: 0.85em;">
                Each iteration uses a lower threshold ({th_str}) to detect progressively smaller devices.
            </p>
            {waterfall_html}
        </section>

        <section>
            <h2>Remaining Power Analysis</h2>
            <p style="color: #666; margin-bottom: 8px; font-size: 0.82em;">
                Unexplained power by magnitude: Noise (&lt;200W), Small Events (200-800W), Large Unmatched (&gt;800W).
            </p>
            {remaining_html}
        </section>

        <footer>
            ElectricPatterns - Module 1: Disaggregation Report
            <div style="font-size: 0.85em; margin-top: 4px; color: #aaa;">
                Device identification analysis available in the separate M2 report.
            </div>
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
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #e67e22;">{ph.get('above_th_kwh', 0)} kWh ({ph.get('above_th_pct', 0):.1f}%)</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #a38600;">{ph.get('sub_threshold_kwh', 0)} kWh ({ph.get('sub_threshold_pct', 0):.1f}%)</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #6c757d;">{ph.get('background_kwh', 0)} kWh ({ph.get('background_pct', 0):.1f}%)</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #6f42c1;">{ph.get('no_data_pct', 0):.1f}%</td>
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
                <th style="padding: 10px 15px; text-align: right; color: #e67e22;" title="Above minimum threshold, not matched to any device">Unmatched</th>
                <th style="padding: 10px 15px; text-align: right; color: #a38600;" title="Between background and minimum threshold — undetectable by design">Sub-threshold</th>
                <th style="padding: 10px 15px; text-align: right; color: #6c757d;" title="Baseline power (5th percentile) - always-on devices">Background</th>
                <th style="padding: 10px 15px; text-align: right; color: #6f42c1;" title="% of time with no power reading (NaN)">No Data</th>
                <th style="padding: 10px 15px; text-align: right;" title="Explained / (Explained + Unmatched) — only detectable power">Efficiency</th>
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
    """Build aggregate report for multiple houses (M1 disaggregation only)."""
    # Compute aggregate stats
    valid = [m for m in all_metrics if m.get('totals', {}).get('total_power', 0) > 0]
    n_houses = len(valid)

    if n_houses == 0:
        return _build_empty_aggregate_html(generated_at, experiment_dir, len(all_metrics))

    efficiencies = [m['totals']['efficiency'] for m in valid]
    explained_pcts = [m['totals']['explained_pct'] for m in valid]
    background_pcts = [m['totals']['background_pct'] for m in valid]
    above_th_pcts = [m['totals'].get('above_th_pct', 0) for m in valid]
    sub_threshold_pcts = [m['totals'].get('sub_threshold_pct', 0) for m in valid]
    no_data_pcts = [m['totals'].get('no_data_pct', 0) for m in valid]

    import numpy as np
    avg_eff = np.mean(efficiencies)
    avg_explained = np.mean(explained_pcts)
    avg_background = np.mean(background_pcts)
    avg_above_th = np.mean(above_th_pcts)
    avg_sub_threshold = np.mean(sub_threshold_pcts)
    avg_no_data = np.mean(no_data_pcts)

    # Link prefix for per-house reports
    link_prefix = f"{house_reports_subdir}/" if house_reports_subdir else ""

    # House rows with tier and continuity data
    house_rows = ''
    for m in sorted(valid, key=lambda x: x['totals']['efficiency'], reverse=True):
        hid = m['house_id']
        t = m['totals']
        pre_quality = m.get('pre_quality')
        tier = _assign_tier(pre_quality)
        nan_cont = m.get('nan_continuity', 'unknown')

        eff = t.get('efficiency', 0)
        if eff >= 70:
            eff_color = '#28a745'
        elif eff >= 50:
            eff_color = '#eab308'
        else:
            eff_color = '#e67e22'

        pq_html = _format_pre_quality(pre_quality)
        days = m.get('data_period', {}).get('days', 0)

        house_rows += f'''
        <tr data-tier="{tier}" data-continuity="{nan_cont}"
            data-explained="{t.get('explained_pct', 0):.1f}" data-background="{t.get('background_pct', 0):.1f}"
            data-aboveth="{t.get('above_th_pct', 0):.1f}" data-subth="{t.get('sub_threshold_pct', 0):.1f}"
            data-nodata="{t.get('no_data_pct', 0):.1f}" data-efficiency="{eff:.1f}">
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee;">
                <a href="{link_prefix}dynamic_report_{hid}.html" style="color: #667eea; text-decoration: none;">{hid}</a>
            </td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: center;" data-value="{days}">{days}</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: center;">{pq_html}</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right;" data-value="{t.get('total_kwh', 0)}">{t.get('total_kwh', 0)} kWh</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #28a745;" data-value="{t.get('explained_pct', 0):.1f}">{t.get('explained_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #e67e22;" data-value="{t.get('above_th_pct', 0):.1f}">{t.get('above_th_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #a38600;" data-value="{t.get('sub_threshold_pct', 0):.1f}">{t.get('sub_threshold_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #6c757d;" data-value="{t.get('background_pct', 0):.1f}">{t.get('background_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; color: #6f42c1;" data-value="{t.get('no_data_pct', 0):.1f}">{t.get('no_data_pct', 0):.1f}%</td>
            <td style="padding: 10px 15px; border-bottom: 1px solid #eee; text-align: right; font-weight: bold; color: {eff_color};" data-value="{eff:.1f}">{eff:.1f}%</td>
        </tr>
        '''

    # Count tiers and continuity labels
    tier_counts = {}
    continuity_counts = {}
    for m in valid:
        tier = _assign_tier(m.get('pre_quality'))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        cont = m.get('nan_continuity', 'unknown')
        continuity_counts[cont] = continuity_counts.get(cont, 0) + 1

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
        'marker': {'color': ['#e67e22', '#e67e22', '#eab308', '#28a745', '#28a745']},
    })
    dist_layout = json.dumps({
        'title': f'Detection Efficiency Distribution<br><sub>Avg: {avg_eff:.1f}% across {n_houses} houses</sub>',
        'xaxis': {'title': 'Efficiency Range'},
        'yaxis': {'title': 'Number of Houses'},
        'showlegend': False,
        'height': 350,
    })

    # Build filter bar
    filter_bar = _build_filter_bar(tier_counts, continuity_counts)

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
        .tier-faulty_dead_phase {{ background: #d4c5e2; color: #5a3d7a; }}
        .tier-faulty_high_nan {{ background: #e2d5f0; color: #6f42c1; }}
        .tier-faulty_both {{ background: #c9a3d4; color: #4a0e6b; }}
        .tier-unknown {{ background: #e9ecef; color: #495057; }}
        .cont-continuous {{ background: #d4edda; color: #155724; }}
        .cont-minor_gaps {{ background: #cce5ff; color: #004085; }}
        .cont-discontinuous {{ background: #fff3cd; color: #856404; }}
        .cont-fragmented {{ background: #f8d7da; color: #721c24; }}
        .cont-unknown {{ background: #e9ecef; color: #495057; }}
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
            <div style="border: 2px solid #dee2e6; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
                <div style="font-size: 0.82em; font-weight: 600; color: #555; margin-bottom: 10px;">
                    Power Decomposition (= 100%) &mdash; {n_houses} houses
                </div>
                <div class="summary-grid">
                    <div class="summary-card" title="Average % of total power attributed to detected device activations">
                        <div class="summary-number" id="value-explained" style="color: #28a745;">{avg_explained:.1f}%</div>
                        <div class="summary-label">Avg Explained</div>
                    </div>
                    <div class="summary-card" title="Average above-threshold power not matched to any device">
                        <div class="summary-number" id="value-aboveth" style="color: #e67e22;">{avg_above_th:.1f}%</div>
                        <div class="summary-label">Avg Unmatched</div>
                    </div>
                    <div class="summary-card" title="Average sub-threshold power — below detection threshold, undetectable by design">
                        <div class="summary-number" id="value-subth" style="color: #a38600;">{avg_sub_threshold:.1f}%</div>
                        <div class="summary-label">Avg Sub-threshold</div>
                    </div>
                    <div class="summary-card" title="Average baseline power (5th percentile) - always-on devices">
                        <div class="summary-number" id="value-background" style="color: #6c757d;">{avg_background:.1f}%</div>
                        <div class="summary-label">Avg Background</div>
                    </div>
                    <div class="summary-card" title="Average % of time with no power reading (NaN). Not included in any calculation.">
                        <div class="summary-number" id="value-nodata" style="color: #6f42c1;">{avg_no_data:.1f}%</div>
                        <div class="summary-label">Avg No Data</div>
                    </div>
                </div>
            </div>
            <div class="summary-grid" style="grid-template-columns: 1fr;">
                <div class="summary-card" title="Explained / (Explained + Unmatched) — only above-threshold power in scope">
                    <div class="summary-number" id="value-efficiency" style="color: #667eea;">{avg_eff:.1f}%</div>
                    <div class="summary-label">Avg Detection Efficiency</div>
                    <div style="font-size: 0.78em; color: #888; margin-top: 4px;">Explained / (Explained + Unmatched) &mdash; background and sub-threshold excluded</div>
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
                        <th onclick="sortTable(1, 'num')" style="text-align: center;" title="Number of days in the data period">Days <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(2, 'quality')" style="text-align: center;" title="Pre-analysis data quality score (0-100)">Pre-Quality <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(3, 'num')" style="text-align: right;" title="Total original power consumption across all phases">Total <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(4, 'num')" style="text-align: right;" title="% of total power attributed to detected ON/OFF device activations">Explained <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(5, 'num')" style="text-align: right;" title="% of above-threshold power not matched to any device">Unmatched <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(6, 'num')" style="text-align: right;" title="% of power between background and min threshold">Sub-threshold <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(7, 'num')" style="text-align: right;" title="% of total power that is baseline (5th percentile)">Background <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(8, 'num')" style="text-align: right;" title="% of time with no power reading (NaN)">No Data <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                        <th onclick="sortTable(9, 'num')" style="text-align: right;" title="Explained / (Explained + Unmatched)">Efficiency <span class="sort-arrow">&#x25B4;&#x25BE;</span></th>
                    </tr>
                </thead>
                <tbody>
                    {house_rows}
                </tbody>
            </table>
        </section>

        <footer>
            ElectricPatterns - Module 1: Disaggregation Aggregate Report
            <div style="font-size: 0.85em; margin-top: 4px; color: #aaa;">
                Device identification analysis available in the separate M2 report.
            </div>
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
                valA = (txtA === 'Dead Phase' || txtA === 'High NaN' || txtA === 'Both') ? -1 : txtA === '-' ? -2 : parseFloat(txtA) || 0;
                valB = (txtB === 'Dead Phase' || txtB === 'High NaN' || txtB === 'Both') ? -1 : txtB === '-' ? -2 : parseFloat(txtB) || 0;
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
    function getCheckedValues(className) {{
        var checkboxes = document.querySelectorAll('.' + className + ' input[type=checkbox]');
        var values = [];
        checkboxes.forEach(function(cb) {{ if (cb.checked) values.push(cb.value); }});
        return values;
    }}

    function getCheckedTiers() {{
        return getCheckedValues('tier-filter');
    }}

    function getCheckedContinuity() {{
        return getCheckedValues('cont-filter');
    }}

    function updateFilter() {{
        var selectedTiers = getCheckedTiers();
        var selectedCont = getCheckedContinuity();
        var table = document.getElementById('houses-table');
        var rows = table.querySelectorAll('tbody tr');
        var visible = 0;
        var sumExpl = 0, sumBg = 0, sumAbove = 0, sumSub = 0, sumNoData = 0, sumEff = 0;

        rows.forEach(function(row) {{
            var tier = row.getAttribute('data-tier');
            var cont = row.getAttribute('data-continuity');
            if (selectedTiers.indexOf(tier) !== -1 && selectedCont.indexOf(cont) !== -1) {{
                row.classList.remove('hidden');
                visible++;
                sumExpl += parseFloat(row.getAttribute('data-explained')) || 0;
                sumBg += parseFloat(row.getAttribute('data-background')) || 0;
                sumAbove += parseFloat(row.getAttribute('data-aboveth')) || 0;
                sumSub += parseFloat(row.getAttribute('data-subth')) || 0;
                sumNoData += parseFloat(row.getAttribute('data-nodata')) || 0;
                sumEff += parseFloat(row.getAttribute('data-efficiency')) || 0;
            }} else {{
                row.classList.add('hidden');
            }}
        }});

        // Update summary cards
        document.getElementById('value-explained').textContent = (visible > 0 ? (sumExpl / visible).toFixed(1) : '0.0') + '%';
        document.getElementById('value-background').textContent = (visible > 0 ? (sumBg / visible).toFixed(1) : '0.0') + '%';
        document.getElementById('value-aboveth').textContent = (visible > 0 ? (sumAbove / visible).toFixed(1) : '0.0') + '%';
        document.getElementById('value-subth').textContent = (visible > 0 ? (sumSub / visible).toFixed(1) : '0.0') + '%';
        document.getElementById('value-nodata').textContent = (visible > 0 ? (sumNoData / visible).toFixed(1) : '0.0') + '%';
        document.getElementById('value-efficiency').textContent = (visible > 0 ? (sumEff / visible).toFixed(1) : '0.0') + '%';

        var status = document.getElementById('filter-status');
        if (status) status.textContent = 'Showing ' + visible + ' / ' + rows.length + ' houses';
    }}

    function allExceptFaulty() {{
        document.querySelectorAll('.tier-filter input[type=checkbox]').forEach(function(cb) {{ cb.checked = !cb.value.startsWith('faulty'); }});
        document.querySelectorAll('.cont-filter input[type=checkbox]').forEach(function(cb) {{ cb.checked = true; }});
        updateFilter();
    }}

    function selectAll() {{
        document.querySelectorAll('.filter-bar input[type=checkbox]').forEach(function(cb) {{ cb.checked = true; }});
        updateFilter();
    }}
    </script>
</body>
</html>"""


def _build_filter_bar(tier_counts: Dict[str, int], continuity_counts: Optional[Dict[str, int]] = None) -> str:
    """Build the tier and continuity filter bars HTML."""
    tiers = [
        ('excellent', 'Excellent', 'tier-excellent'),
        ('good', 'Good', 'tier-good'),
        ('fair', 'Fair', 'tier-fair'),
        ('poor', 'Poor', 'tier-poor'),
        ('faulty_dead_phase', 'Dead Phase', 'tier-faulty_dead_phase'),
        ('faulty_high_nan', 'High NaN', 'tier-faulty_high_nan'),
        ('faulty_both', 'Both', 'tier-faulty_both'),
        ('unknown', 'Unknown', 'tier-unknown'),
    ]

    tier_checkboxes = ''
    for value, label, css_class in tiers:
        count = tier_counts.get(value, 0)
        if count == 0:
            continue
        tier_checkboxes += f'''
            <span class="filter-checkbox tier-filter {css_class}">
                <input type="checkbox" value="{value}" checked onchange="updateFilter()">
                {label} ({count})
            </span>'''

    if not tier_checkboxes:
        return ''

    # NaN continuity filter
    cont_checkboxes = ''
    if continuity_counts:
        cont_items = [
            ('continuous', 'Continuous', 'cont-continuous'),
            ('minor_gaps', 'Minor Gaps', 'cont-minor_gaps'),
            ('discontinuous', 'Discontinuous', 'cont-discontinuous'),
            ('fragmented', 'Fragmented', 'cont-fragmented'),
            ('unknown', 'Unknown', 'cont-unknown'),
        ]
        for value, label, css_class in cont_items:
            count = continuity_counts.get(value, 0)
            if count == 0:
                continue
            cont_checkboxes += f'''
            <span class="filter-checkbox cont-filter {css_class}">
                <input type="checkbox" value="{value}" checked onchange="updateFilter()">
                {label} ({count})
            </span>'''

    cont_bar = ''
    if cont_checkboxes:
        cont_bar = f'''
    <div class="filter-bar">
        <label>Filter by NaN Continuity:</label>
        {cont_checkboxes}
    </div>'''

    return f'''
    <div class="filter-bar">
        <label>Filter by Pre-Quality:</label>
        {tier_checkboxes}
        <button class="filter-btn" onclick="allExceptFaulty()" style="font-weight:bold;">All except Faulty</button>
        <button class="filter-btn" onclick="selectAll()">Show All</button>
        <span class="filter-status" id="filter-status"></span>
    </div>{cont_bar}'''


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
