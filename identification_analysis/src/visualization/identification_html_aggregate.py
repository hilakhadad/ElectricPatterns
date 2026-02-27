"""
Aggregate identification report builders.

Extracted from identification_html_report.py.
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from shared.html_utils import (
    build_quality_dist_bar as _build_quality_dist_bar,
    assign_tier as _assign_tier,
    format_pre_quality as _format_pre_quality,
    build_glossary_section as _build_glossary_section,
    build_about_section as _build_about_section,
)

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# Cross-house patterns: show only top N in aggregate report
CROSS_HOUSE_DISPLAY_LIMIT = 5

from metrics.classification_quality import calculate_classification_quality
from metrics.confidence_scoring import calculate_confidence_scores
from metrics.population_statistics import compute_population_statistics

logger = logging.getLogger(__name__)


def generate_identification_aggregate_report(
    experiment_dir: str,
    house_ids: List[str],
    output_path: Optional[str] = None,
    house_reports_subdir: Optional[str] = None,
    show_progress: bool = False,
    precomputed_metrics: Optional[Dict[str, dict]] = None,
    show_timing: bool = False,
    per_house_filename_pattern: Optional[str] = None,
    pre_analysis_scores: Optional[Dict[str, Any]] = None,
    cross_house_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate aggregate identification report across multiple houses.

    Args:
        experiment_dir: Root experiment output directory
        house_ids: List of house IDs
        output_path: Where to save (optional)
        house_reports_subdir: Subdirectory for per-house report links
        show_progress: Show tqdm progress bar
        precomputed_metrics: Optional dict {house_id: {'quality': ..., 'confidence': ...}}
                             from per-house phase — avoids recalculating
        show_timing: If True, print timing info

    Returns:
        Path to generated HTML file
    """
    import time as _time

    experiment_dir = Path(experiment_dir)
    precomputed_metrics = precomputed_metrics or {}

    all_quality = []
    all_confidence = []
    house_summaries = []

    houses_iter = house_ids
    if show_progress and _HAS_TQDM:
        houses_iter = _tqdm(house_ids, desc="Aggregate M2 metrics", unit="house")

    t0 = _time.time()
    reused = 0

    for house_id in houses_iter:
        sessions_data = _load_sessions(experiment_dir, house_id)
        sessions = sessions_data.get('sessions', [])
        summary = sessions_data.get('summary', {})

        # Spike filter info
        spike_filter = sessions_data.get('spike_filter', {})
        spike_count = spike_filter.get('spike_count', 0)

        # Report link (always generate, even for empty houses)
        report_link = None
        if house_reports_subdir:
            if per_house_filename_pattern:
                per_house_file = per_house_filename_pattern.replace('{house_id}', house_id)
            else:
                per_house_file = f'identification_report_{house_id}.html'
            report_link = f'{house_reports_subdir}/{per_house_file}'

        # Pre-analysis quality for this house
        house_pre = (pre_analysis_scores or {}).get(house_id, {})
        if isinstance(house_pre, dict):
            pre_quality = house_pre.get('quality_score')
        else:
            pre_quality = house_pre

        if not sessions:
            house_summaries.append({
                'house_id': house_id,
                'total_sessions': 0,
                'total_minutes': 0,
                'classified_minutes': 0,
                'classified_pct': 0,
                'avg_confidence': 0,
                'quality_score': None,
                'device_counts': {},
                'report_link': report_link,
                'days_span': 0,
                'classified_min_per_day': 0,
                'spike_count': spike_count,
                'pre_quality': pre_quality,
            })
            continue

        # Collect per-house summary
        total = len(sessions)
        classified_count = sum(1 for s in sessions if s.get('device_type') not in ('unknown', 'unclassified'))

        # Minutes-based classification
        total_minutes = sum(s.get('duration_minutes', 0) or 0 for s in sessions)
        classified_minutes = sum(
            (s.get('duration_minutes', 0) or 0)
            for s in sessions if s.get('device_type') not in ('unknown', 'unclassified')
        )
        classified_pct = (classified_minutes / total_minutes * 100) if total_minutes > 0 else 0

        # Confidence of classified sessions only
        conf_vals = [
            s.get('confidence', 0)
            for s in sessions
            if s.get('confidence') and s.get('device_type') not in ('unknown', 'unclassified')
        ]
        avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0

        # Device counts
        device_counts = {}
        for s in sessions:
            dt = s.get('device_type', 'unknown')
            if not isinstance(dt, str):
                dt = str(dt) if dt is not None else 'unknown'
            device_counts[dt] = device_counts.get(dt, 0) + 1

        # Days span and classified minutes/day from session timestamps
        min_date = max_date = None
        for s in sessions:
            start = s.get('start', '')
            if start:
                try:
                    d = datetime.fromisoformat(str(start)).date()
                    if min_date is None or d < min_date:
                        min_date = d
                    if max_date is None or d > max_date:
                        max_date = d
                except (ValueError, TypeError):
                    pass

        days_span = (max_date - min_date).days + 1 if min_date and max_date else 0
        classified_min_per_day = classified_minutes / days_span if days_span > 0 else 0

        # Quality + confidence metrics — reuse from per-house phase if available
        cached = precomputed_metrics.get(house_id, {})
        quality = cached.get('quality')
        confidence = cached.get('confidence')

        if quality is None:
            try:
                quality = calculate_classification_quality(experiment_dir, house_id)
            except Exception:
                quality = None
        else:
            reused += 1

        if confidence is None:
            try:
                confidence = calculate_confidence_scores(experiment_dir, house_id)
            except Exception:
                confidence = None

        if quality:
            all_quality.append(quality)
        if confidence:
            all_confidence.append(confidence)

        quality_score = quality.get('overall_quality_score') if quality else None

        house_summaries.append({
            'house_id': house_id,
            'total_sessions': total,
            'total_minutes': round(total_minutes, 1),
            'classified_minutes': round(classified_minutes, 1),
            'classified_pct': classified_pct,
            'avg_confidence': avg_conf,
            'quality_score': quality_score,
            'device_counts': device_counts,
            'report_link': report_link,
            'days_span': days_span,
            'classified_min_per_day': classified_min_per_day,
            'spike_count': spike_count,
            'pre_quality': pre_quality,
        })

    if show_timing:
        metrics_time = _time.time() - t0
        cache_msg = f", {reused} reused from cache" if reused else ""
        print(f"  Aggregate: collected metrics for {len(house_summaries)} houses "
              f"({metrics_time:.1f}s{cache_msg})", flush=True)

    # Population statistics
    t1 = _time.time()
    population_stats = {}
    if all_quality and all_confidence:
        try:
            population_stats = compute_population_statistics(all_quality, all_confidence)
        except Exception as e:
            logger.warning(f"Population statistics failed: {e}")
    if show_timing:
        print(f"  Aggregate: population statistics ({_time.time() - t1:.1f}s)", flush=True)

    # Build HTML
    t2 = _time.time()
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    # Extract experiment name (strip timestamp suffix)
    import re as _re
    _m = _re.match(r'^(.+?)_\d{8}_\d{6}$', experiment_dir.name)
    _experiment_name = _m.group(1) if _m else experiment_dir.name

    html = _build_aggregate_html(
        generated_at=generated_at,
        house_summaries=house_summaries,
        population_stats=population_stats,
        experiment_dir=str(experiment_dir),
        cross_house_result=cross_house_result,
        experiment_name=_experiment_name,
    )

    # Save
    if output_path is None:
        output_path = str(experiment_dir / "identification_report_aggregate.html")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    if show_timing:
        print(f"  Aggregate: built HTML + saved ({_time.time() - t2:.1f}s)", flush=True)

    logger.info(f"Aggregate identification report saved to {output_path}")
    return output_path


def _build_aggregate_html(
    generated_at: str,
    house_summaries: List[Dict],
    population_stats: Dict[str, Any],
    experiment_dir: str,
    cross_house_result: Optional[Dict[str, Any]] = None,
    experiment_name: str = '',
) -> str:
    """Build complete aggregate HTML document."""
    n = len(house_summaries)
    if n == 0:
        return _build_empty_aggregate_html(generated_at, experiment_dir)

    # Aggregate stats
    avg_classified = sum(h['classified_pct'] for h in house_summaries) / n
    avg_conf = sum(h['avg_confidence'] for h in house_summaries) / n
    quality_scores = [h['quality_score'] for h in house_summaries
                      if h['quality_score'] is not None and isinstance(h['quality_score'], (int, float))]
    median_quality = sorted(quality_scores)[len(quality_scores) // 2] if quality_scores else 0

    # Total data days
    total_days = sum(h.get('days_span', 0) for h in house_summaries)

    # Device detection rates
    has_boiler = sum(1 for h in house_summaries if h['device_counts'].get('boiler', 0) > 0)
    has_central = sum(1 for h in house_summaries if h['device_counts'].get('central_ac', 0) > 0)
    has_regular = sum(1 for h in house_summaries if h['device_counts'].get('regular_ac', 0) > 0)

    # Input quality tier counts
    tier_counts = {}
    for h in house_summaries:
        tier = _assign_tier(h.get('pre_quality'))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    quality_dist_bar = _build_quality_dist_bar(tier_counts, n)

    # Population stats section
    pop_html = _build_population_section(population_stats) if population_stats else ''
    cross_house_html = _build_cross_house_section(cross_house_result, display_limit=CROSS_HOUSE_DISPLAY_LIMIT) if cross_house_result else ''

    # Per-house table rows
    _td = 'padding:8px 10px;border-bottom:1px solid #eee;'
    table_rows = ''
    for h in sorted(house_summaries, key=lambda x: x['house_id']):
        hid = h['house_id']
        link = f'<a href="{h["report_link"]}" style="color: #667eea; text-decoration: none;">{hid}</a>' if h.get('report_link') else hid

        # Pre-quality
        pq_html = _format_pre_quality(h.get('pre_quality'))
        tier = _assign_tier(h.get('pre_quality'))

        # Device type badges
        badges = ''
        for dt, color in [('boiler', '#007bff'), ('central_ac', '#dc3545'), ('regular_ac', '#e67e22'), ('recurring_pattern', '#17a2b8')]:
            cnt = h['device_counts'].get(dt, 0)
            if cnt > 0:
                badges += f'<span style="background:{color};color:white;padding:1px 6px;border-radius:8px;font-size:0.75em;margin-right:3px;">{dt.split("_")[0]} {cnt}</span>'

        q = h['quality_score']
        q_display = f'{q * 100:.0f}' if q is not None else '-'
        q_val = q if q is not None else 0
        q_color = '#28a745' if q and q >= 0.8 else '#eab308' if q and q >= 0.4 else '#e67e22' if q else '#aaa'
        c_display = f'{h["avg_confidence"] * 100:.0f}'
        c_color = '#28a745' if h['avg_confidence'] >= 0.8 else '#eab308' if h['avg_confidence'] >= 0.4 else '#e67e22'

        days = h.get('days_span', 0)
        cmd = h.get('classified_min_per_day', 0)
        spikes = h.get('spike_count', 0)

        table_rows += f'''
        <tr data-tier="{tier}">
            <td style="{_td}" data-value="{hid}">{link}</td>
            <td style="{_td}text-align:center;" data-value="{days}">{days}</td>
            <td style="{_td}text-align:center;">{pq_html}</td>
            <td style="{_td}text-align:center;" data-value="{h['total_sessions']}">{h['total_sessions']}</td>
            <td style="{_td}text-align:center;" data-value="{cmd:.2f}">{cmd:.1f}</td>
            <td style="{_td}text-align:center;" data-value="{spikes}">{spikes}</td>
            <td style="{_td}text-align:center;" data-value="{h['classified_pct']:.1f}">{h['classified_pct']:.0f}%</td>
            <td style="{_td}text-align:center;color:{c_color};font-weight:600;" data-value="{h['avg_confidence']:.3f}">{c_display}</td>
            <td style="{_td}text-align:center;color:{q_color};font-weight:600;" data-value="{q_val:.3f}">{q_display}</td>
            <td style="{_td}">{badges}</td>
        </tr>'''

    # Tier filter bar
    _tier_styles = {
        'excellent': ('Excellent', '#d4edda', '#155724'),
        'good': ('Good', '#cce5ff', '#004085'),
        'fair': ('Fair', '#fff3cd', '#856404'),
        'poor': ('Poor', '#fde2d4', '#813e1a'),
        'faulty_dead_phase': ('Dead Phase', '#d4c5e2', '#5a3d7a'),
        'faulty_high_nan': ('High NaN', '#e2d5f0', '#6f42c1'),
        'faulty_both': ('Faulty Both', '#c9a3d4', '#4a0e6b'),
        'unknown': ('Unknown', '#e9ecef', '#495057'),
    }
    filter_checkboxes = ''
    for tier_key, (label, bg, fg) in _tier_styles.items():
        cnt = tier_counts.get(tier_key, 0)
        if cnt == 0:
            continue
        filter_checkboxes += (
            f'<label style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;'
            f'border-radius:4px;font-size:0.85em;background:{bg};color:{fg};cursor:pointer;">'
            f'<input type="checkbox" data-filter-tier="{tier_key}" checked '
            f'onchange="updateIdFilter()"> {label} ({cnt})</label> '
        )

    about_html = _build_about_section('identification')
    glossary_html = _build_glossary_section()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Identification - Aggregate Report{' (' + experiment_name + ')' if experiment_name else ''}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background: #FAFBFF; color: #3D3D50; line-height: 1.6;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #7B9BC4 0%, #B488B4 100%);
            color: white; padding: 40px 30px; margin-bottom: 30px; border-radius: 16px;
        }}
        header h1 {{ font-size: 2.2em; margin-bottom: 5px; letter-spacing: -0.3px; font-weight: 700; }}
        section {{
            background: #FFFFFF; border-radius: 14px; padding: 28px;
            margin-bottom: 22px; box-shadow: 0 2px 12px rgba(120,100,160,0.07);
            border: 1px solid #E8E4F0;
        }}
        section h2 {{ color: #3D3D50; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #E8E4F0; font-size: 1.35em; }}
        .summary-card {{
            background: #FAFBFF;
            padding: 25px; border-radius: 12px; text-align: center;
            border: 1px solid #E8E4F0;
        }}
        .summary-number {{ font-size: 2.2em; font-weight: bold; color: #3D3D50; }}
        .summary-label {{ color: #7D7D92; margin-top: 5px; font-size: 0.9em; }}
        .filter-bar {{
            display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
            margin-bottom: 12px; padding: 14px 18px; background: #FAFBFF; border-radius: 10px;
            border: 1px solid #E8E4F0;
        }}
        .filter-bar label {{ font-weight: 600; color: #3D3D50; margin-right: 5px; }}
        footer {{ text-align: center; padding: 20px; color: #7D7D92; font-size: 0.9em; }}
        details.ep-collapsible {{ background:#FFFFFF;border-radius:14px;margin-bottom:22px;box-shadow:0 2px 12px rgba(120,100,160,0.07);border:1px solid #E8E4F0; }}
        details.ep-collapsible > summary {{ cursor:pointer;padding:20px 28px;font-size:1.3em;font-weight:700;color:#3D3D50;list-style:none;user-select:none; }}
        details.ep-collapsible > summary::-webkit-details-marker {{ display:none; }}
        details.ep-collapsible > summary::before {{ content:'\\25B8';display:inline-block;margin-right:10px;transition:transform 0.2s; }}
        details.ep-collapsible[open] > summary::before {{ transform:rotate(90deg); }}
        details.ep-collapsible[open] > summary {{ padding-bottom:16px;border-bottom:2px solid #E8E4F0; }}
        details.ep-collapsible > .collapsible-body {{ padding:20px 28px 28px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Identification &mdash; Aggregate Report</h1>
            <div style="opacity:0.92;">{'<strong>' + experiment_name + '</strong> | ' if experiment_name else ''}Generated: {generated_at}</div>
        </header>

        {about_html}

        <section>
            <h2>Summary</h2>
            <!-- Row 1: Houses + Days -->
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:18px;">
                <div class="summary-card">
                    <div class="summary-number" id="dyn-houses">{n}</div>
                    <div class="summary-label">Houses Analyzed</div>
                </div>
                <div class="summary-card">
                    <div class="summary-number" id="dyn-days">{total_days:,}</div>
                    <div class="summary-label">Total Days of Data</div>
                </div>
            </div>
            <!-- Row 2: Quality distribution -->
            {quality_dist_bar}
            <!-- Row 3: Report-specific metrics -->
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px;margin-bottom:20px;">
                <div style="background:#d4edda;border-radius:8px;padding:16px;text-align:center;">
                    <div id="dyn-classified" style="font-size:2em;font-weight:700;color:#28a745;">{avg_classified:.0f}%</div>
                    <div style="font-size:0.85em;color:#666;">Avg Classified (by minutes)</div>
                </div>
                <div style="background:#f0f4ff;border-radius:8px;padding:16px;text-align:center;">
                    <div id="dyn-confidence" style="font-size:2em;font-weight:700;color:#7B9BC4;">{avg_conf * 100:.0f}/100</div>
                    <div style="font-size:0.85em;color:#7D7D92;">Avg Confidence</div>
                </div>
                <div style="background:#f0f4ff;border-radius:8px;padding:16px;text-align:center;">
                    <div id="dyn-quality" style="font-size:2em;font-weight:700;color:#7B9BC4;">{median_quality * 100:.0f}/100</div>
                    <div style="font-size:0.85em;color:#7D7D92;">Median Quality</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                <div style="background:#f0f4ff;border:1px solid #c3d4ff;border-radius:8px;padding:12px;text-align:center;">
                    <div id="dyn-boiler" style="font-size:1.5em;font-weight:700;color:#007bff;">{has_boiler}/{n}</div>
                    <div style="font-size:0.82em;color:#666;">Houses with Boiler</div>
                </div>
                <div style="background:#fff5f5;border:1px solid #fed7d7;border-radius:8px;padding:12px;text-align:center;">
                    <div id="dyn-central" style="font-size:1.5em;font-weight:700;color:#dc3545;">{has_central}/{n}</div>
                    <div style="font-size:0.82em;color:#666;">Houses with Central AC</div>
                </div>
                <div style="background:#fff8f0;border:1px solid #feebc8;border-radius:8px;padding:12px;text-align:center;">
                    <div id="dyn-regular" style="font-size:1.5em;font-weight:700;color:#e67e22;">{has_regular}/{n}</div>
                    <div style="font-size:0.82em;color:#666;">Houses with Regular AC</div>
                </div>
            </div>
        </section>

        {pop_html}

        {cross_house_html}

        <section>
            <h2>Per-House Results</h2>
            <div class="filter-bar">
                <span style="font-weight:600;color:#555;margin-right:5px;">Quality Tier:</span>
                {filter_checkboxes}
                <span id="id-filter-status" style="font-size:0.85em;color:#888;margin-left:auto;"></span>
            </div>
            <div style="font-size:0.82em;color:#666;margin-bottom:10px;line-height:1.7;">
                <strong>Column descriptions:</strong>
                <strong>Days</strong> = calendar days from first to last session |
                <strong>Pre-Quality</strong> = input data quality from house pre-analysis |
                <strong>Sessions</strong> = total device sessions found |
                <strong>Cls min/day</strong> = classified minutes per day (boiler, AC time identified per day of data) |
                <strong>Spikes</strong> = transient events filtered out (&lt;3 min) |
                <strong>Classified</strong> = % of segregated minutes assigned to a device type (by duration, not count) |
                <strong>Confidence</strong> = avg classification confidence of classified sessions (0&ndash;100) |
                <strong>Quality</strong> = internal consistency score (0&ndash;100)
            </div>
            <div style="overflow-x:auto;">
            <table id="agg-table" style="width:100%;border-collapse:collapse;font-size:0.88em;">
                <thead>
                    <tr style="background:#7888A0;color:white;">
                        <th style="padding:8px 10px;text-align:left;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(0,'str')" title="House identifier">House &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(1,'num')" title="Calendar days from first to last session">Days &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(2,'num')" title="Input data quality score from house pre-analysis">Pre-Quality &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(3,'num')" title="Total device sessions found">Sessions &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(4,'num')" title="Classified minutes per day (boiler, AC, etc.)">Cls min/day &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(5,'num')" title="Transient events filtered (<3 min)">Spikes &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(6,'num')" title="% of segregated minutes assigned to a device type">Classified &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(7,'num')" title="Average classification confidence (0-100)">Confidence &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:center;cursor:pointer;white-space:nowrap;" onclick="sortAggTable(8,'num')" title="Internal consistency quality score (0-100)">Quality &#x25B4;&#x25BE;</th>
                        <th style="padding:8px 10px;text-align:left;white-space:nowrap;">Devices</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            </div>
        </section>

        {glossary_html}

        <footer>
            ElectricPatterns &mdash; Module 2: Device Identification Aggregate Report
        </footer>
    </div>

    <script>
    var aggSortState = {{}};
    function sortAggTable(colIdx, type) {{
        var table = document.getElementById('agg-table');
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var key = 'agg-' + colIdx;
        var asc = aggSortState[key] === undefined ? true : !aggSortState[key];
        aggSortState[key] = asc;
        rows.sort(function(a, b) {{
            var cellA = a.cells[colIdx], cellB = b.cells[colIdx];
            var vA, vB;
            if (type === 'num') {{
                vA = parseFloat(cellA.getAttribute('data-value') || cellA.textContent.replace(/[^0-9.-]/g, '')) || 0;
                vB = parseFloat(cellB.getAttribute('data-value') || cellB.textContent.replace(/[^0-9.-]/g, '')) || 0;
            }} else {{
                vA = (cellA.getAttribute('data-value') || cellA.textContent).trim();
                vB = (cellB.getAttribute('data-value') || cellB.textContent).trim();
                if (vA < vB) return asc ? -1 : 1;
                if (vA > vB) return asc ? 1 : -1;
                return 0;
            }}
            return asc ? (vA - vB) : (vB - vA);
        }});
        rows.forEach(function(row) {{ tbody.appendChild(row); }});
        var ths = table.querySelectorAll('thead th');
        ths.forEach(function(th, i) {{
            th.style.fontWeight = (i === colIdx) ? '900' : 'normal';
        }});
    }}

    function updateIdFilter() {{
        var checkedTiers = [];
        document.querySelectorAll('[data-filter-tier]').forEach(function(cb) {{
            if (cb.checked) checkedTiers.push(cb.getAttribute('data-filter-tier'));
        }});
        var rows = document.querySelectorAll('#agg-table tbody tr');
        var shown = 0;
        rows.forEach(function(row) {{
            var tier = row.getAttribute('data-tier') || 'unknown';
            if (checkedTiers.indexOf(tier) !== -1) {{
                row.style.display = '';
                shown++;
            }} else {{
                row.style.display = 'none';
            }}
        }});
        var status = document.getElementById('id-filter-status');
        if (status) status.textContent = 'Showing ' + shown + ' / ' + rows.length + ' houses';
        updateIdentSummary();
    }}

    function updateIdentSummary() {{
        var rows = document.querySelectorAll('#agg-table tbody tr');
        var vis = [];
        rows.forEach(function(r) {{ if (r.style.display !== 'none') vis.push(r); }});
        var n = vis.length;
        var totalDays = 0, sumCls = 0, sumConf = 0;
        var qVals = [];
        var nBoiler = 0, nCentral = 0, nRegular = 0;
        vis.forEach(function(row) {{
            totalDays += parseFloat(row.cells[1].getAttribute('data-value') || 0);
            sumCls += parseFloat(row.cells[6].getAttribute('data-value') || 0);
            sumConf += parseFloat(row.cells[7].getAttribute('data-value') || 0);
            var qv = parseFloat(row.cells[8].getAttribute('data-value') || 0);
            if (qv > 0) qVals.push(qv);
            var dev = row.cells[9].textContent;
            if (dev.indexOf('boiler') !== -1) nBoiler++;
            if (dev.indexOf('central') !== -1) nCentral++;
            if (dev.indexOf('regular') !== -1) nRegular++;
        }});
        var avgCls = n > 0 ? sumCls / n : 0;
        var avgConf = n > 0 ? sumConf / n : 0;
        qVals.sort(function(a,b){{return a-b;}});
        var medQ = qVals.length > 0 ? qVals[Math.floor(qVals.length/2)] : 0;
        function u(id, val) {{ var el = document.getElementById(id); if (el) el.textContent = val; }}
        u('dyn-houses', n);
        u('dyn-days', totalDays.toLocaleString());
        u('dyn-classified', avgCls.toFixed(0) + '%');
        u('dyn-confidence', (avgConf * 100).toFixed(0) + '/100');
        u('dyn-quality', (medQ * 100).toFixed(0) + '/100');
        u('dyn-boiler', nBoiler + '/' + n);
        u('dyn-central', nCentral + '/' + n);
        u('dyn-regular', nRegular + '/' + n);
    }}

    updateIdFilter();
    </script>
</body>
</html>"""


def _build_cross_house_section(
    cross_house_result: Optional[Dict[str, Any]],
    display_limit: Optional[int] = None,
) -> str:
    """Build HTML section for cross-house recurring pattern matches.

    Parameters
    ----------
    display_limit : int or None
        If set, only show the top N patterns (by total session count).
        Remaining patterns are summarised in a short note.
    """
    if not cross_house_result:
        return ''

    summary = cross_house_result.get('summary', {})
    global_patterns = cross_house_result.get('global_patterns', [])
    unmatched = cross_house_result.get('unmatched_patterns', [])
    n_global = summary.get('total_global_patterns', 0)
    n_unmatched = summary.get('total_unmatched', 0)
    n_houses = summary.get('houses_with_patterns', 0)
    n_sigs = summary.get('total_signatures', 0)

    if n_sigs == 0:
        return ''

    # Sort by total session count across houses (most-used first)
    sorted_patterns = sorted(
        global_patterns,
        key=lambda gp: sum(h.get('n_sessions', 0) for h in gp.get('houses', [])),
        reverse=True,
    )
    if display_limit is not None and len(sorted_patterns) > display_limit:
        display_patterns = sorted_patterns[:display_limit]
        n_omitted = len(sorted_patterns) - display_limit
    else:
        display_patterns = sorted_patterns
        n_omitted = 0

    # Build pattern cards
    pattern_cards = ''
    for gp in display_patterns:
        name = gp['global_name']
        desc = gp['descriptive_name']
        n_h = gp.get('n_houses', len(gp.get('houses', [])))
        quality = gp.get('match_quality', 0)
        q_color = '#48bb78' if quality >= 0.7 else '#ecc94b' if quality >= 0.5 else '#fc8181'

        house_rows = ''
        for h in gp.get('houses', []):
            house_rows += f'''
                <tr>
                    <td style="padding:4px 8px;">{h['house_id']}</td>
                    <td style="padding:4px 8px;text-align:center;">#{h['pattern_id']}</td>
                    <td style="padding:4px 8px;text-align:right;">{h['avg_magnitude_w']:,.0f}W</td>
                    <td style="padding:4px 8px;text-align:right;">{h['avg_duration_min']:.1f} min</td>
                    <td style="padding:4px 8px;text-align:center;">{h['n_sessions']}</td>
                    <td style="padding:4px 8px;text-align:center;">{h['confidence']:.0%}</td>
                </tr>'''

        pattern_cards += f'''
        <div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;padding:14px;margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <strong style="font-size:1.1em;color:#0d9488;">{name}</strong>
                <span style="font-size:0.85em;color:#666;">{desc}</span>
                <span style="background:#17a2b8;color:white;padding:1px 8px;border-radius:10px;font-size:0.8em;">{n_h} houses</span>
                <span style="color:{q_color};font-weight:600;font-size:0.85em;">Match: {quality:.0%}</span>
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:0.85em;">
                <thead>
                    <tr style="background:#e6fffa;">
                        <th style="padding:4px 8px;text-align:left;">House</th>
                        <th style="padding:4px 8px;text-align:center;">Pattern</th>
                        <th style="padding:4px 8px;text-align:right;">Avg Power</th>
                        <th style="padding:4px 8px;text-align:right;">Avg Duration</th>
                        <th style="padding:4px 8px;text-align:center;">Sessions</th>
                        <th style="padding:4px 8px;text-align:center;">Confidence</th>
                    </tr>
                </thead>
                <tbody>{house_rows}
                </tbody>
            </table>
        </div>'''

    # Note about omitted patterns
    if n_omitted > 0:
        pattern_cards += f'''
        <div style="padding:10px 16px;background:#f0f9ff;border:1px solid #bae6fd;
                    border-radius:8px;margin-top:4px;margin-bottom:12px;font-size:0.85em;color:#0369a1;">
            {n_omitted} additional cross-house pattern{'s' if n_omitted != 1 else ''}
            not shown here. These can be found in the per-house reports.
        </div>'''

    # Unmatched summary (collapsed)
    unmatched_html = ''
    if unmatched:
        unmatched_rows = ''
        for u in unmatched:
            unmatched_rows += f'''
                <tr>
                    <td style="padding:3px 8px;">{u['house_id']}</td>
                    <td style="padding:3px 8px;text-align:center;">#{u['pattern_id']}</td>
                    <td style="padding:3px 8px;text-align:right;">{u['avg_magnitude_w']:,.0f}W</td>
                    <td style="padding:3px 8px;text-align:right;">{u['avg_duration_min']:.1f} min</td>
                    <td style="padding:3px 8px;text-align:center;">{u['n_sessions']}</td>
                </tr>'''
        unmatched_html = f'''
        <div style="margin-top:12px;">
            <div style="cursor:pointer;font-size:0.85em;color:#888;"
                 onclick="var el=document.getElementById('cross-house-unmatched');el.style.display=el.style.display==='none'?'block':'none';this.querySelector('.arrow').textContent=el.style.display==='none'?'\\u25B6':'\\u25BC';">
                <span class="arrow" style="font-size:0.8em;">&#x25B6;</span>
                {n_unmatched} house-unique patterns (no cross-house match)
            </div>
            <div id="cross-house-unmatched" style="display:none;margin-top:6px;">
                <table style="width:100%;border-collapse:collapse;font-size:0.82em;">
                    <thead>
                        <tr style="background:#f8f9fa;">
                            <th style="padding:3px 8px;text-align:left;">House</th>
                            <th style="padding:3px 8px;text-align:center;">Pattern</th>
                            <th style="padding:3px 8px;text-align:right;">Avg Power</th>
                            <th style="padding:3px 8px;text-align:right;">Avg Duration</th>
                            <th style="padding:3px 8px;text-align:center;">Sessions</th>
                        </tr>
                    </thead>
                    <tbody>{unmatched_rows}
                    </tbody>
                </table>
            </div>
        </div>'''

    settings = cross_house_result.get('settings', {})
    mag_tol_pct = int(settings.get('magnitude_tolerance', 0.15) * 100)
    dur_tol_pct = int(settings.get('duration_tolerance', 0.20) * 100)

    return f'''
        <details class="ep-collapsible">
            <summary>Cross-House Recurring Patterns</summary>
            <div class="collapsible-body">
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:15px;">
                <div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5em;font-weight:700;color:#0d9488;">{n_global}</div>
                    <div style="font-size:0.82em;color:#666;">Cross-House Patterns</div>
                </div>
                <div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5em;font-weight:700;color:#0d9488;">{n_houses}</div>
                    <div style="font-size:0.82em;color:#666;">Houses with Patterns</div>
                </div>
                <div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5em;font-weight:700;color:#0d9488;">{n_sigs}</div>
                    <div style="font-size:0.82em;color:#666;">Total Pattern Signatures</div>
                </div>
                <div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5em;font-weight:700;color:#0d9488;">{n_unmatched}</div>
                    <div style="font-size:0.82em;color:#666;">House-Unique Patterns</div>
                </div>
            </div>
            <p style="font-size:0.85em;color:#666;margin-bottom:12px;">
                Recurring patterns discovered independently per-house are compared across all houses.
                Patterns with similar power (&le;{mag_tol_pct}%)
                and duration (&le;{dur_tol_pct}%)
                are grouped under a shared name (Device A, B, ...).
            </p>
            {pattern_cards}
            {unmatched_html}
            </div>
        </details>'''


def _build_population_section(population_stats: Dict[str, Any]) -> str:
    """Build HTML section for population-level statistics."""
    if not population_stats or population_stats.get('houses_analyzed', 0) == 0:
        return ''

    n = population_stats['houses_analyzed']
    outliers = population_stats.get('outlier_houses', [])
    per_device = population_stats.get('per_device_type', {})

    device_cards = ''
    for dtype in ['boiler', 'central_ac', 'regular_ac', 'recurring_pattern']:
        if dtype not in per_device:
            continue
        d = per_device[dtype]
        count_dist = d.get('count_per_month', {})
        mag_dist = d.get('mean_magnitude', {})
        dur_dist = d.get('median_duration', {})
        houses_with = d.get("houses_with_device", 0)
        pct = (houses_with / n * 100) if n > 0 else 0
        device_cards += f'''
        <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px;">
            <div style="font-weight:600;color:#2a4365;margin-bottom:6px;">{dtype.replace("_", " ").title()}</div>
            <div style="font-size:0.85em;color:#555;">
                Houses: <strong>{houses_with}/{n}</strong> ({pct:.0f}%)<br>
                Median magnitude: {mag_dist.get("median", 0):.0f}W<br>
                Median duration: {dur_dist.get("median", 0):.0f} min<br>
                Months active/house: median {count_dist.get("median", 0):.0f}
            </div>
        </div>'''

    outlier_html = ''
    if outliers:
        outlier_list = ', '.join(str(o.get('house_id', '')) for o in outliers[:10])
        outlier_html = f'''
        <div style="margin-top:12px;padding:10px;background:#fff3cd;border:1px solid #ffc107;border-radius:6px;font-size:0.85em;">
            <strong>Outlier houses ({len(outliers)}):</strong> {outlier_list}
        </div>'''

    return f'''
    <details class="ep-collapsible">
        <summary>Population Statistics</summary>
        <div class="collapsible-body">
        <p style="color:#666;margin-bottom:12px;font-size:0.85em;">
            Cross-house analysis of device identification patterns.
            Z-scores flag houses with unusual device characteristics.
        </p>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
            {device_cards}
        </div>
        {outlier_html}
        </div>
    </details>'''


def _build_empty_aggregate_html(generated_at: str, experiment_dir: str) -> str:
    """Fallback HTML when no valid data found."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Identification - Aggregate Report{' (' + experiment_name + ')' if experiment_name else ''}</title>
</head>
<body style="font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px;">
    <h1>Device Identification - Aggregate Report</h1>
    <p>Generated: {generated_at}</p>
    <p style="color: #888;">No device session data found in: {experiment_dir}</p>
    <p>Ensure the pipeline was run with device identification enabled.</p>
</body>
</html>"""
