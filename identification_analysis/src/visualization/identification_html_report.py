"""
HTML report generator for device identification (Module 2).

Generates stand-alone HTML reports per house and aggregate reports.
Follows the same patterns as dynamic_html_report.py (inline CSS, Plotly CDN).

Sections:
  1. Session Overview Dashboard — pie/bar of device types, summary cards
  2. Confidence Distribution — histogram + tier breakdown
  3. Boiler Analysis — daily pattern, monthly consistency, magnitude
  4. AC Analysis — central vs regular, seasonal, cycles
  5. Temporal Patterns — hour x day-of-week heatmap
  6. Classification Quality — 5 metrics + flags (from classification_quality.py)
  7. Unclassified Analysis — power/duration distributions
  8. Device Activations Detail — per-event sortable tables
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from metrics.classification_quality import calculate_classification_quality
from metrics.confidence_scoring import calculate_confidence_scores
from metrics.population_statistics import compute_population_statistics

from visualization.identification_charts import (
    create_session_overview,
    create_confidence_overview,
    create_boiler_analysis,
    create_ac_analysis,
    create_temporal_heatmap,
    create_unclassified_analysis,
    create_device_activations_detail,
)
from visualization.classification_charts import create_quality_section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_sessions(experiment_dir: Path, house_id: str) -> Dict[str, Any]:
    """Load device_sessions JSON for a house."""
    candidates = [
        experiment_dir / "device_sessions" / f"device_sessions_{house_id}.json",
        experiment_dir / f"device_sessions_{house_id}.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load sessions JSON from {path}: {e}")
    return {}


def _load_activations(experiment_dir: Path, house_id: str) -> list:
    """Load device_activations JSON (flat format) for backward-compatible detail tables."""
    candidates = [
        experiment_dir / "device_activations" / f"device_activations_{house_id}.json",
        experiment_dir / f"device_activations_{house_id}.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return data.get('activations', [])
            except Exception as e:
                logger.warning(f"Failed to load activations JSON from {path}: {e}")
    return []


# ---------------------------------------------------------------------------
# Per-house report
# ---------------------------------------------------------------------------

def generate_identification_report(
    experiment_dir: str,
    house_id: str,
    output_path: Optional[str] = None,
    skip_activations_detail: bool = False,
) -> str:
    """
    Generate device identification HTML report for a single house.

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID
        output_path: Where to save the HTML file (optional)
        skip_activations_detail: If True, omit the Device Activations Detail section

    Returns:
        Path to generated HTML file
    """
    experiment_dir = Path(experiment_dir)

    # Load data
    sessions_data = _load_sessions(experiment_dir, house_id)
    sessions = sessions_data.get('sessions', [])
    threshold_schedule = sessions_data.get('threshold_schedule', [])

    # Generate chart sections
    overview_html = create_session_overview(sessions)
    confidence_html = create_confidence_overview(sessions)
    boiler_html = create_boiler_analysis(sessions)
    ac_html = create_ac_analysis(sessions)
    heatmap_html = create_temporal_heatmap(sessions)
    unclassified_html = create_unclassified_analysis(sessions)

    # Classification quality metrics (reuse existing M2 metrics)
    quality_html = ''
    try:
        quality = calculate_classification_quality(experiment_dir, house_id)
        confidence = calculate_confidence_scores(experiment_dir, house_id)
        quality_html = create_quality_section(quality, confidence)
    except Exception as e:
        logger.warning(f"Classification quality failed for house {house_id}: {e}")

    # Device activations detail (flat format)
    activations_detail_html = ''
    if not skip_activations_detail:
        activations = _load_activations(experiment_dir, house_id)
        activations_detail_html = create_device_activations_detail(activations)

    # Build HTML
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    summary = sessions_data.get('summary', {})

    html = _build_house_html(
        house_id=house_id,
        generated_at=generated_at,
        threshold_schedule=threshold_schedule,
        overview_html=overview_html,
        confidence_html=confidence_html,
        boiler_html=boiler_html,
        ac_html=ac_html,
        heatmap_html=heatmap_html,
        quality_html=quality_html,
        unclassified_html=unclassified_html,
        activations_detail_html=activations_detail_html,
        summary=summary,
    )

    # Save
    if output_path is None:
        output_path = str(experiment_dir / f"identification_report_{house_id}.html")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Identification report saved to {output_path}")
    return output_path


def _build_house_html(
    house_id: str,
    generated_at: str,
    threshold_schedule: list,
    overview_html: str,
    confidence_html: str,
    boiler_html: str,
    ac_html: str,
    heatmap_html: str,
    quality_html: str,
    unclassified_html: str,
    activations_detail_html: str,
    summary: Dict[str, Any],
) -> str:
    """Build complete HTML document for a single house identification report."""
    th_str = ' \u2192 '.join(f'{t}W' for t in threshold_schedule) if threshold_schedule else 'N/A'
    total_sessions = summary.get('total_sessions', 0)
    by_type = summary.get('by_device_type', {})

    # Type summary for header
    type_parts = []
    for dtype in ['boiler', 'central_ac', 'regular_ac', 'unknown']:
        info = by_type.get(dtype, {})
        count = info.get('count', 0)
        if count > 0:
            type_parts.append(f'{dtype}: {count}')
    type_str = ', '.join(type_parts) if type_parts else 'No sessions'

    # Activations detail section
    activations_section_html = ''
    if activations_detail_html:
        activations_section_html = f'''
        <section>
            <h2>Device Activations Detail</h2>
            <p style="color: #666; margin-bottom: 10px; font-size: 0.85em;">
                Individual ON&rarr;OFF activations grouped by device type.
                Click column headers to sort. Use "Copy Dates" for external tools.
            </p>
            {activations_detail_html}
        </section>'''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Device Identification Report - House {house_id}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #1a365d 0%, #2a4365 50%, #2c5282 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}
        header h1 {{ font-size: 2em; margin-bottom: 5px; }}
        header .subtitle {{ opacity: 0.9; font-size: 1.1em; }}
        .info-bar {{ display: flex; gap: 30px; margin-top: 15px; flex-wrap: wrap; }}
        .info-item {{ font-size: 0.9em; opacity: 0.85; }}
        .info-item strong {{ opacity: 1; }}
        section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        section h2 {{
            color: #2a4365;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e2e8f0;
        }}
        footer {{ text-align: center; padding: 20px; color: #888; font-size: 0.9em; }}
        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            header {{ padding: 20px; }}
            section {{ padding: 15px; }}
            .info-bar {{ flex-direction: column; gap: 5px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Device Identification Report</h1>
            <div class="subtitle">House {house_id} &mdash; Module 2 Analysis</div>
            <div class="info-bar">
                <div class="info-item"><strong>Generated:</strong> {generated_at}</div>
                <div class="info-item"><strong>Thresholds:</strong> {th_str}</div>
                <div class="info-item"><strong>Sessions:</strong> {total_sessions} ({type_str})</div>
            </div>
        </header>

        <section>
            <h2>Session Overview</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Summary of all detected device sessions. Each session is a group of temporally close
                matched ON&rarr;OFF events on the same phase, classified by power, duration, and phase patterns.
            </p>
            {overview_html}
        </section>

        <section>
            <h2>Confidence Distribution</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Confidence scores (0&ndash;1) for classified sessions based on how well each session
                matches its device type criteria (duration, magnitude, isolation, phase consistency).
            </p>
            {confidence_html}
        </section>

        <section>
            <h2>Boiler Analysis</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Water heater detection analysis: daily usage patterns, monthly consistency,
                power magnitude stability, and dominant phase.
                Boiler criteria: &ge;25 min, &ge;1500W, single-phase, isolated.
            </p>
            {boiler_html}
        </section>

        <section>
            <h2>AC Analysis</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Air conditioning detection: Central AC (multi-phase synchronized) vs Regular AC
                (single-phase compressor cycling). Seasonal patterns and cycle characteristics.
            </p>
            {ac_html}
        </section>

        <section>
            <h2>Temporal Patterns</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                When are devices active? Heatmap shows session start times by hour and day of week.
                Expected patterns: boiler morning/evening, AC afternoon/night in summer.
            </p>
            {heatmap_html}
        </section>

        {quality_html}

        <section>
            <h2>Unclassified Sessions</h2>
            <p style="color: #666; margin-bottom: 12px; font-size: 0.85em;">
                Sessions that did not match any device classification rule.
                These may represent small appliances, partial detections, or device types
                not covered by the current heuristic rules.
            </p>
            {unclassified_html}
        </section>

        {activations_section_html}

        <footer>
            ElectricPatterns &mdash; Module 2: Device Identification Report
        </footer>
    </div>

    <script>
    var deviceSortState = {{}};
    function sortDeviceTable(tableId, colIdx, type) {{
        var table = document.getElementById(tableId);
        if (!table) return;
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var key = tableId + '-' + colIdx;
        var asc = deviceSortState[key] === undefined ? true : !deviceSortState[key];
        deviceSortState[key] = asc;
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
        rows.forEach(function(row, i) {{
            row.cells[0].textContent = i + 1;
            tbody.appendChild(row);
        }});
    }}
    </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Aggregate report (cross-house)
# ---------------------------------------------------------------------------

def generate_identification_aggregate_report(
    experiment_dir: str,
    house_ids: List[str],
    output_path: Optional[str] = None,
    house_reports_subdir: Optional[str] = None,
    show_progress: bool = False,
) -> str:
    """
    Generate aggregate identification report across multiple houses.

    Args:
        experiment_dir: Root experiment output directory
        house_ids: List of house IDs
        output_path: Where to save (optional)
        house_reports_subdir: Subdirectory for per-house report links
        show_progress: Show tqdm progress bar

    Returns:
        Path to generated HTML file
    """
    experiment_dir = Path(experiment_dir)

    all_quality = []
    all_confidence = []
    house_summaries = []

    houses_iter = house_ids
    if show_progress and _HAS_TQDM:
        houses_iter = _tqdm(house_ids, desc="Aggregate M2 metrics", unit="house")

    for house_id in houses_iter:
        sessions_data = _load_sessions(experiment_dir, house_id)
        sessions = sessions_data.get('sessions', [])
        summary = sessions_data.get('summary', {})

        if not sessions:
            continue

        # Collect per-house summary
        total = len(sessions)
        classified = sum(1 for s in sessions if s.get('device_type') not in ('unknown', 'unclassified'))
        conf_vals = [s.get('confidence', 0) for s in sessions if s.get('confidence')]
        avg_conf = sum(conf_vals) / len(conf_vals) if conf_vals else 0

        # Device counts
        device_counts = {}
        for s in sessions:
            dt = s.get('device_type', 'unknown')
            device_counts[dt] = device_counts.get(dt, 0) + 1

        # Quality + confidence metrics
        try:
            quality = calculate_classification_quality(experiment_dir, house_id)
            all_quality.append(quality)
        except Exception:
            quality = None

        try:
            confidence = calculate_confidence_scores(experiment_dir, house_id)
            all_confidence.append(confidence)
        except Exception:
            confidence = None

        quality_score = quality.get('overall_quality_score') if quality else None
        report_link = None
        if house_reports_subdir:
            report_link = f'{house_reports_subdir}/identification_report_{house_id}.html'

        house_summaries.append({
            'house_id': house_id,
            'total_sessions': total,
            'classified': classified,
            'classified_pct': classified / total * 100 if total > 0 else 0,
            'avg_confidence': avg_conf,
            'quality_score': quality_score,
            'device_counts': device_counts,
            'report_link': report_link,
        })

    # Population statistics
    population_stats = {}
    if all_quality and all_confidence:
        try:
            population_stats = compute_population_statistics(all_quality, all_confidence)
        except Exception as e:
            logger.warning(f"Population statistics failed: {e}")

    # Build HTML
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = _build_aggregate_html(
        generated_at=generated_at,
        house_summaries=house_summaries,
        population_stats=population_stats,
        experiment_dir=str(experiment_dir),
    )

    # Save
    if output_path is None:
        output_path = str(experiment_dir / "identification_report_aggregate.html")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"Aggregate identification report saved to {output_path}")
    return output_path


def _build_aggregate_html(
    generated_at: str,
    house_summaries: List[Dict],
    population_stats: Dict[str, Any],
    experiment_dir: str,
) -> str:
    """Build complete aggregate HTML document."""
    n = len(house_summaries)
    if n == 0:
        return _build_empty_aggregate_html(generated_at, experiment_dir)

    # Aggregate stats
    avg_classified = sum(h['classified_pct'] for h in house_summaries) / n
    avg_conf = sum(h['avg_confidence'] for h in house_summaries) / n
    quality_scores = [h['quality_score'] for h in house_summaries if h['quality_score'] is not None]
    median_quality = sorted(quality_scores)[len(quality_scores) // 2] if quality_scores else 0

    # Device detection rates
    has_boiler = sum(1 for h in house_summaries if h['device_counts'].get('boiler', 0) > 0)
    has_central = sum(1 for h in house_summaries if h['device_counts'].get('central_ac', 0) > 0)
    has_regular = sum(1 for h in house_summaries if h['device_counts'].get('regular_ac', 0) > 0)

    # Population stats section
    pop_html = _build_population_section(population_stats) if population_stats else ''

    # Per-house table rows
    table_rows = ''
    for h in sorted(house_summaries, key=lambda x: x['house_id']):
        hid = h['house_id']
        link = f'<a href="{h["report_link"]}" style="color: #667eea; text-decoration: none;">{hid}</a>' if h.get('report_link') else hid

        # Device type badges
        badges = ''
        for dt, color in [('boiler', '#007bff'), ('central_ac', '#dc3545'), ('regular_ac', '#e67e22')]:
            cnt = h['device_counts'].get(dt, 0)
            if cnt > 0:
                badges += f'<span style="background:{color};color:white;padding:1px 6px;border-radius:8px;font-size:0.75em;margin-right:3px;">{dt.split("_")[0]} {cnt}</span>'

        q = h['quality_score']
        q_str = f'{q:.2f}' if q is not None else '-'
        q_color = '#28a745' if q and q >= 0.7 else '#eab308' if q and q >= 0.4 else '#e67e22' if q else '#aaa'
        c_color = '#28a745' if h['avg_confidence'] >= 0.7 else '#eab308' if h['avg_confidence'] >= 0.4 else '#e67e22'

        table_rows += f'''
        <tr>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;">{link}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center;">{h['total_sessions']}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center;">{h['classified_pct']:.0f}%</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center;color:{c_color};font-weight:600;">{h['avg_confidence']:.2f}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center;color:{q_color};font-weight:600;">{q_str}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #eee;">{badges}</td>
        </tr>'''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Device Identification - Aggregate Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa; color: #333; line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        header {{
            background: linear-gradient(135deg, #1a365d 0%, #2a4365 50%, #2c5282 100%);
            color: white; padding: 30px; margin-bottom: 30px; border-radius: 10px;
        }}
        header h1 {{ font-size: 2em; margin-bottom: 5px; }}
        section {{
            background: white; border-radius: 10px; padding: 25px;
            margin-bottom: 25px; box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        section h2 {{ color: #2a4365; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid #e2e8f0; }}
        footer {{ text-align: center; padding: 20px; color: #888; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Device Identification &mdash; Aggregate Report</h1>
            <div style="opacity:0.9;">Generated: {generated_at} | {n} houses analyzed</div>
        </header>

        <section>
            <h2>Population Summary</h2>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;">
                <div style="background:#f0f4ff;border-radius:8px;padding:16px;text-align:center;">
                    <div style="font-size:2em;font-weight:700;color:#2d3748;">{n}</div>
                    <div style="font-size:0.85em;color:#666;">Houses Analyzed</div>
                </div>
                <div style="background:#d4edda;border-radius:8px;padding:16px;text-align:center;">
                    <div style="font-size:2em;font-weight:700;color:#28a745;">{avg_classified:.0f}%</div>
                    <div style="font-size:0.85em;color:#666;">Avg Classified</div>
                </div>
                <div style="background:#f0f4ff;border-radius:8px;padding:16px;text-align:center;">
                    <div style="font-size:2em;font-weight:700;color:#667eea;">{avg_conf:.2f}</div>
                    <div style="font-size:0.85em;color:#666;">Avg Confidence</div>
                </div>
                <div style="background:#f0f4ff;border-radius:8px;padding:16px;text-align:center;">
                    <div style="font-size:2em;font-weight:700;color:#667eea;">{median_quality:.2f}</div>
                    <div style="font-size:0.85em;color:#666;">Median Quality</div>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                <div style="background:#f0f4ff;border:1px solid #c3d4ff;border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5em;font-weight:700;color:#007bff;">{has_boiler}/{n}</div>
                    <div style="font-size:0.82em;color:#666;">Houses with Boiler</div>
                </div>
                <div style="background:#fff5f5;border:1px solid #fed7d7;border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5em;font-weight:700;color:#dc3545;">{has_central}/{n}</div>
                    <div style="font-size:0.82em;color:#666;">Houses with Central AC</div>
                </div>
                <div style="background:#fff8f0;border:1px solid #feebc8;border-radius:8px;padding:12px;text-align:center;">
                    <div style="font-size:1.5em;font-weight:700;color:#e67e22;">{has_regular}/{n}</div>
                    <div style="font-size:0.82em;color:#666;">Houses with Regular AC</div>
                </div>
            </div>
        </section>

        {pop_html}

        <section>
            <h2>Per-House Results</h2>
            <table style="width:100%;border-collapse:collapse;font-size:0.9em;">
                <thead>
                    <tr style="background:#2d3748;color:white;">
                        <th style="padding:10px 12px;text-align:left;">House</th>
                        <th style="padding:10px 12px;text-align:center;">Sessions</th>
                        <th style="padding:10px 12px;text-align:center;">Classified</th>
                        <th style="padding:10px 12px;text-align:center;">Confidence</th>
                        <th style="padding:10px 12px;text-align:center;">Quality</th>
                        <th style="padding:10px 12px;text-align:left;">Devices</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </section>

        <footer>
            ElectricPatterns &mdash; Module 2: Device Identification Aggregate Report
        </footer>
    </div>
</body>
</html>"""


def _build_population_section(population_stats: Dict[str, Any]) -> str:
    """Build HTML section for population-level statistics."""
    if not population_stats or population_stats.get('houses_analyzed', 0) == 0:
        return ''

    n = population_stats['houses_analyzed']
    outliers = population_stats.get('outlier_houses', [])
    per_device = population_stats.get('per_device_type', {})

    device_cards = ''
    for dtype in ['boiler', 'central_ac', 'regular_ac']:
        if dtype not in per_device:
            continue
        d = per_device[dtype]
        count_dist = d.get('count', {})
        mag_dist = d.get('magnitude', {})
        device_cards += f'''
        <div style="background:white;border:1px solid #e2e8f0;border-radius:8px;padding:14px;">
            <div style="font-weight:600;color:#2a4365;margin-bottom:6px;">{dtype.replace("_", " ").title()}</div>
            <div style="font-size:0.85em;color:#555;">
                Houses: {d.get("houses_with_device", 0)}/{n}<br>
                Sessions/house: median {count_dist.get("median", 0):.0f}<br>
                Avg magnitude: {mag_dist.get("median", 0):.0f}W
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
    <section>
        <h2>Population Statistics</h2>
        <p style="color:#666;margin-bottom:12px;font-size:0.85em;">
            Cross-house analysis of device identification patterns.
            Z-scores flag houses with unusual device characteristics.
        </p>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
            {device_cards}
        </div>
        {outlier_html}
    </section>'''


def _build_empty_aggregate_html(generated_at: str, experiment_dir: str) -> str:
    """Fallback HTML when no valid data found."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Device Identification - Aggregate Report</title>
</head>
<body style="font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px;">
    <h1>Device Identification - Aggregate Report</h1>
    <p>Generated: {generated_at}</p>
    <p style="color: #888;">No device session data found in: {experiment_dir}</p>
    <p>Ensure the pipeline was run with device identification enabled.</p>
</body>
</html>"""
