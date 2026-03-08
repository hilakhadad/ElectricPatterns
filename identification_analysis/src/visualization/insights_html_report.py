"""
Cross-house insights HTML report generator.

Produces a single interactive HTML file with Plotly charts and sortable tables,
following the ElectricPatterns design language (shared/html_utils.py).
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from shared.html_utils import (
    get_unified_css,
    build_collapsible_section,
    build_glossary_section,
    EP_HEADER_GRADIENT,
    EP_TEXT_PRIMARY,
    EP_TEXT_SECONDARY,
    EP_CARD_BG,
    EP_CARD_BORDER,
    EP_BG,
    EP_TABLE_HEADER_BG,
)

# Reuse device colors/names
from visualization.charts_device import DEVICE_COLORS, DEVICE_DISPLAY_NAMES

logger = logging.getLogger(__name__)

SEVERITY_COLORS = {
    'info': '#007bff',
    'warning': '#e67e22',
    'critical': '#dc3545',
    'success': '#28a745',
}

DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# ============================================================================
# Main entry point
# ============================================================================

def generate_insights_report(
    insights: Dict[str, Any],
    experiment_dir: str,
    output_path: Optional[str] = None,
    experiment_name: str = '',
    house_reports_dir: Optional[str] = None,
) -> str:
    """Generate cross-house insights HTML report.

    Args:
        insights: Output from compute_all_insights()
        experiment_dir: For title/metadata
        output_path: Where to save (if None, auto-generate path)
        experiment_name: Experiment name for header
        house_reports_dir: Path to per-house reports (for links)

    Returns:
        Path to saved HTML file.
    """
    n_houses = insights.get('n_houses', 0)
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Build sections
    top_findings_html = _build_top_findings(insights.get('top_findings', []))
    discovery_html = _build_device_discovery_section(insights.get('device_discovery', {}),
                                                     n_houses, house_reports_dir)
    temporal_html = _build_temporal_section(insights.get('temporal_patterns', {}))
    classification_html = _build_classification_section(
        insights.get('classification_quality', {}), house_reports_dir)
    segregation_html = _build_segregation_section(
        insights.get('segregation_effectiveness'), house_reports_dir)
    anomalies_html = _build_anomalies_section(
        insights.get('anomalies', {}), house_reports_dir)
    signatures_html = _build_signatures_section(insights.get('device_signatures', {}))
    clusters_html = _build_clusters_section(
        insights.get('house_clusters', {}), house_reports_dir)

    # Build CSS: get_unified_css() returns double-braced CSS for f-string embedding,
    # so we construct the extra CSS separately and concatenate.
    extra_css = f"""
        .findings-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 16px;
            margin-bottom: 30px;
        }}

        .finding-card {{
            padding: 20px;
            border-radius: 12px;
            border-left: 4px solid;
            background: {EP_BG};
            border-color: {EP_CARD_BORDER};
        }}

        .finding-title {{
            font-weight: 700;
            font-size: 1.1em;
            margin-bottom: 6px;
        }}

        .finding-desc {{
            font-size: 0.88em;
            color: {EP_TEXT_SECONDARY};
        }}

        .insight-note {{
            font-size: 0.85em;
            color: {EP_TEXT_SECONDARY};
            font-style: italic;
            margin: 10px 0;
        }}
    """

    # get_unified_css() is an f-string with {{{{ escaping → returns {{ in output.
    # We embed it via f-string so the {{ resolves to { in final HTML.
    full_css = f"""{get_unified_css()}{extra_css}"""

    title = experiment_name or 'ElectricPatterns'
    subtitle = experiment_name or 'ElectricPatterns Population Analysis'
    glossary = build_glossary_section()
    sort_js = _build_sort_js()

    # Use %-formatting for the HTML shell to avoid brace conflicts
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-House Insights — %(title)s</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>%(css)s</style>
</head>
<body>
<div class="container">

    <header>
        <h1>Cross-House Insights</h1>
        <div class="subtitle">%(subtitle)s</div>
        <div class="info-bar">
            <span class="info-item"><strong>%(n_houses)s</strong> houses analyzed</span>
            <span class="info-item">Generated: %(now)s</span>
        </div>
    </header>

    %(top_findings)s

    %(discovery)s

    %(temporal)s

    %(classification)s

    %(segregation)s

    %(anomalies)s

    %(signatures)s

    %(clusters)s

    %(glossary)s

    <footer>ElectricPatterns &mdash; Cross-House Insights Report</footer>
</div>

%(sort_js)s

</body>
</html>""" % {
        'title': title,
        'css': full_css,
        'subtitle': subtitle,
        'n_houses': n_houses,
        'now': now,
        'top_findings': top_findings_html,
        'discovery': discovery_html,
        'temporal': temporal_html,
        'classification': classification_html,
        'segregation': segregation_html,
        'anomalies': anomalies_html,
        'signatures': signatures_html,
        'clusters': clusters_html,
        'glossary': glossary,
        'sort_js': sort_js,
    }

    # Save
    if output_path is None:
        out_dir = Path(__file__).parent.parent.parent / 'OUTPUT'
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = str(out_dir / f'cross_house_insights_{ts}.html')

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f'Insights report saved to {output_path}')
    return output_path


# ============================================================================
# Top Findings Dashboard
# ============================================================================

def _build_top_findings(findings: List[Dict]) -> str:
    if not findings:
        return ''

    cards = ''
    for f in findings:
        color = SEVERITY_COLORS.get(f.get('severity', 'info'), '#007bff')
        cards += f'''
        <div class="finding-card" style="border-left-color: {color};">
            <div class="finding-title" style="color: {color};">{f['title']}</div>
            <div class="finding-desc">{f['description']}</div>
        </div>'''

    return f'''
    <section>
        <h2>Key Findings</h2>
        <div class="findings-grid">{cards}</div>
    </section>'''


# ============================================================================
# A. Device Discovery
# ============================================================================

def _build_device_discovery_section(dd: Dict, n_houses: int,
                                     house_reports_dir: Optional[str]) -> str:
    prevalence = dd.get('prevalence', {})

    # Summary cards
    cards = ''
    for dt in ['boiler', 'central_ac', 'regular_ac', 'recurring_pattern', 'unknown']:
        info = prevalence.get(dt, {})
        count = info.get('count', 0)
        pct = info.get('pct', 0)
        color = DEVICE_COLORS.get(dt, '#6c757d')
        name = DEVICE_DISPLAY_NAMES.get(dt, dt)
        cards += f'''
        <div class="summary-card">
            <div class="summary-number" style="color:{color};">{count}</div>
            <div class="summary-label">{name}<br>({pct:.0f}% of houses)</div>
        </div>'''

    # Prevalence chart
    chart_id = 'chart-prevalence'
    device_types = [dt for dt in ['boiler', 'three_phase_device', 'central_ac',
                                   'regular_ac', 'recurring_pattern']
                    if dt in prevalence]
    chart_counts = [prevalence[dt]['count'] for dt in device_types]
    chart_labels = [DEVICE_DISPLAY_NAMES.get(dt, dt) for dt in device_types]
    chart_colors = [DEVICE_COLORS.get(dt, '#999') for dt in device_types]

    prevalence_chart = f'''
    <div id="{chart_id}" style="min-height:300px;"></div>
    <script>
    Plotly.newPlot('{chart_id}', [{{
        y: {json.dumps(chart_labels)},
        x: {json.dumps(chart_counts)},
        type: 'bar',
        orientation: 'h',
        marker: {{ color: {json.dumps(chart_colors)} }},
        text: {json.dumps([f'{c}/{n_houses}' for c in chart_counts])},
        textposition: 'auto',
    }}], {{
        title: 'Device Type Prevalence (houses with at least 1 session)',
        xaxis: {{ title: 'Number of Houses' }},
        margin: {{ l: 220, r: 30, t: 50, b: 50 }},
        height: 300,
    }}, {{responsive: true}});
    </script>'''

    # Missing devices tables
    missing_boiler = dd.get('missing_boiler', [])
    missing_ac = dd.get('missing_ac', [])

    missing_html = ''
    if missing_boiler:
        rows = ''.join(f'<tr><td>{_house_link(h, house_reports_dir)}</td></tr>'
                       for h in missing_boiler[:30])
        extra = f'<p class="insight-note">...and {len(missing_boiler) - 30} more</p>' if len(missing_boiler) > 30 else ''
        missing_html += f'''
        <h3>Houses Without Boiler ({len(missing_boiler)})</h3>
        <table class="data-table"><thead><tr><th>House ID</th></tr></thead>
        <tbody>{rows}</tbody></table>{extra}'''

    if missing_ac:
        rows = ''.join(f'<tr><td>{_house_link(h, house_reports_dir)}</td></tr>'
                       for h in missing_ac[:30])
        extra = f'<p class="insight-note">...and {len(missing_ac) - 30} more</p>' if len(missing_ac) > 30 else ''
        missing_html += f'''
        <h3>Houses Without AC ({len(missing_ac)})</h3>
        <table class="data-table"><thead><tr><th>House ID</th></tr></thead>
        <tbody>{rows}</tbody></table>{extra}'''

    # Session count distribution chart
    counts = dd.get('session_counts', {})
    count_dist = dd.get('session_count_distribution', {})
    dist_chart = ''
    if counts:
        vals = list(counts.values())
        dist_chart = f'''
        <div id="chart-session-dist" style="min-height:300px;margin-top:20px;"></div>
        <script>
        Plotly.newPlot('chart-session-dist', [{{
            x: {json.dumps(vals)},
            type: 'histogram',
            nbinsx: 30,
            marker: {{ color: '#7B9BC4' }},
        }}], {{
            title: 'Session Count Distribution Across Houses (median={count_dist.get("median", "?")})',
            xaxis: {{ title: 'Total Sessions per House' }},
            yaxis: {{ title: 'Number of Houses' }},
            margin: {{ l: 60, r: 30, t: 50, b: 50 }},
            height: 300,
        }}, {{responsive: true}});
        </script>'''

    body = f'''
    <div class="summary-grid">{cards}</div>
    {prevalence_chart}
    {dist_chart}
    <div class="charts-grid" style="margin-top:20px;">{missing_html}</div>'''

    return f'''
    <section>
        <h2>Device Discovery Overview</h2>
        {body}
    </section>'''


# ============================================================================
# B. Temporal Patterns
# ============================================================================

def _build_temporal_section(tp: Dict) -> str:
    hourly = tp.get('hourly', {})
    weekday = tp.get('weekday', {})
    monthly = tp.get('monthly', {})
    peak_hours = tp.get('peak_hours', {})

    # Heatmap: device types (rows) x hours (columns)
    device_types = [dt for dt in ['boiler', 'regular_ac', 'central_ac', 'recurring_pattern']
                    if dt in hourly and sum(hourly[dt]) > 0]
    z_data = [hourly[dt] for dt in device_types]
    y_labels = [DEVICE_DISPLAY_NAMES.get(dt, dt) for dt in device_types]
    x_labels = [f'{h:02d}:00' for h in range(24)]

    heatmap = f'''
    <div id="chart-hourly-heatmap" style="min-height:350px;"></div>
    <script>
    Plotly.newPlot('chart-hourly-heatmap', [{{
        z: {json.dumps(z_data)},
        x: {json.dumps(x_labels)},
        y: {json.dumps(y_labels)},
        type: 'heatmap',
        colorscale: 'YlOrRd',
        hovertemplate: '%{{y}}<br>Hour: %{{x}}<br>Count: %{{z}}<extra></extra>',
    }}], {{
        title: 'Activation Count by Hour of Day (all houses combined)',
        margin: {{ l: 220, r: 30, t: 50, b: 50 }},
        height: 350,
    }}, {{responsive: true}});
    </script>'''

    # Weekday bar chart
    weekday_traces = []
    for dt in device_types:
        weekday_traces.append(f'''{{
            x: {json.dumps(DAY_NAMES)},
            y: {json.dumps(weekday[dt])},
            name: '{DEVICE_DISPLAY_NAMES.get(dt, dt)}',
            type: 'bar',
            marker: {{ color: '{DEVICE_COLORS.get(dt, "#999")}' }},
        }}''')

    weekday_chart = f'''
    <div id="chart-weekday" style="min-height:300px;"></div>
    <script>
    Plotly.newPlot('chart-weekday', [{','.join(weekday_traces)}], {{
        title: 'Activation Count by Day of Week',
        barmode: 'group',
        xaxis: {{ title: 'Day' }},
        yaxis: {{ title: 'Sessions' }},
        margin: {{ l: 60, r: 30, t: 50, b: 50 }},
        height: 300,
        legend: {{ orientation: 'h', y: -0.2 }},
    }}, {{responsive: true}});
    </script>'''

    # Monthly line chart
    monthly_traces = []
    for dt in device_types:
        monthly_traces.append(f'''{{
            x: {json.dumps(MONTH_NAMES)},
            y: {json.dumps(monthly[dt])},
            name: '{DEVICE_DISPLAY_NAMES.get(dt, dt)}',
            type: 'scatter',
            mode: 'lines+markers',
            line: {{ color: '{DEVICE_COLORS.get(dt, "#999")}' }},
        }}''')

    monthly_chart = f'''
    <div id="chart-monthly" style="min-height:300px;"></div>
    <script>
    Plotly.newPlot('chart-monthly', [{','.join(monthly_traces)}], {{
        title: 'Activation Count by Month (seasonal pattern)',
        xaxis: {{ title: 'Month' }},
        yaxis: {{ title: 'Sessions' }},
        margin: {{ l: 60, r: 30, t: 50, b: 50 }},
        height: 300,
        legend: {{ orientation: 'h', y: -0.2 }},
    }}, {{responsive: true}});
    </script>'''

    # Boiler regularity table
    reg_table = ''
    most_regular = tp.get('most_regular_houses', [])
    most_irregular = tp.get('most_irregular_houses', [])
    if most_regular:
        rows_reg = ''.join(
            f'<tr><td>{r["house_id"]}</td><td>{r["std_hours"]:.1f}h</td>'
            f'<td>{r["n_activations"]}</td></tr>'
            for r in most_regular[:10])
        rows_irr = ''.join(
            f'<tr><td>{r["house_id"]}</td><td>{r["std_hours"]:.1f}h</td>'
            f'<td>{r["n_activations"]}</td></tr>'
            for r in most_irregular[:10])
        reg_table = f'''
        <div class="charts-grid" style="margin-top:20px;">
            <div>
                <h3>Most Regular Boiler Users</h3>
                <p class="insight-note">Low std = activations at consistent times</p>
                <table class="data-table">
                    <thead><tr><th>House</th><th>Std (hours)</th><th>Activations</th></tr></thead>
                    <tbody>{rows_reg}</tbody>
                </table>
            </div>
            <div>
                <h3>Most Irregular Boiler Users</h3>
                <p class="insight-note">High std = activations spread across all hours</p>
                <table class="data-table">
                    <thead><tr><th>House</th><th>Std (hours)</th><th>Activations</th></tr></thead>
                    <tbody>{rows_irr}</tbody>
                </table>
            </div>
        </div>'''

    body = f'''
    {heatmap}
    <div class="charts-grid" style="margin-top:20px;">
        <div>{weekday_chart}</div>
        <div>{monthly_chart}</div>
    </div>
    {reg_table}'''

    return build_collapsible_section('Temporal Patterns', body)


# ============================================================================
# C. Classification Quality
# ============================================================================

def _build_classification_section(cq: Dict, house_reports_dir: Optional[str]) -> str:
    # Confidence histogram
    hist = cq.get('confidence_histogram', {})
    hist_chart = ''
    if hist.get('counts'):
        edges = hist['edges']
        bin_labels = [f'{edges[i]:.1f}-{edges[i+1]:.1f}' for i in range(len(edges) - 1)]
        hist_chart = f'''
        <div id="chart-conf-hist" style="min-height:300px;"></div>
        <script>
        Plotly.newPlot('chart-conf-hist', [{{
            x: {json.dumps(bin_labels)},
            y: {json.dumps(hist['counts'])},
            type: 'bar',
            marker: {{ color: '#7B9BC4' }},
        }}], {{
            title: 'Confidence Score Distribution (all sessions, all houses)',
            xaxis: {{ title: 'Confidence Range' }},
            yaxis: {{ title: 'Number of Sessions' }},
            margin: {{ l: 60, r: 30, t: 50, b: 50 }},
            height: 300,
        }}, {{responsive: true}});
        </script>'''

    # Per-device average confidence bar
    per_device = cq.get('per_device_avg_confidence', {})
    dt_list = [dt for dt in ['boiler', 'three_phase_device', 'central_ac',
                              'regular_ac', 'recurring_pattern']
               if dt in per_device and per_device[dt] is not None]
    device_conf_chart = ''
    if dt_list:
        device_conf_chart = f'''
        <div id="chart-device-conf" style="min-height:300px;"></div>
        <script>
        Plotly.newPlot('chart-device-conf', [{{
            y: {json.dumps([DEVICE_DISPLAY_NAMES.get(dt, dt) for dt in dt_list])},
            x: {json.dumps([per_device[dt] for dt in dt_list])},
            type: 'bar',
            orientation: 'h',
            marker: {{ color: {json.dumps([DEVICE_COLORS.get(dt, '#999') for dt in dt_list])} }},
            text: {json.dumps([f'{per_device[dt]:.0%}' for dt in dt_list])},
            textposition: 'auto',
        }}], {{
            title: 'Average Confidence by Device Type',
            xaxis: {{ title: 'Avg Confidence', range: [0, 1] }},
            margin: {{ l: 220, r: 30, t: 50, b: 50 }},
            height: 300,
        }}, {{responsive: true}});
        </script>'''

    # Summary cards
    total = cq.get('total_sessions', 0)
    borderline = cq.get('borderline_total', 0)
    hardest = cq.get('hardest_to_classify', '—')
    hardest_name = DEVICE_DISPLAY_NAMES.get(hardest, hardest) if hardest else '—'

    summary = f'''
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-number">{total:,}</div>
            <div class="summary-label">Total Sessions</div>
        </div>
        <div class="summary-card">
            <div class="summary-number" style="color:#e67e22;">{borderline:,}</div>
            <div class="summary-label">Borderline (0.3-0.5)</div>
        </div>
        <div class="summary-card">
            <div class="summary-number" style="font-size:1.2em;">{hardest_name}</div>
            <div class="summary-label">Hardest to Classify</div>
        </div>
    </div>'''

    # Low confidence houses table
    low_conf = cq.get('low_confidence_houses', [])
    low_table = ''
    if low_conf:
        rows = ''.join(
            f'<tr><td>{_house_link(h["house_id"], house_reports_dir)}</td>'
            f'<td>{h["avg_confidence"]:.3f}</td></tr>'
            for h in low_conf)
        low_table = f'''
        <h3>Lowest Average Confidence Houses</h3>
        <table class="data-table">
            <thead><tr><th>House</th><th>Avg Confidence</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>'''

    # High unknown houses table
    high_unk = cq.get('high_unknown_houses', [])
    unk_table = ''
    if high_unk:
        rows = ''.join(
            f'<tr><td>{_house_link(h["house_id"], house_reports_dir)}</td>'
            f'<td>{h["unknown_pct"]:.1f}%</td></tr>'
            for h in high_unk)
        unk_table = f'''
        <h3>Highest Unknown Session Rate (&gt;50%)</h3>
        <p class="insight-note">Percentage of session-minutes classified as unknown</p>
        <table class="data-table">
            <thead><tr><th>House</th><th>Unknown %</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>'''

    body = f'''
    {summary}
    <div class="charts-grid">
        <div>{hist_chart}</div>
        <div>{device_conf_chart}</div>
    </div>
    <div class="charts-grid" style="margin-top:20px;">
        <div>{low_table}</div>
        <div>{unk_table}</div>
    </div>'''

    return build_collapsible_section('Classification Quality', body)


# ============================================================================
# D. Segregation Effectiveness
# ============================================================================

def _build_segregation_section(seg: Optional[Dict],
                                house_reports_dir: Optional[str]) -> str:
    if not seg:
        return build_collapsible_section(
            'Segregation Effectiveness',
            '<p class="insight-note">No segregation data available. '
            'Run without --no-segregation to include this section.</p>')

    dist = seg.get('distribution', {})
    per_house = seg.get('per_house', {})

    # Distribution histogram
    dist_chart = ''
    if per_house:
        vals = list(per_house.values())
        dist_chart = f'''
        <div id="chart-seg-dist" style="min-height:300px;"></div>
        <script>
        Plotly.newPlot('chart-seg-dist', [{{
            x: {json.dumps(vals)},
            type: 'histogram',
            nbinsx: 25,
            marker: {{ color: '#28a745' }},
        }}], {{
            title: 'Segregation Rate Distribution (median={dist.get("median", "?")}%)',
            xaxis: {{ title: 'Average Segregated Power (%)' }},
            yaxis: {{ title: 'Number of Houses' }},
            margin: {{ l: 60, r: 30, t: 50, b: 50 }},
            height: 300,
        }}, {{responsive: true}});
        </script>'''

    # Per-phase median bar
    per_phase = seg.get('per_phase_median', {})
    phase_chart = ''
    if per_phase:
        phases = sorted(per_phase.keys())
        phase_chart = f'''
        <div id="chart-seg-phase" style="min-height:280px;"></div>
        <script>
        Plotly.newPlot('chart-seg-phase', [{{
            x: {json.dumps(phases)},
            y: {json.dumps([per_phase[p] for p in phases])},
            type: 'bar',
            marker: {{ color: ['#007bff', '#e67e22', '#28a745'] }},
            text: {json.dumps([f'{per_phase[p]:.1f}%' for p in phases])},
            textposition: 'auto',
        }}], {{
            title: 'Median Segregation by Phase',
            yaxis: {{ title: 'Segregated Power (%)' }},
            margin: {{ l: 60, r: 30, t: 50, b: 50 }},
            height: 280,
        }}, {{responsive: true}});
        </script>'''

    # Top/bottom tables
    bottom = seg.get('bottom_houses', [])
    top = seg.get('top_houses', [])
    tables = ''
    if bottom:
        rows_b = ''.join(
            f'<tr><td>{_house_link(h["house_id"], house_reports_dir)}</td>'
            f'<td>{h["avg_pct"]:.1f}%</td></tr>'
            for h in bottom)
        rows_t = ''.join(
            f'<tr><td>{_house_link(h["house_id"], house_reports_dir)}</td>'
            f'<td>{h["avg_pct"]:.1f}%</td></tr>'
            for h in top)
        tables = f'''
        <div class="charts-grid" style="margin-top:20px;">
            <div>
                <h3>Lowest Segregation</h3>
                <table class="data-table">
                    <thead><tr><th>House</th><th>Avg Segregated %</th></tr></thead>
                    <tbody>{rows_b}</tbody>
                </table>
            </div>
            <div>
                <h3>Highest Segregation</h3>
                <table class="data-table">
                    <thead><tr><th>House</th><th>Avg Segregated %</th></tr></thead>
                    <tbody>{rows_t}</tbody>
                </table>
            </div>
        </div>'''

    # Phase imbalance
    imbalance = seg.get('phase_imbalance_houses', [])
    imbalance_html = ''
    if imbalance:
        rows = ''.join(
            f'<tr><td>{_house_link(h["house_id"], house_reports_dir)}</td>'
            f'<td>{h["max_diff"]:.1f}%</td>'
            f'<td>{_format_phases(h.get("phases", {}))}</td></tr>'
            for h in imbalance[:15])
        imbalance_html = f'''
        <h3 style="margin-top:20px;">Phase Imbalance (&gt;10% difference between phases)</h3>
        <table class="data-table">
            <thead><tr><th>House</th><th>Max Phase Diff</th><th>Per Phase</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>'''

    body = f'''
    <div class="charts-grid">
        <div>{dist_chart}</div>
        <div>{phase_chart}</div>
    </div>
    {tables}
    {imbalance_html}'''

    return build_collapsible_section('Segregation Effectiveness', body)


# ============================================================================
# E. Anomalies
# ============================================================================

def _build_anomalies_section(anom: Dict, house_reports_dir: Optional[str]) -> str:
    # Zero-session houses
    zero = anom.get('zero_session_houses', [])
    zero_html = ''
    if zero:
        items = ', '.join(_house_link(h, house_reports_dir) for h in zero)
        zero_html = f'''
        <div style="padding:15px;background:#FFF3CD;border-radius:10px;margin-bottom:20px;">
            <strong>{len(zero)} houses with zero sessions:</strong> {items}
        </div>'''

    # Unusual combinations
    unusual = anom.get('unusual_combinations', [])
    unusual_html = ''
    if unusual:
        rows = ''.join(
            f'<tr><td>{_house_link(u["house_id"], house_reports_dir)}</td>'
            f'<td>{u["description"]}</td></tr>'
            for u in unusual[:20])
        unusual_html = f'''
        <h3>Unusual Patterns</h3>
        <table class="data-table">
            <thead><tr><th>House</th><th>Description</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>'''

    # High spike houses
    spikes = anom.get('high_spike_houses', [])
    spike_html = ''
    if spikes:
        rows = ''.join(
            f'<tr><td>{_house_link(s["house_id"], house_reports_dir)}</td>'
            f'<td>{s["spike_count"]}</td></tr>'
            for s in spikes)
        spike_html = f'''
        <h3 style="margin-top:20px;">Highest Spike Count (top 5%)</h3>
        <p class="insight-note">Houses with unusually many transient events filtered out</p>
        <table class="data-table">
            <thead><tr><th>House</th><th>Spike Count</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>'''

    # Top outliers
    outliers = anom.get('top_outliers', [])
    outlier_html = ''
    if outliers:
        rows = ''.join(
            f'<tr><td>{_house_link(o["house_id"], house_reports_dir)}</td>'
            f'<td>{o["score"]:.1f}</td>'
            f'<td>{"<br>".join(o.get("reasons", []))}</td></tr>'
            for o in outliers)
        outlier_html = f'''
        <h3 style="margin-top:20px;">Top Multi-Dimension Outliers</h3>
        <p class="insight-note">Composite outlier score from session count, unknown rate, spikes, and segregation</p>
        <table class="data-table">
            <thead><tr><th>House</th><th>Score</th><th>Reasons</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>'''

    # Outlier scatter chart
    scatter_chart = ''
    if outliers:
        house_ids = [o['house_id'] for o in outliers]
        scores = [o['score'] for o in outliers]
        hover = [f"House {o['house_id']}: {'; '.join(o.get('reasons', []))}"
                 for o in outliers]
        scatter_chart = f'''
        <div id="chart-outlier-scatter" style="min-height:300px;margin-top:20px;"></div>
        <script>
        Plotly.newPlot('chart-outlier-scatter', [{{
            x: {json.dumps(house_ids)},
            y: {json.dumps(scores)},
            text: {json.dumps(hover)},
            type: 'bar',
            marker: {{ color: {json.dumps(scores)}, colorscale: 'YlOrRd' }},
            hoverinfo: 'text',
        }}], {{
            title: 'Composite Outlier Scores',
            xaxis: {{ title: 'House ID', type: 'category' }},
            yaxis: {{ title: 'Outlier Score' }},
            margin: {{ l: 60, r: 30, t: 50, b: 70 }},
            height: 300,
        }}, {{responsive: true}});
        </script>'''

    body = f'''
    {zero_html}
    {unusual_html}
    {spike_html}
    {scatter_chart}
    {outlier_html}'''

    return build_collapsible_section('Anomalies &amp; Outliers', body)


# ============================================================================
# F. Device Signatures
# ============================================================================

def _build_signatures_section(sigs: Dict) -> str:
    mag = sigs.get('magnitude_raw', {})
    dur = sigs.get('duration_raw', {})
    phase_pref = sigs.get('phase_preference', {})
    recurring = sigs.get('recurring_patterns', [])

    # Box plots for magnitude
    mag_traces = []
    for dt in ['boiler', 'three_phase_device', 'central_ac', 'regular_ac', 'recurring_pattern']:
        if dt in mag and mag[dt]:
            mag_traces.append(f'''{{
                y: {json.dumps(mag[dt][:5000])},
                name: '{DEVICE_DISPLAY_NAMES.get(dt, dt)}',
                type: 'box',
                marker: {{ color: '{DEVICE_COLORS.get(dt, "#999")}' }},
                boxpoints: false,
            }}''')

    mag_chart = ''
    if mag_traces:
        mag_chart = f'''
        <div id="chart-mag-box" style="min-height:400px;"></div>
        <script>
        Plotly.newPlot('chart-mag-box', [{','.join(mag_traces)}], {{
            title: 'Power Magnitude Distribution by Device Type',
            yaxis: {{ title: 'Watts (W)' }},
            margin: {{ l: 70, r: 30, t: 50, b: 50 }},
            height: 400,
            showlegend: false,
        }}, {{responsive: true}});
        </script>'''

    # Box plots for duration
    dur_traces = []
    for dt in ['boiler', 'three_phase_device', 'central_ac', 'regular_ac', 'recurring_pattern']:
        if dt in dur and dur[dt]:
            # Cap at reasonable values for visualization
            capped = [min(d, 500) for d in dur[dt][:5000]]
            dur_traces.append(f'''{{
                y: {json.dumps(capped)},
                name: '{DEVICE_DISPLAY_NAMES.get(dt, dt)}',
                type: 'box',
                marker: {{ color: '{DEVICE_COLORS.get(dt, "#999")}' }},
                boxpoints: false,
            }}''')

    dur_chart = ''
    if dur_traces:
        dur_chart = f'''
        <div id="chart-dur-box" style="min-height:400px;"></div>
        <script>
        Plotly.newPlot('chart-dur-box', [{','.join(dur_traces)}], {{
            title: 'Session Duration Distribution by Device Type',
            yaxis: {{ title: 'Minutes' }},
            margin: {{ l: 70, r: 30, t: 50, b: 50 }},
            height: 400,
            showlegend: false,
        }}, {{responsive: true}});
        </script>'''

    # Phase preference stacked bar
    phase_chart = ''
    if phase_pref:
        dt_list = [dt for dt in ['boiler', 'central_ac', 'regular_ac', 'recurring_pattern']
                    if dt in phase_pref]
        phases = ['w1', 'w2', 'w3', 'multi']
        phase_colors = {'w1': '#007bff', 'w2': '#e67e22', 'w3': '#28a745', 'multi': '#6f42c1'}
        traces = []
        for p in phases:
            vals = [phase_pref.get(dt, {}).get(p, 0) for dt in dt_list]
            if sum(vals) > 0:
                traces.append(f'''{{
                    x: {json.dumps([DEVICE_DISPLAY_NAMES.get(dt, dt) for dt in dt_list])},
                    y: {json.dumps(vals)},
                    name: '{p}',
                    type: 'bar',
                    marker: {{ color: '{phase_colors.get(p, "#999")}' }},
                }}''')
        if traces:
            phase_chart = f'''
            <div id="chart-phase-pref" style="min-height:350px;"></div>
            <script>
            Plotly.newPlot('chart-phase-pref', [{','.join(traces)}], {{
                title: 'Phase Preference by Device Type',
                barmode: 'stack',
                yaxis: {{ title: 'Session Count' }},
                margin: {{ l: 60, r: 30, t: 50, b: 100 }},
                height: 350,
                legend: {{ orientation: 'h', y: -0.25 }},
            }}, {{responsive: true}});
            </script>'''

    # Recurring patterns table
    rp_html = ''
    if recurring:
        rows = ''.join(
            f'<tr><td><strong>{r["name"]}</strong></td>'
            f'<td>{r["house_count"]}</td>'
            f'<td>{r["total_sessions"]}</td>'
            f'<td>{r["avg_magnitude"]:.0f}W</td>'
            f'<td>{r["avg_duration"]:.1f}m</td></tr>'
            for r in recurring[:20])
        rp_html = f'''
        <h3 style="margin-top:20px;">Cross-House Recurring Patterns</h3>
        <p class="insight-note">Patterns found in multiple houses (from cross-house matching)</p>
        <table class="data-table">
            <thead><tr><th>Pattern</th><th>Houses</th><th>Sessions</th>
                       <th>Avg Magnitude</th><th>Avg Duration</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>'''

    body = f'''
    <div class="charts-grid">
        <div>{mag_chart}</div>
        <div>{dur_chart}</div>
    </div>
    {phase_chart}
    {rp_html}'''

    return build_collapsible_section('Device Signatures', body)


# ============================================================================
# G. Research Insights — Behavioral Similarity & Outcome Divergence
# ============================================================================

def _build_clusters_section(hc: Dict, house_reports_dir: Optional[str]) -> str:
    error = hc.get('error')
    if error:
        return build_collapsible_section(
            'Research Insights',
            f'<p class="insight-note">{error}</p>')

    if not hc:
        return ''

    parts = []

    # ── G1. Success Predictors ────────────────────────────────────
    success = hc.get('success_analysis', {})
    parts.append(_build_success_predictors(success))

    # ── G2. Failure Patterns ──────────────────────────────────────
    failure = hc.get('failure_analysis', {})
    parts.append(_build_failure_patterns(failure, house_reports_dir))

    # ── G3. Divergent Pairs ───────────────────────────────────────
    divergent = hc.get('divergent_pairs', [])
    parts.append(_build_divergent_pairs(divergent, hc.get('profiles', {}),
                                         house_reports_dir))

    # ── G4. Full Comparison Matrix ────────────────────────────────
    matrix = hc.get('comparison_matrix', {})
    parts.append(_build_comparison_matrix_html(matrix, house_reports_dir))

    body = '\n'.join(p for p in parts if p)
    return build_collapsible_section('Research Insights — What Makes the Pipeline Succeed or Fail?', body)


def _build_success_predictors(success: Dict) -> str:
    corrs = success.get('correlations', [])
    if not corrs:
        return ''

    # Correlation bar chart
    labels = [c['label'] for c in corrs]
    values = [c['correlation'] for c in corrs]
    colors = ['#28a745' if v > 0 else '#dc3545' for v in values]

    chart = f'''
    <div id="chart-success-corr" style="min-height:300px;"></div>
    <script>
    Plotly.newPlot('chart-success-corr', [{{
        y: {json.dumps(labels)},
        x: {json.dumps(values)},
        type: 'bar',
        orientation: 'h',
        marker: {{ color: {json.dumps(colors)} }},
        text: {json.dumps([f'r={v:.2f}' for v in values])},
        textposition: 'auto',
    }}], {{
        title: 'Correlation with Pipeline Success Score',
        xaxis: {{ title: 'Pearson r', range: [-1, 1] }},
        margin: {{ l: 200, r: 30, t: 50, b: 50 }},
        height: 300,
        shapes: [{{ type: 'line', x0: 0, x1: 0, y0: -0.5,
                    y1: {len(labels) - 0.5},
                    line: {{ color: '#999', dash: 'dash' }} }}],
    }}, {{responsive: true}});
    </script>'''

    # Interpretations
    interp_rows = ''.join(
        f'<tr><td><strong>{c["label"]}</strong></td>'
        f'<td style="color:{"#28a745" if c["correlation"] > 0 else "#dc3545"};">'
        f'{c["correlation"]:+.2f}</td>'
        f'<td>{c["interpretation"]}</td></tr>'
        for c in corrs)

    interp_table = f'''
    <table class="data-table" style="margin-top:15px;">
        <thead><tr><th>Feature</th><th>r</th><th>Interpretation</th></tr></thead>
        <tbody>{interp_rows}</tbody>
    </table>'''

    # Tier comparison
    tier = success.get('tier_comparison', {})
    tier_rows = ''
    for feat, info in tier.items():
        diff = info.get('difference_pct', 0)
        color = '#28a745' if diff > 0 else ('#dc3545' if diff < 0 else '#999')
        tier_rows += (f'<tr><td>{info["label"]}</td>'
                      f'<td>{info["low_tier_mean"]:.1f}</td>'
                      f'<td>{info["high_tier_mean"]:.1f}</td>'
                      f'<td style="color:{color};">{diff:+.1f}%</td></tr>')

    tier_table = ''
    if tier_rows:
        low_info = success.get('low_tier', {})
        high_info = success.get('high_tier', {})
        tier_table = f'''
        <h3 style="margin-top:20px;">Bottom vs Top Tier Comparison</h3>
        <p class="insight-note">
            Bottom tier ({len(low_info.get("house_ids", []))} houses,
            avg score {low_info.get("avg_success", 0):.2f}) vs
            Top tier ({len(high_info.get("house_ids", []))} houses,
            avg score {high_info.get("avg_success", 0):.2f})
        </p>
        <table class="data-table">
            <thead><tr><th>Feature</th><th>Bottom Tier Mean</th>
                       <th>Top Tier Mean</th><th>Difference</th></tr></thead>
            <tbody>{tier_rows}</tbody>
        </table>'''

    return f'''
    <h3>What Predicts Pipeline Success?</h3>
    <p class="insight-note">
        Correlation between house characteristics and a composite success score
        (classification rate + confidence + segregation + inverse unknown rate).
        Positive = more of this feature → better results.
    </p>
    {chart}
    {interp_table}
    {tier_table}'''


def _build_failure_patterns(failure: Dict, house_reports_dir: Optional[str]) -> str:
    modes = failure.get('failure_modes', {})
    if not modes:
        return ''

    n_struggling = failure.get('n_struggling', 0)
    n_healthy = failure.get('n_healthy', 0)

    # Summary cards
    cards = f'''
    <div class="summary-grid">
        <div class="summary-card" style="border-left:4px solid #28a745;">
            <div class="summary-number" style="color:#28a745;">{n_healthy}</div>
            <div class="summary-label">Healthy Houses</div>
        </div>
        <div class="summary-card" style="border-left:4px solid #dc3545;">
            <div class="summary-number" style="color:#dc3545;">{n_struggling}</div>
            <div class="summary-label">Struggling Houses</div>
        </div>
    </div>'''

    # Failure modes breakdown
    mode_rows = ''
    for key, mode in modes.items():
        houses = mode.get('houses', [])
        if not houses:
            continue
        avg = mode.get('avg_profile', {})
        house_links = ', '.join(_house_link(h, house_reports_dir) for h in sorted(houses)[:15])
        extra = f' +{len(houses) - 15} more' if len(houses) > 15 else ''
        mode_rows += f'''
        <div style="margin-bottom:15px;padding:15px;border-radius:10px;
                    border:1px solid {EP_CARD_BORDER};border-left:4px solid #dc3545;">
            <strong>{mode["label"]}</strong>
            <span style="color:{EP_TEXT_SECONDARY};font-size:0.85em;">
                — {len(houses)} houses</span>
            <p style="font-size:0.85em;margin:5px 0;">{mode["description"]}</p>
            <p style="font-size:0.82em;color:{EP_TEXT_SECONDARY};">
                {house_links}{extra}
            </p>
        </div>'''

    # Compound failures
    compound = failure.get('compound_failures', [])
    compound_html = ''
    if compound:
        rows = ''.join(
            f'<tr><td>{_house_link(c["house_id"], house_reports_dir)}</td>'
            f'<td>{c["n_modes"]}</td>'
            f'<td>{"<br>".join(c["failure_modes"])}</td></tr>'
            for c in compound[:15])
        compound_html = f'''
        <h4 style="margin-top:15px;">Compound Failures (multiple issues)</h4>
        <table class="data-table">
            <thead><tr><th>House</th><th># Issues</th><th>Failure Modes</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>'''

    return f'''
    <h3 style="margin-top:30px;">Failure Mode Analysis</h3>
    <p class="insight-note">
        Houses are flagged for specific failure modes. A house can appear in
        multiple categories. Houses with zero flags are "healthy".
    </p>
    {cards}
    {mode_rows}
    {compound_html}'''


def _build_divergent_pairs(pairs: List, profiles: Dict,
                            house_reports_dir: Optional[str]) -> str:
    if not pairs:
        return ''

    # Table of most interesting divergent pairs
    rows = ''
    for p in pairs[:10]:
        better = p['better']
        worse = p['worse']
        b_prof = profiles.get(better, {})
        w_prof = profiles.get(worse, {})
        rows += (
            f'<tr>'
            f'<td>{_house_link(better, house_reports_dir)}</td>'
            f'<td>{_house_link(worse, house_reports_dir)}</td>'
            f'<td>{p["behavioral_similarity"]:.2f}</td>'
            f'<td style="color:#28a745;">{p["score_a"]:.2f}</td>'
            f'<td style="color:#dc3545;">{p["score_b"]:.2f}</td>'
            f'<td>{b_prof.get("classified_pct", 0):.0f}% vs {w_prof.get("classified_pct", 0):.0f}%</td>'
            f'<td>{b_prof.get("segregation_pct", 0):.1f}% vs {w_prof.get("segregation_pct", 0):.1f}%</td>'
            f'</tr>')

    return f'''
    <h3 style="margin-top:30px;">Similar Behavior, Different Outcomes</h3>
    <p class="insight-note">
        House pairs with similar input characteristics (session count, magnitude,
        spike ratio) but different pipeline performance. These are the most
        interesting cases for investigating why the pipeline succeeds in one house
        but not another with similar data.
    </p>
    <table class="data-table">
        <thead>
            <tr>
                <th>Better House</th><th>Worse House</th>
                <th>Similarity</th>
                <th>Score (better)</th><th>Score (worse)</th>
                <th>Classification</th><th>Segregation</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>'''


def _build_comparison_matrix_html(matrix: Dict,
                                    house_reports_dir: Optional[str]) -> str:
    rows_data = matrix.get('rows', [])
    if not rows_data:
        return ''

    tier_colors = {'high': '#28a745', 'mid': '#e67e22', 'low': '#dc3545'}
    tier_labels = {'high': 'Top', 'mid': 'Mid', 'low': 'Bottom'}

    rows = ''
    for r in rows_data:
        tc = tier_colors.get(r['tier'], '#999')
        tl = tier_labels.get(r['tier'], '?')
        rows += (
            f'<tr>'
            f'<td>{_house_link(r["house_id"], house_reports_dir)}</td>'
            f'<td><span class="badge" style="background:{tc}20;color:{tc};">{tl}</span></td>'
            f'<td>{r["success_score"]:.3f}</td>'
            f'<td>{r["total_sessions"]}</td>'
            f'<td>{r["avg_magnitude"]:.0f}W</td>'
            f'<td>{r["spike_ratio"]:.0%}</td>'
            f'<td>{r["boiler"]}</td><td>{r["ac"]}</td>'
            f'<td>{r["recurring"]}</td><td>{r["unknown"]}</td>'
            f'<td>{r["classified_pct"]:.0f}%</td>'
            f'<td>{r["segregation_pct"]:.1f}%</td>'
            f'<td>{r["avg_confidence"]:.0%}</td>'
            f'</tr>')

    # Success score scatter chart
    house_ids = [r['house_id'] for r in rows_data]
    scores = [r['success_score'] for r in rows_data]
    seg_pcts = [r['segregation_pct'] for r in rows_data]
    sizes = [max(5, r['total_sessions'] / 10) for r in rows_data]
    colors = [tier_colors.get(r['tier'], '#999') for r in rows_data]
    hover = [f"House {r['house_id']}: score={r['success_score']:.2f}, "
             f"seg={r['segregation_pct']:.1f}%, "
             f"classified={r['classified_pct']:.0f}%"
             for r in rows_data]

    chart = f'''
    <div id="chart-success-scatter" style="min-height:400px;margin-bottom:20px;"></div>
    <script>
    Plotly.newPlot('chart-success-scatter', [{{
        x: {json.dumps(scores)},
        y: {json.dumps(seg_pcts)},
        text: {json.dumps(hover)},
        mode: 'markers+text',
        textposition: 'top center',
        textfont: {{ size: 9 }},
        marker: {{
            size: {json.dumps(sizes)},
            color: {json.dumps(colors)},
            line: {{ width: 1, color: '#fff' }},
        }},
        customdata: {json.dumps(house_ids)},
        hoverinfo: 'text',
    }}], {{
        title: 'Pipeline Success Score vs Segregation % (bubble size = session count)',
        xaxis: {{ title: 'Success Score' }},
        yaxis: {{ title: 'Segregation %' }},
        margin: {{ l: 60, r: 30, t: 50, b: 50 }},
        height: 400,
    }}, {{responsive: true}});
    </script>'''

    return f'''
    <h3 style="margin-top:30px;">Full House Comparison</h3>
    <p class="insight-note">
        Every house ranked by pipeline success score.
        Score = 35% classification rate + 25% confidence + 25% segregation + 15% inverse-unknown.
        Click column headers to sort.
    </p>
    {chart}
    <table class="data-table">
        <thead>
            <tr>
                <th>House</th><th>Tier</th><th>Score</th>
                <th>Sessions</th><th>Avg Mag</th><th>Spike %</th>
                <th>Boiler</th><th>AC</th><th>Recurring</th><th>Unknown</th>
                <th>Classified %</th><th>Segregation %</th><th>Confidence</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>'''


# ============================================================================
# Utilities
# ============================================================================

def _house_link(house_id: str, house_reports_dir: Optional[str] = None) -> str:
    """Create a clickable house link if report directory is available."""
    if house_reports_dir:
        path = f'{house_reports_dir}/identification_report_{house_id}.html'
        return f'<a href="{path}" class="house-link" target="_blank">{house_id}</a>'
    return house_id


def _format_phases(phases: Dict) -> str:
    """Format {phase: pct} dict for display."""
    parts = [f'{p}: {v:.1f}%' for p, v in sorted(phases.items())]
    return ', '.join(parts)


def _build_sort_js() -> str:
    """Return JavaScript for sortable tables (same pattern as aggregate report)."""
    return '''
<script>
(function() {
    document.querySelectorAll('.data-table th').forEach(function(th) {
        th.addEventListener('click', function() {
            var table = th.closest('table');
            var idx = Array.from(th.parentNode.children).indexOf(th);
            var tbody = table.querySelector('tbody');
            var rows = Array.from(tbody.querySelectorAll('tr'));
            var asc = th.dataset.sortDir !== 'asc';
            th.dataset.sortDir = asc ? 'asc' : 'desc';

            // Reset other headers
            th.parentNode.querySelectorAll('th').forEach(function(h) {
                if (h !== th) h.dataset.sortDir = '';
            });

            rows.sort(function(a, b) {
                var va = a.children[idx] ? a.children[idx].textContent.trim() : '';
                var vb = b.children[idx] ? b.children[idx].textContent.trim() : '';
                var na = parseFloat(va.replace(/[^\\d.\\-]/g, ''));
                var nb = parseFloat(vb.replace(/[^\\d.\\-]/g, ''));
                if (!isNaN(na) && !isNaN(nb)) {
                    return asc ? na - nb : nb - na;
                }
                return asc ? va.localeCompare(vb) : vb.localeCompare(va);
            });

            rows.forEach(function(r) { tbody.appendChild(r); });
        });
    });
})();
</script>'''
