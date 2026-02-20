"""
Chart generation for classification quality (Module 2) sections.

Creates HTML sections with inline Plotly charts for:
  - Quality score gauge
  - Temporal consistency bars
  - Magnitude stability dot plot
  - Duration box plots
  - Seasonal pattern grouped bars
  - Confidence histogram
  - Flags summary table
"""
import json
from typing import Dict, Any, List

# Color constants (consistent with dynamic_report_charts.py)
GREEN = '#28a745'
ORANGE = '#e67e22'
RED = '#dc3545'
GRAY = '#6c757d'
PURPLE = '#6f42c1'
BLUE = '#007bff'

# Device type colors
DEVICE_COLORS = {
    'boiler': '#007bff',
    'central_ac': '#dc3545',
    'regular_ac': '#e67e22',
    'unclassified': '#6c757d',
}

# Confidence tier colors
TIER_COLORS = {
    'high': GREEN,
    'medium': ORANGE,
    'low': RED,
}

# Quality tier colors
QUALITY_TIER_COLORS = {
    'excellent': GREEN,
    'good': '#007bff',
    'fair': ORANGE,
    'poor': RED,
}


def create_quality_section(quality: Dict[str, Any], confidence: Dict[str, Any]) -> str:
    """
    Create the full Classification Quality section HTML.

    Args:
        quality: Output from calculate_classification_quality()
        confidence: Output from calculate_confidence_scores()

    Returns:
        HTML string for the entire classification quality section.
    """
    if not quality or quality.get('total_activations', 0) == 0:
        return ''

    parts = [
        '<section>',
        '<h2>Classification Quality</h2>',
        '<div style="color: #555; margin-bottom: 16px; font-size: 0.85em; line-height: 1.5;">',
        '<p style="margin:0 0 8px;">',
        'This section evaluates <strong>how consistent the automatic device classification is</strong>, ',
        'without relying on ground truth. Each detected ON\u2192OFF event was classified as a device type ',
        '(boiler, central AC, regular AC, or unclassified) based on power, duration, and phase patterns. ',
        'The metrics below check whether activations within each category behave consistently.',
        '</p>',
        '<p style="margin:0;font-size:0.92em;color:#888;">',
        'Note: Each activation represents a single matched ON\u2192OFF pair (e.g., one compressor cycle for AC), ',
        'not a full usage session.',
        '</p>',
        '</div>',
    ]

    # Quality summary box
    parts.append(_create_quality_summary_box(quality, confidence))

    # Metrics grid (2x2)
    parts.append('<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0;">')
    parts.append(_create_temporal_chart(quality.get('metrics', {}).get('temporal_consistency', {})))
    parts.append(_create_magnitude_chart(quality.get('metrics', {}).get('magnitude_stability', {})))
    parts.append(_create_duration_chart(quality.get('metrics', {}).get('duration_plausibility', {})))
    parts.append(_create_seasonal_chart(quality.get('metrics', {}).get('seasonal_coherence', {})))
    parts.append('</div>')

    # Confidence histogram
    parts.append(_create_confidence_histogram(confidence))

    # Flags table
    flags = quality.get('flags', [])
    if flags:
        parts.append(_create_flags_table(flags))

    parts.append('</section>')
    return '\n'.join(parts)


def _create_quality_summary_box(quality: Dict, confidence: Dict) -> str:
    """Create quality score summary box with gauge."""
    score = quality.get('overall_quality_score', 0)
    tier = quality.get('quality_tier', 'poor')
    tier_color = QUALITY_TIER_COLORS.get(tier, GRAY)
    total = quality.get('total_activations', 0)
    flags_count = len(quality.get('flags', []))

    conf_summary = confidence.get('confidence_summary', {})
    mean_conf = conf_summary.get('mean', 0)
    high_count = conf_summary.get('high_count', 0)
    total_conf = confidence.get('total_activations', 0)
    high_rate = high_count / total_conf if total_conf > 0 else 0

    return f'''
    <div style="display:grid;grid-template-columns:auto 1fr;gap:20px;align-items:center;
                background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;margin-bottom:16px;">
        <div style="text-align:center;">
            <div style="font-size:2.5em;font-weight:bold;color:{tier_color};">{score:.0%}</div>
            <div style="font-size:0.9em;color:{tier_color};font-weight:600;text-transform:capitalize;">{tier}</div>
            <div style="font-size:0.75em;color:#999;margin-top:4px;">Quality Score</div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
            <div style="text-align:center;padding:8px;background:#f8f9fa;border-radius:6px;">
                <div style="font-size:1.4em;font-weight:bold;">{total}</div>
                <div style="font-size:0.75em;color:#666;">Activations</div>
            </div>
            <div style="text-align:center;padding:8px;background:#f8f9fa;border-radius:6px;">
                <div style="font-size:1.4em;font-weight:bold;">{mean_conf:.0%}</div>
                <div style="font-size:0.75em;color:#666;">Avg Confidence</div>
            </div>
            <div style="text-align:center;padding:8px;background:#f8f9fa;border-radius:6px;">
                <div style="font-size:1.4em;font-weight:bold;">{high_rate:.0%}</div>
                <div style="font-size:0.75em;color:#666;">High Confidence</div>
            </div>
            <div style="text-align:center;padding:8px;background:#f8f9fa;border-radius:6px;">
                <div style="font-size:1.4em;font-weight:bold;color:{"#dc3545" if flags_count > 2 else "#e67e22" if flags_count > 0 else "#28a745"}">{flags_count}</div>
                <div style="font-size:0.75em;color:#666;">Flags</div>
            </div>
        </div>
    </div>'''


def _create_temporal_chart(temporal: Dict) -> str:
    """Create temporal consistency horizontal bar chart."""
    if not temporal:
        return '<div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;"><h3 style="font-size:0.95em;margin-bottom:8px;">Temporal Consistency</h3><p style="color:#999;font-size:0.8em;">No data</p></div>'

    types = list(temporal.keys())
    values = [temporal[t].get('value', 0) for t in types]
    colors = [DEVICE_COLORS.get(t, GRAY) for t in types]
    flags = [temporal[t].get('flag', '') or '' for t in types]

    # Build bar chart with Plotly
    chart_id = 'temporal_chart'
    traces = json.dumps([{
        'type': 'bar',
        'y': types,
        'x': values,
        'orientation': 'h',
        'marker': {'color': colors},
        'text': [f'{v:.0%}' for v in values],
        'textposition': 'auto',
        'hovertemplate': '%{y}: %{x:.1%}<extra></extra>',
    }])
    layout = json.dumps({
        'margin': {'l': 80, 'r': 20, 't': 10, 'b': 30},
        'xaxis': {'range': [0, 1], 'tickformat': '.0%', 'title': ''},
        'yaxis': {'title': ''},
        'height': 180,
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#f8f9fa',
    })

    flag_html = ''
    for t, f in zip(types, flags):
        if f:
            flag_html += f'<div style="font-size:0.7em;color:#dc3545;">⚠ {t}: {f}</div>'

    return f'''
    <div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;">
        <h3 style="font-size:0.95em;margin-bottom:4px;">Temporal Consistency</h3>
        <p style="font-size:0.75em;color:#888;margin:0 0 8px;">
            What fraction of months contain at least one activation of this type.
            A device that appears consistently across months scores higher.
            Low values may indicate misclassification or seasonal-only usage.
        </p>
        <div id="{chart_id}"></div>
        <script>Plotly.newPlot('{chart_id}', {traces}, {layout}, {{displayModeBar:false}});</script>
        {flag_html}
    </div>'''


def _create_magnitude_chart(magnitude: Dict) -> str:
    """Create magnitude stability dot plot with CV annotations."""
    if not magnitude:
        return '<div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;"><h3 style="font-size:0.95em;margin-bottom:8px;">Magnitude Stability</h3><p style="color:#999;font-size:0.8em;">No data</p></div>'

    types = list(magnitude.keys())
    means = [magnitude[t].get('mean', 0) for t in types]
    stds = [magnitude[t].get('std', 0) for t in types]
    cvs = [magnitude[t].get('cv', 0) for t in types]
    colors = [DEVICE_COLORS.get(t, GRAY) for t in types]

    chart_id = 'magnitude_chart'
    traces = json.dumps([{
        'type': 'scatter',
        'x': types,
        'y': means,
        'error_y': {'type': 'data', 'array': stds, 'visible': True},
        'mode': 'markers+text',
        'marker': {'size': 12, 'color': colors},
        'text': [f'CV={cv:.2f}' for cv in cvs],
        'textposition': 'top center',
        'textfont': {'size': 10},
        'hovertemplate': '%{x}: %{y:.0f}W ± %{error_y.array:.0f}W<extra></extra>',
    }])
    layout = json.dumps({
        'margin': {'l': 60, 'r': 20, 't': 20, 'b': 40},
        'yaxis': {'title': 'Watts'},
        'height': 180,
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#f8f9fa',
    })

    return f'''
    <div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;">
        <h3 style="font-size:0.95em;margin-bottom:4px;">Magnitude Stability</h3>
        <p style="font-size:0.75em;color:#888;margin:0 0 8px;">
            Mean power (W) per device type, with standard deviation bars.
            CV (Coefficient of Variation) = std/mean &mdash; lower is better.
            A real device should have consistent power draw across activations (CV &lt; 0.3).
        </p>
        <div id="{chart_id}"></div>
        <script>Plotly.newPlot('{chart_id}', {traces}, {layout}, {{displayModeBar:false}});</script>
    </div>'''


def _create_duration_chart(duration: Dict) -> str:
    """Create duration distribution box plots."""
    if not duration:
        return '<div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;"><h3 style="font-size:0.95em;margin-bottom:8px;">Duration Plausibility</h3><p style="color:#999;font-size:0.8em;">No data</p></div>'

    # Build a summary bar chart with median + IQR
    types = list(duration.keys())
    medians = [duration[t].get('median', 0) for t in types]
    q25s = [duration[t].get('q25', 0) for t in types]
    q75s = [duration[t].get('q75', 0) for t in types]
    colors = [DEVICE_COLORS.get(t, GRAY) for t in types]

    chart_id = 'duration_chart'
    # Use bar for median with error bars for IQR
    traces = json.dumps([{
        'type': 'bar',
        'x': types,
        'y': medians,
        'error_y': {
            'type': 'data',
            'symmetric': False,
            'array': [q75 - med for q75, med in zip(q75s, medians)],
            'arrayminus': [med - q25 for med, q25 in zip(medians, q25s)],
            'visible': True,
        },
        'marker': {'color': colors},
        'text': [f'{m:.0f}min' for m in medians],
        'textposition': 'auto',
        'hovertemplate': '%{x}: median=%{y:.0f}min<extra></extra>',
    }])
    layout = json.dumps({
        'margin': {'l': 60, 'r': 20, 't': 10, 'b': 40},
        'yaxis': {'title': 'Minutes'},
        'height': 180,
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#f8f9fa',
    })

    return f'''
    <div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;">
        <h3 style="font-size:0.95em;margin-bottom:4px;">Duration Plausibility</h3>
        <p style="font-size:0.75em;color:#888;margin:0 0 8px;">
            Median activation duration per device type, with IQR (25th&ndash;75th percentile) bars.
            Expected ranges: boiler &ge;25 min, regular AC 3&ndash;30 min (compressor cycle), central AC 5&ndash;30 min.
            Values far outside these ranges suggest misclassification.
        </p>
        <div id="{chart_id}"></div>
        <script>Plotly.newPlot('{chart_id}', {traces}, {layout}, {{displayModeBar:false}});</script>
    </div>'''


def _create_seasonal_chart(seasonal: Dict) -> str:
    """Create seasonal pattern grouped bar chart."""
    if not seasonal:
        return '<div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;"><h3 style="font-size:0.95em;margin-bottom:8px;">Seasonal Coherence</h3><p style="color:#999;font-size:0.8em;">No data</p></div>'

    types = list(seasonal.keys())
    warm = [seasonal[t].get('warm_count', 0) for t in types]
    cool = [seasonal[t].get('cool_count', 0) for t in types]
    ratios = [seasonal[t].get('ratio', 0) for t in types]

    chart_id = 'seasonal_chart'
    traces = json.dumps([
        {
            'type': 'bar',
            'x': types,
            'y': warm,
            'name': 'Warm (May-Oct)',
            'marker': {'color': '#e67e22'},
            'hovertemplate': '%{x}: %{y} warm<extra></extra>',
        },
        {
            'type': 'bar',
            'x': types,
            'y': cool,
            'name': 'Cool (Nov-Apr)',
            'marker': {'color': '#007bff'},
            'hovertemplate': '%{x}: %{y} cool<extra></extra>',
        },
    ])
    layout = json.dumps({
        'margin': {'l': 50, 'r': 20, 't': 10, 'b': 40},
        'barmode': 'group',
        'yaxis': {'title': 'Count'},
        'legend': {'orientation': 'h', 'y': 1.15, 'x': 0.5, 'xanchor': 'center'},
        'height': 180,
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#f8f9fa',
        'annotations': [
            {'x': t, 'y': max(w, c) + max(w, c) * 0.1, 'text': f'{r:.1f}x',
             'showarrow': False, 'font': {'size': 9}}
            for t, w, c, r in zip(types, warm, cool, ratios)
        ],
    })

    return f'''
    <div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;">
        <h3 style="font-size:0.95em;margin-bottom:4px;">Seasonal Coherence</h3>
        <p style="font-size:0.75em;color:#888;margin:0 0 8px;">
            Activation count by season &mdash; warm (May&ndash;Oct) vs cool (Nov&ndash;Apr).
            The ratio annotation shows warm/cool. AC should peak in warm months (ratio &gt;1),
            boiler in cool months (ratio &lt;1). Reversed patterns indicate possible misclassification.
        </p>
        <div id="{chart_id}"></div>
        <script>Plotly.newPlot('{chart_id}', {traces}, {layout}, {{displayModeBar:false}});</script>
    </div>'''


def _create_confidence_histogram(confidence: Dict) -> str:
    """Create confidence score histogram stacked by device type."""
    activations = confidence.get('activations', [])
    if not activations:
        return ''

    # We need the original activations to get device_type; use index-based mapping
    # For now, just show overall histogram without device type stacking
    scores = [a.get('confidence', 0) for a in activations]
    tiers = [a.get('confidence_tier', 'low') for a in activations]

    # Color by tier
    colors = [TIER_COLORS.get(t, GRAY) for t in tiers]

    chart_id = 'confidence_hist'
    # Group by tier for stacked histogram
    high_scores = [s for s, t in zip(scores, tiers) if t == 'high']
    med_scores = [s for s, t in zip(scores, tiers) if t == 'medium']
    low_scores = [s for s, t in zip(scores, tiers) if t == 'low']

    traces = json.dumps([
        {'type': 'histogram', 'x': high_scores, 'name': f'High ({len(high_scores)})',
         'marker': {'color': GREEN}, 'xbins': {'start': 0, 'end': 1, 'size': 0.05},
         'opacity': 0.8},
        {'type': 'histogram', 'x': med_scores, 'name': f'Medium ({len(med_scores)})',
         'marker': {'color': ORANGE}, 'xbins': {'start': 0, 'end': 1, 'size': 0.05},
         'opacity': 0.8},
        {'type': 'histogram', 'x': low_scores, 'name': f'Low ({len(low_scores)})',
         'marker': {'color': RED}, 'xbins': {'start': 0, 'end': 1, 'size': 0.05},
         'opacity': 0.8},
    ])
    layout = json.dumps({
        'margin': {'l': 50, 'r': 20, 't': 20, 'b': 40},
        'barmode': 'stack',
        'xaxis': {'title': 'Confidence Score', 'range': [0, 1]},
        'yaxis': {'title': 'Count'},
        'legend': {'orientation': 'h', 'y': 1.12, 'x': 0.5, 'xanchor': 'center'},
        'height': 220,
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#f8f9fa',
        'shapes': [
            {'type': 'line', 'x0': 0.40, 'x1': 0.40, 'y0': 0, 'y1': 1, 'yref': 'paper',
             'line': {'dash': 'dash', 'color': '#999', 'width': 1}},
            {'type': 'line', 'x0': 0.70, 'x1': 0.70, 'y0': 0, 'y1': 1, 'yref': 'paper',
             'line': {'dash': 'dash', 'color': '#999', 'width': 1}},
        ],
    })

    return f'''
    <div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;margin:16px 0;">
        <h3 style="font-size:0.95em;margin-bottom:4px;">Confidence Distribution</h3>
        <p style="font-size:0.75em;color:#888;margin:0 0 8px;">
            Each activation receives a confidence score (0&ndash;1) based on magnitude consistency,
            duration fit, phase pattern, and match quality tag. Tiers: High (&ge;0.70), Medium (0.40&ndash;0.70),
            Low (&lt;0.40). Dashed lines mark tier boundaries. A healthy classification has most activations
            in the High tier.
        </p>
        <div id="{chart_id}"></div>
        <script>Plotly.newPlot('{chart_id}', {traces}, {layout}, {{displayModeBar:false}});</script>
    </div>'''


def _create_flags_table(flags: List[Dict]) -> str:
    """Create flags summary table."""
    rows = ''
    for f in flags:
        severity_color = RED if 'ANOMALY' in f.get('flag', '') or 'INVERTED' in f.get('flag', '') else ORANGE
        rows += f'''
        <tr>
            <td>{f.get('metric', '')}</td>
            <td><span style="color:{DEVICE_COLORS.get(f.get('device_type', ''), GRAY)};font-weight:600;">{f.get('device_type', '')}</span></td>
            <td><span style="color:{severity_color};font-weight:600;">{f.get('flag', '')}</span></td>
            <td style="font-size:0.85em;color:#666;">{f.get('detail', '')}</td>
        </tr>'''

    return f'''
    <div style="background:white;border:1px solid #dee2e6;border-radius:8px;padding:16px;margin:16px 0;">
        <h3 style="font-size:0.95em;margin-bottom:4px;">Quality Flags ({len(flags)})</h3>
        <p style="font-size:0.75em;color:#888;margin:0 0 8px;">
            Specific issues detected in the classification. Each flag points to a device type
            and metric that fell outside expected ranges, suggesting possible misclassification.
        </p>
        <table style="width:100%;border-collapse:collapse;font-size:0.85em;">
            <thead>
                <tr style="border-bottom:2px solid #dee2e6;text-align:left;">
                    <th style="padding:6px;">Metric</th>
                    <th style="padding:6px;">Device</th>
                    <th style="padding:6px;">Flag</th>
                    <th style="padding:6px;">Detail</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>'''
