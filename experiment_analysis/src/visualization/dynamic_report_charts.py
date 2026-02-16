"""
Chart generation for dynamic threshold experiment reports.

Creates 6 visualization sections, each returning an HTML string
with embedded Plotly charts or pure HTML/CSS.

Color scheme (no red):
  Green  = #28a745 (explained / success)
  Gray   = #6c757d (background / baseload)
  Orange = #fd7e14 (improvable / attention)
  Yellow = #ffc107 (medium efficiency)
"""
import json
from typing import Dict, Any, List

# Color constants
GREEN = '#28a745'
GRAY = '#6c757d'
ORANGE = '#fd7e14'
YELLOW = '#ffc107'
LIGHT_GREEN = '#d4edda'
LIGHT_GRAY = '#e9ecef'
LIGHT_ORANGE = '#fff3cd'


def create_summary_boxes(metrics: Dict[str, Any]) -> str:
    """
    Create 3 large summary number boxes (pure HTML/CSS).

    Shows: X% Explained | Y% Background | Z% Improvable
    Plus detection efficiency below.
    """
    totals = metrics.get('totals', {})
    explained_pct = totals.get('explained_pct', 0)
    background_pct = totals.get('background_pct', 0)
    improvable_pct = totals.get('improvable_pct', 0)
    efficiency = totals.get('efficiency', 0)

    explained_kwh = totals.get('explained_kwh', 0)
    background_kwh = totals.get('background_kwh', 0)
    improvable_kwh = totals.get('improvable_kwh', 0)

    # Efficiency color
    if efficiency >= 70:
        eff_color = GREEN
    elif efficiency >= 50:
        eff_color = YELLOW
    else:
        eff_color = ORANGE

    return f'''
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
        <div style="background: {LIGHT_GREEN}; border-left: 5px solid {GREEN}; border-radius: 10px; padding: 25px; text-align: center;">
            <div style="font-size: 2.8em; font-weight: bold; color: {GREEN};">{explained_pct:.1f}%</div>
            <div style="font-size: 1.1em; color: #155724; font-weight: 600; margin-top: 5px;">Explained</div>
            <div style="font-size: 0.85em; color: #666; margin-top: 3px;">{explained_kwh} kWh</div>
        </div>
        <div style="background: {LIGHT_GRAY}; border-left: 5px solid {GRAY}; border-radius: 10px; padding: 25px; text-align: center;">
            <div style="font-size: 2.8em; font-weight: bold; color: {GRAY};">{background_pct:.1f}%</div>
            <div style="font-size: 1.1em; color: #495057; font-weight: 600; margin-top: 5px;">Background</div>
            <div style="font-size: 0.85em; color: #666; margin-top: 3px;">{background_kwh} kWh</div>
        </div>
        <div style="background: {LIGHT_ORANGE}; border-left: 5px solid {ORANGE}; border-radius: 10px; padding: 25px; text-align: center;">
            <div style="font-size: 2.8em; font-weight: bold; color: {ORANGE};">{improvable_pct:.1f}%</div>
            <div style="font-size: 1.1em; color: #856404; font-weight: 600; margin-top: 5px;">Improvable</div>
            <div style="font-size: 0.85em; color: #666; margin-top: 3px;">{improvable_kwh} kWh</div>
        </div>
    </div>
    <div style="text-align: center; margin-bottom: 15px;">
        <span style="font-size: 1.1em; color: #444;">
            Detection Efficiency:
            <span style="font-weight: bold; color: {eff_color}; font-size: 1.3em;">{efficiency:.1f}%</span>
        </span>
        <span style="font-size: 0.85em; color: #888; margin-left: 10px;">
            (explained / targetable power, excluding background)
        </span>
    </div>
    '''


def create_power_breakdown_bar(metrics: Dict[str, Any]) -> str:
    """
    Create stacked horizontal bar chart showing power decomposition per phase.

    Each phase gets one bar: [Explained (green) | Background (gray) | Improvable (orange)]
    """
    phases = metrics.get('phases', {})
    chart_id = 'power-breakdown-chart'

    phase_labels = ['w1', 'w2', 'w3']
    explained_vals = []
    background_vals = []
    improvable_vals = []
    hover_texts_explained = []
    hover_texts_bg = []
    hover_texts_imp = []

    for p in phase_labels:
        ph = phases.get(p, {})
        explained_vals.append(ph.get('explained_pct', 0))
        background_vals.append(ph.get('background_pct', 0))
        improvable_vals.append(ph.get('improvable_pct', 0))
        hover_texts_explained.append(
            f"{p}: {ph.get('explained_kwh', 0)} kWh ({ph.get('explained_pct', 0):.1f}%)"
        )
        hover_texts_bg.append(
            f"{p}: {ph.get('background_kwh', 0)} kWh ({ph.get('background_pct', 0):.1f}%)"
        )
        hover_texts_imp.append(
            f"{p}: {ph.get('improvable_kwh', 0)} kWh ({ph.get('improvable_pct', 0):.1f}%)"
        )

    trace_explained = {
        'y': phase_labels,
        'x': explained_vals,
        'name': 'Explained',
        'type': 'bar',
        'orientation': 'h',
        'marker': {'color': GREEN},
        'text': [f'{v:.1f}%' for v in explained_vals],
        'textposition': 'inside',
        'hovertext': hover_texts_explained,
        'hoverinfo': 'text',
    }

    trace_background = {
        'y': phase_labels,
        'x': background_vals,
        'name': 'Background',
        'type': 'bar',
        'orientation': 'h',
        'marker': {'color': GRAY},
        'text': [f'{v:.1f}%' for v in background_vals],
        'textposition': 'inside',
        'hovertext': hover_texts_bg,
        'hoverinfo': 'text',
    }

    trace_improvable = {
        'y': phase_labels,
        'x': improvable_vals,
        'name': 'Improvable',
        'type': 'bar',
        'orientation': 'h',
        'marker': {'color': ORANGE},
        'text': [f'{v:.1f}%' for v in improvable_vals],
        'textposition': 'inside',
        'hovertext': hover_texts_imp,
        'hoverinfo': 'text',
    }

    layout = {
        'title': 'Power Decomposition by Phase',
        'barmode': 'stack',
        'xaxis': {'title': '% of Total Power', 'range': [0, 105]},
        'yaxis': {'title': ''},
        'legend': {'orientation': 'h', 'y': -0.2},
        'margin': {'l': 50, 'r': 30, 't': 50, 'b': 60},
        'height': 250,
    }

    data_json = json.dumps([trace_explained, trace_background, trace_improvable])

    return f'''
    <div id="{chart_id}" style="width:100%;height:250px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', {data_json}, {json.dumps(layout)});
    </script>
    '''


def create_efficiency_gauge(metrics: Dict[str, Any]) -> str:
    """
    Create donut chart (gauge) showing detection efficiency per phase.

    Green > 70%, Yellow 50-70%, Orange < 50%.
    """
    phases = metrics.get('phases', {})
    chart_id = 'efficiency-gauge-chart'

    charts_html = '<div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">'

    for i, phase in enumerate(['w1', 'w2', 'w3']):
        ph = phases.get(phase, {})
        eff = ph.get('efficiency', 0)
        remaining = max(0, 100 - eff)

        if eff >= 70:
            color = GREEN
        elif eff >= 50:
            color = YELLOW
        else:
            color = ORANGE

        phase_chart_id = f'{chart_id}-{phase}'

        data = {
            'values': [eff, remaining],
            'labels': ['Efficiency', ''],
            'type': 'pie',
            'hole': 0.7,
            'marker': {'colors': [color, '#f0f0f0']},
            'textinfo': 'none',
            'hoverinfo': 'skip',
            'sort': False,
        }

        layout = {
            'showlegend': False,
            'margin': {'l': 10, 'r': 10, 't': 30, 'b': 10},
            'height': 200,
            'width': 200,
            'annotations': [{
                'text': f'{eff:.0f}%',
                'showarrow': False,
                'font': {'size': 24, 'color': color, 'family': 'Arial Black'},
                'x': 0.5,
                'y': 0.5,
            }],
            'title': {'text': phase, 'font': {'size': 14}},
        }

        charts_html += f'''
        <div style="text-align: center;">
            <div id="{phase_chart_id}" style="width:200px;height:200px;display:inline-block;"></div>
            <div style="font-size: 0.85em; color: #666;">
                {ph.get('explained_kwh', 0)} / {round(ph.get('total_kwh', 0) - ph.get('background_kwh', 0), 2)} kWh
            </div>
            <script>
                Plotly.newPlot('{phase_chart_id}', [{json.dumps(data)}], {json.dumps(layout)},
                    {{displayModeBar: false}});
            </script>
        </div>
        '''

    charts_html += '</div>'
    return charts_html


def create_threshold_waterfall(metrics: Dict[str, Any]) -> str:
    """
    Create waterfall chart showing per-threshold contributions.

    TH=2000 -> TH=1500 -> TH=1100 -> TH=800 -> Total
    """
    per_threshold = metrics.get('per_threshold', [])
    chart_id = 'threshold-waterfall-chart'

    if not per_threshold:
        return '<p style="color: #888;">No per-threshold data available</p>'

    x_labels = [f'TH={t["threshold"]}W' for t in per_threshold]
    x_labels.append('Total')

    values = [t.get('explained_pct', 0) for t in per_threshold]
    total = sum(values)

    # Waterfall: each bar is relative, total is absolute
    measures = ['relative'] * len(per_threshold) + ['total']
    y_values = values + [total]

    text_labels = [f'+{v:.1f}%' for v in values] + [f'{total:.1f}%']

    data = {
        'type': 'waterfall',
        'x': x_labels,
        'y': y_values,
        'measure': measures,
        'text': text_labels,
        'textposition': 'outside',
        'connector': {'line': {'color': '#ccc'}},
        'increasing': {'marker': {'color': GREEN}},
        'totals': {'marker': {'color': '#667eea'}},
    }

    layout = {
        'title': 'Contribution by Threshold Level',
        'yaxis': {'title': '% of Total Power Explained'},
        'xaxis': {'title': ''},
        'showlegend': False,
        'margin': {'t': 50, 'b': 50},
        'height': 350,
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:350px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_remaining_analysis_chart(metrics: Dict[str, Any]) -> str:
    """
    Create pie/donut chart showing classification of remaining power.

    Categories: Noise | Small Events | Large Unmatched
    """
    rc = metrics.get('remaining_classification', {})
    chart_id = 'remaining-analysis-chart'

    noise = rc.get('noise_minutes', 0)
    small = rc.get('small_event_minutes', 0)
    large = rc.get('large_unmatched_minutes', 0)

    if noise + small + large == 0:
        return '<p style="color: #888;">No remaining power data available</p>'

    labels = [
        f'Noise / Baseload<br>(<200W above base)',
        f'Small Events<br>(200-800W)',
        f'Large Unmatched<br>(>800W)',
    ]
    values = [noise, small, large]
    colors = [GRAY, YELLOW, ORANGE]

    data = {
        'values': values,
        'labels': labels,
        'type': 'pie',
        'marker': {'colors': colors},
        'textinfo': 'percent+label',
        'hole': 0.4,
        'textposition': 'outside',
    }

    total_minutes = rc.get('total_minutes', 0)
    layout = {
        'title': f'Remaining Power Classification<br>'
                 f'<sub>{total_minutes:,} total minutes across all phases</sub>',
        'showlegend': False,
        'margin': {'t': 60, 'b': 30, 'l': 100, 'r': 100},
        'height': 400,
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_device_summary_table(metrics: Dict[str, Any]) -> str:
    """
    Create HTML table summarizing detected devices.

    Columns: Device Type | Count | Avg Power | Avg Duration | % of Explained
    """
    devices = metrics.get('devices', {})

    if not devices.get('available', False):
        return '''
        <p style="color: #888;">
            Device activations data not available.
            Run with <code>--minimal_output=False</code> to generate device_activations JSON.
        </p>
        '''

    types = devices.get('types', {})
    if not types:
        return '<p style="color: #888;">No matched device activations found.</p>'

    # Device type display names
    display_names = {
        'boiler': 'Water Heater (Boiler)',
        'central_ac': 'Central AC (Multi-phase)',
        'regular_ac': 'Regular AC (Single-phase)',
        'unclassified': 'Unclassified',
    }

    rows = ''
    for dtype in ['boiler', 'central_ac', 'regular_ac', 'unclassified']:
        info = types.get(dtype)
        if info is None:
            continue

        name = display_names.get(dtype, dtype)
        count = info.get('count', 0)
        avg_mag = info.get('avg_magnitude', 0)
        avg_dur = info.get('avg_duration', 0)
        energy_pct = info.get('energy_pct', 0)

        # Duration formatting
        if avg_dur >= 60:
            dur_str = f'{avg_dur / 60:.1f} hr'
        else:
            dur_str = f'{avg_dur:.0f} min'

        # Energy bar
        bar_width = min(energy_pct, 100)

        rows += f'''
        <tr>
            <td style="padding: 12px 15px; border-bottom: 1px solid #eee; font-weight: 600;">{name}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid #eee; text-align: center;">{count}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid #eee; text-align: center;">{avg_mag:,}W</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid #eee; text-align: center;">{dur_str}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid #eee;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="background: {LIGHT_GREEN}; border-radius: 4px; width: 100px; height: 18px; overflow: hidden;">
                        <div style="background: {GREEN}; height: 100%; width: {bar_width}%;"></div>
                    </div>
                    <span>{energy_pct:.1f}%</span>
                </div>
            </td>
        </tr>
        '''

    # Summary row
    total_matched = devices.get('total_matched', 0)
    total_unmatched = devices.get('total_unmatched', 0)

    return f'''
    <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
        <thead>
            <tr style="background: #2d3748; color: white;">
                <th style="padding: 12px 15px; text-align: left;">Device Type</th>
                <th style="padding: 12px 15px; text-align: center;">Count</th>
                <th style="padding: 12px 15px; text-align: center;">Avg Power</th>
                <th style="padding: 12px 15px; text-align: center;">Avg Duration</th>
                <th style="padding: 12px 15px; text-align: left;">% of Explained Energy</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
        Total: {total_matched} matched activations, {total_unmatched} unmatched events
    </div>
    '''
