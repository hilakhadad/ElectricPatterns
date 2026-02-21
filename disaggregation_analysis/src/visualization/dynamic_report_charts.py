"""
Chart generation for dynamic threshold experiment reports.

Creates visualization sections, each returning an HTML string
with embedded Plotly charts or pure HTML/CSS.

Color scheme (no red):
  Green  = #28a745 (segregated / success)
  Gray   = #6c757d (background / baseload)
  Orange = #e67e22 (unmatched / attention)
  Yellow = #eab308 (sub-threshold / warm yellow)
"""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# Color constants
GREEN = '#28a745'           # Segregated (success)
GRAY = '#6c757d'            # Background (baseload)
ORANGE = '#e67e22'          # Unmatched (distinct orange)
YELLOW = '#eab308'          # Sub-threshold (warm yellow)
PURPLE = '#6f42c1'          # No Data (NaN minutes)
LIGHT_GREEN = '#d4edda'
LIGHT_GRAY = '#e9ecef'
LIGHT_ORANGE = '#fde8cd'    # Unmatched card background
LIGHT_YELLOW = '#fef9e7'    # Sub-threshold card background
LIGHT_PURPLE = '#e8daf0'


def create_summary_boxes(metrics: Dict[str, Any]) -> str:
    """
    Create summary section with decomposition box and efficiency card.

    Decomposition box (bordered): Segregated + Background + Unmatched + Sub-threshold [+ No Data] = 100%
    Efficiency card (separate): Segregated / (Segregated + Unmatched) — only detectable power.
    """
    totals = metrics.get('totals', {})
    phases = metrics.get('phases', {})
    segregated_pct = totals.get('segregated_pct', 0)
    background_pct = totals.get('background_pct', 0)
    above_th_pct = totals.get('above_th_pct', 0)
    sub_threshold_pct = totals.get('sub_threshold_pct', 0)
    no_data_pct = totals.get('no_data_pct', 0)
    efficiency = totals.get('efficiency', 0)

    segregated_kwh = totals.get('segregated_kwh', 0)
    background_kwh = totals.get('background_kwh', 0)
    above_th_kwh = totals.get('above_th_kwh', 0)
    sub_threshold_kwh = totals.get('sub_threshold_kwh', 0)
    total_kwh = totals.get('total_kwh', 0)

    has_no_data = no_data_pct >= 0.1

    # Average background per minute across phases (watts)
    bg_per_min = [phases.get(p, {}).get('background_per_minute', 0) for p in ['w1', 'w2', 'w3']]
    avg_bg_watts = round(sum(bg_per_min) / 3) if bg_per_min else 0

    min_threshold = metrics.get('threshold_schedule', [800])[-1]

    # Efficiency color
    if efficiency >= 70:
        eff_color = GREEN
    elif efficiency >= 50:
        eff_color = YELLOW
    else:
        eff_color = ORANGE

    # Efficiency background color (lighter tint)
    if efficiency >= 70:
        eff_bg = '#e8f5e9'
        eff_border = GREEN
    elif efficiency >= 50:
        eff_bg = '#fff8e1'
        eff_border = YELLOW
    else:
        eff_bg = '#fff3e0'
        eff_border = ORANGE

    # Card count for grid (4 or 5 columns depending on No Data)
    grid_cols = 5 if has_no_data else 4
    total_sum = segregated_pct + background_pct + above_th_pct + sub_threshold_pct + no_data_pct

    # Build No Data card (pre-built for Python 3.9 compat)
    no_data_card_html = ''
    no_data_formula_html = ''
    if has_no_data:
        no_data_card_html = (
            '<div style="background:' + LIGHT_PURPLE + '; border-left: 4px solid ' + PURPLE
            + '; border-radius: 8px; padding: 14px; text-align: center;">'
            '<div style="font-size: 1.8em; font-weight: bold; color: ' + PURPLE + ';">'
            + str(round(no_data_pct, 1)) + '%</div>'
            '<div style="font-size: 0.9em; color: #4a0e6b; font-weight: 600;">No Data</div>'
            '<div style="font-size: 0.75em; color: #888; margin-top: 4px; line-height: 1.4;">'
            'Minutes with no power reading (NaN).<br>'
            '<em>Not included in any calculation.</em>'
            '</div></div>'
        )
        no_data_formula_html = (
            ' + <span style="color:' + PURPLE + ';">No Data ('
            + str(round(no_data_pct, 1)) + '%)</span>'
        )

    return f'''
    <div style="border: 2px solid #dee2e6; border-radius: 12px; padding: 18px; margin-bottom: 18px;">
        <div style="font-size: 0.82em; font-weight: 600; color: #555; margin-bottom: 12px;">
            Power Decomposition (= 100%)
        </div>
        <div style="display: grid; grid-template-columns: repeat({grid_cols}, 1fr); gap: 12px; margin-bottom: 12px;">
            <div style="background: {LIGHT_GREEN}; border-left: 4px solid {GREEN}; border-radius: 8px; padding: 14px; text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: {GREEN};">{segregated_pct:.1f}%</div>
                <div style="font-size: 0.9em; color: #155724; font-weight: 600;">Segregated</div>
                <div style="font-size: 0.8em; color: #666;">{segregated_kwh} kWh</div>
                <div style="font-size: 0.75em; color: #888; margin-top: 4px; line-height: 1.4;">
                    Matched ON&rarr;OFF events (boilers, ACs, high-power devices).<br>
                    <em>Per-minute difference: original &minus; remaining.</em>
                </div>
            </div>
            <div style="background: {LIGHT_ORANGE}; border-left: 4px solid {ORANGE}; border-radius: 8px; padding: 14px; text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: {ORANGE};">{above_th_pct:.1f}%</div>
                <div style="font-size: 0.9em; color: #7a4510; font-weight: 600;">Unmatched</div>
                <div style="font-size: 0.8em; color: #666;">{above_th_kwh} kWh</div>
                <div style="font-size: 0.75em; color: #888; margin-top: 4px; line-height: 1.4;">
                    Above {min_threshold}W, not matched. Includes complex-pattern<br>
                    devices not targeted for detection.
                </div>
            </div>
            <div style="background: {LIGHT_YELLOW}; border-left: 4px solid {YELLOW}; border-radius: 8px; padding: 14px; text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: #a38600;">{sub_threshold_pct:.1f}%</div>
                <div style="font-size: 0.9em; color: #7a6400; font-weight: 600;">Sub-threshold</div>
                <div style="font-size: 0.8em; color: #666;">{sub_threshold_kwh} kWh</div>
                <div style="font-size: 0.75em; color: #888; margin-top: 4px; line-height: 1.4;">
                    Between background and {min_threshold}W.<br>
                    Below detection threshold &mdash; undetectable by design.
                </div>
            </div>
            <div style="background: {LIGHT_GRAY}; border-left: 4px solid {GRAY}; border-radius: 8px; padding: 14px; text-align: center;">
                <div style="font-size: 1.8em; font-weight: bold; color: {GRAY};">{background_pct:.1f}%</div>
                <div style="font-size: 0.9em; color: #495057; font-weight: 600;">Background</div>
                <div style="font-size: 0.8em; color: #666;">{background_kwh} kWh</div>
                <div style="font-size: 0.75em; color: #888; margin-top: 4px; line-height: 1.4;">
                    Always-on baseload (~{avg_bg_watts}W avg/phase).<br>
                    <em>P5 (5th percentile) &times; measured minutes.</em>
                </div>
            </div>
            {no_data_card_html}
        </div>
        <div style="background: #f8f9fa; border-radius: 6px; padding: 8px 15px; text-align: center; font-size: 0.82em; color: #555;">
            <span style="color:{GREEN};">Segregated ({segregated_pct:.1f}%)</span> +
            <span style="color:{ORANGE};">Unmatched ({above_th_pct:.1f}%)</span> +
            <span style="color:{YELLOW};">Sub-threshold ({sub_threshold_pct:.1f}%)</span> +
            <span style="color:{GRAY};">Background ({background_pct:.1f}%)</span>
            {no_data_formula_html}
            = {total_sum:.1f}%
        </div>
    </div>

    <div style="background: {eff_bg}; border: 2px solid {eff_border}; border-radius: 12px; padding: 15px 20px; text-align: center; margin-bottom: 8px;">
        <div style="font-size: 2.8em; font-weight: bold; color: {eff_color};">{efficiency:.1f}%</div>
        <div style="font-size: 1em; font-weight: 700; color: #333;">Detection Efficiency</div>
        <div style="font-size: 0.82em; color: #555; margin-top: 4px; line-height: 1.5;">
            <strong>Efficiency</strong> = {segregated_kwh} / ({segregated_kwh} + {above_th_kwh}) kWh
            = <strong style="color:{eff_color};">{efficiency:.1f}%</strong><br>
            <em>Scope: only above-threshold (&ge;{min_threshold}W) power. Background and sub-threshold excluded.</em>
        </div>
    </div>
    '''


def create_power_breakdown_bar(metrics: Dict[str, Any]) -> str:
    """
    Create stacked horizontal bar chart showing power decomposition per phase.

    Each phase gets one bar: [Segregated | Background | Unmatched | No Data]
    """
    phases = metrics.get('phases', {})
    chart_id = 'power-breakdown-chart'

    phase_labels = ['w1', 'w2', 'w3']
    segregated_vals = []
    background_vals = []
    above_th_vals = []
    sub_th_vals = []
    no_data_vals = []
    hover_segregated = []
    hover_bg = []
    hover_above = []
    hover_sub = []
    hover_nd = []

    for p in phase_labels:
        ph = phases.get(p, {})
        segregated_vals.append(ph.get('segregated_pct', 0))
        background_vals.append(ph.get('background_pct', 0))
        above_th_vals.append(ph.get('above_th_pct', 0))
        sub_th_vals.append(ph.get('sub_threshold_pct', 0))
        no_data_vals.append(ph.get('no_data_pct', 0))
        hover_segregated.append(
            f"{p}: {ph.get('segregated_kwh', 0)} kWh ({ph.get('segregated_pct', 0):.1f}%)"
        )
        hover_bg.append(
            f"{p}: {ph.get('background_kwh', 0)} kWh ({ph.get('background_pct', 0):.1f}%)"
        )
        hover_above.append(
            f"{p}: {ph.get('above_th_kwh', 0)} kWh ({ph.get('above_th_pct', 0):.1f}%)"
        )
        hover_sub.append(
            f"{p}: {ph.get('sub_threshold_kwh', 0)} kWh ({ph.get('sub_threshold_pct', 0):.1f}%)"
        )
        nan_min = ph.get('nan_minutes', 0)
        hover_nd.append(
            f"{p}: {nan_min:,} NaN minutes ({ph.get('no_data_pct', 0):.1f}%)"
        )

    has_no_data = any(v >= 0.1 for v in no_data_vals)

    trace_segregated = {
        'y': phase_labels, 'x': segregated_vals, 'name': 'Segregated',
        'type': 'bar', 'orientation': 'h', 'marker': {'color': GREEN},
        'text': [f'{v:.1f}%' for v in segregated_vals], 'textposition': 'inside',
        'hovertext': hover_segregated, 'hoverinfo': 'text',
    }
    trace_above_th = {
        'y': phase_labels, 'x': above_th_vals, 'name': 'Unmatched',
        'type': 'bar', 'orientation': 'h', 'marker': {'color': ORANGE},
        'text': [f'{v:.1f}%' for v in above_th_vals], 'textposition': 'inside',
        'hovertext': hover_above, 'hoverinfo': 'text',
    }
    trace_sub_th = {
        'y': phase_labels, 'x': sub_th_vals, 'name': 'Sub-threshold',
        'type': 'bar', 'orientation': 'h', 'marker': {'color': YELLOW},
        'text': [f'{v:.1f}%' for v in sub_th_vals], 'textposition': 'inside',
        'hovertext': hover_sub, 'hoverinfo': 'text',
    }
    trace_background = {
        'y': phase_labels, 'x': background_vals, 'name': 'Background',
        'type': 'bar', 'orientation': 'h', 'marker': {'color': GRAY},
        'text': [f'{v:.1f}%' for v in background_vals], 'textposition': 'inside',
        'hovertext': hover_bg, 'hoverinfo': 'text',
    }

    traces = [trace_segregated, trace_above_th, trace_sub_th, trace_background]

    if has_no_data:
        trace_no_data = {
            'y': phase_labels, 'x': no_data_vals, 'name': 'No Data',
            'type': 'bar', 'orientation': 'h', 'marker': {'color': PURPLE},
            'text': [f'{v:.1f}%' if v >= 1 else '' for v in no_data_vals],
            'textposition': 'inside',
            'hovertext': hover_nd, 'hoverinfo': 'text',
        }
        traces.append(trace_no_data)

    layout = {
        'title': 'Power Decomposition by Phase',
        'barmode': 'stack',
        'xaxis': {'title': '% of Total Period', 'range': [0, 105]},
        'yaxis': {'title': ''},
        'legend': {'orientation': 'h', 'y': -0.25},
        'margin': {'l': 50, 'r': 30, 't': 50, 'b': 80},
        'height': 320,
    }

    data_json = json.dumps(traces)

    return f'''
    <div id="{chart_id}" style="width:100%;height:320px;"></div>
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
                {ph.get('segregated_kwh', 0)} / {round(ph.get('total_kwh', 0) - ph.get('background_kwh', 0), 2)} kWh
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

    values = [t.get('segregated_pct', 0) for t in per_threshold]
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
        'yaxis': {'title': '% of Total Power Segregated'},
        'xaxis': {'title': ''},
        'showlegend': False,
        'margin': {'t': 60, 'b': 50, 'l': 60, 'r': 30},
        'height': 400,
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
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

    Columns: Device Type | Count | Avg Power | Avg Duration | % of Segregated
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
                <th style="padding: 12px 15px; text-align: left;" title="Classified device type based on power, duration, and phase patterns">Device Type</th>
                <th style="padding: 12px 15px; text-align: center;" title="Number of matched ON/OFF activation pairs">Count</th>
                <th style="padding: 12px 15px; text-align: center;" title="Average ON magnitude across all activations of this type">Avg Power</th>
                <th style="padding: 12px 15px; text-align: center;" title="Average total duration from ON start to OFF end">Avg Duration</th>
                <th style="padding: 12px 15px; text-align: left;" title="This device type's energy share of all segregated energy (magnitude x duration)">% of Segregated Energy</th>
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


def _parse_activation_row(act: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a single activation into display-ready fields."""
    on_start_raw = act.get('on_start')
    off_end_raw = act.get('off_end') or act.get('off_start')

    date_str = on_time_str = off_time_str = ''

    if on_start_raw:
        try:
            on_dt = datetime.fromisoformat(str(on_start_raw))
            date_str = on_dt.strftime('%Y-%m-%d')
            on_time_str = on_dt.strftime('%H:%M')
        except (ValueError, TypeError):
            date_str = str(on_start_raw)[:10]
            on_time_str = str(on_start_raw)[11:16] if len(str(on_start_raw)) > 16 else ''

    if off_end_raw:
        try:
            off_dt = datetime.fromisoformat(str(off_end_raw))
            off_time_str = off_dt.strftime('%H:%M')
        except (ValueError, TypeError):
            off_time_str = str(off_end_raw)[11:16] if len(str(off_end_raw)) > 16 else ''

    duration = act.get('duration', 0) or 0
    dur_str = f'{duration / 60:.1f} hr' if duration >= 60 else f'{duration:.0f} min'
    magnitude = abs(act.get('on_magnitude', 0) or 0)

    confidence = act.get('confidence', 0) or 0

    return {
        'date': date_str, 'start': on_time_str, 'end': off_time_str,
        'duration': duration, 'dur_str': dur_str,
        'magnitude': magnitude, 'phase': act.get('phase', ''),
        'tag': act.get('tag', ''), 'device_type': act.get('device_type') or 'unclassified',
        'confidence': confidence,
        'session_id': act.get('session_id', ''),
        'on_start_iso': str(on_start_raw) if on_start_raw else '',
        'off_end_iso': str(off_end_raw) if off_end_raw else '',
    }


def _group_central_ac_for_display(events: List[Dict]) -> List[Dict]:
    """Group central AC events into session-level rows with V/X per phase."""
    has_session_id = any(e.get('session_id') for e in events)

    if has_session_id:
        groups: Dict[str, list] = {}
        for e in events:
            sid = e['session_id']
            groups.setdefault(sid, []).append(e)
        session_groups = list(groups.values())
    else:
        # Fallback: group by time proximity (within 10 min)
        sorted_events = sorted(events, key=lambda e: e['on_start_iso'] or e['date'] + e['start'])
        session_groups = [[sorted_events[0]]]
        for e in sorted_events[1:]:
            last = session_groups[-1][-1]
            try:
                t1 = datetime.fromisoformat(last['on_start_iso'])
                t2 = datetime.fromisoformat(e['on_start_iso'])
                if abs((t2 - t1).total_seconds()) / 60 <= 10:
                    session_groups[-1].append(e)
                    continue
            except (ValueError, TypeError):
                pass
            session_groups.append([e])

    result = []
    for group in session_groups:
        phases_present = set(e['phase'] for e in group if e.get('phase'))

        iso_starts = [e['on_start_iso'] for e in group if e['on_start_iso']]
        iso_ends = [e['off_end_iso'] for e in group if e['off_end_iso']]

        date_str = start_str = end_str = ''
        session_dur = max((e['duration'] for e in group), default=0)

        if iso_starts:
            earliest = min(iso_starts)
            try:
                dt = datetime.fromisoformat(earliest)
                date_str = dt.strftime('%Y-%m-%d')
                start_str = dt.strftime('%H:%M')
            except (ValueError, TypeError):
                date_str = earliest[:10]
                start_str = earliest[11:16] if len(earliest) > 16 else ''

        if iso_ends:
            latest = max(iso_ends)
            try:
                dt = datetime.fromisoformat(latest)
                end_str = dt.strftime('%H:%M')
            except (ValueError, TypeError):
                end_str = latest[11:16] if len(latest) > 16 else ''

        if iso_starts and iso_ends:
            try:
                t_start = datetime.fromisoformat(min(iso_starts))
                t_end = datetime.fromisoformat(max(iso_ends))
                session_dur = (t_end - t_start).total_seconds() / 60
            except (ValueError, TypeError):
                pass

        if not date_str:
            date_str = group[0].get('date', '')
        if not start_str:
            start_str = group[0].get('start', '')
        if not end_str:
            end_str = group[0].get('end', '')

        dur_str = f'{session_dur / 60:.1f} hr' if session_dur >= 60 else f'{session_dur:.0f} min'
        avg_mag = sum(e['magnitude'] for e in group) / len(group) if group else 0
        conf_val = max((e.get('confidence', 0) for e in group), default=0)

        result.append({
            'date': date_str, 'start': start_str, 'end': end_str,
            'duration': session_dur, 'dur_str': dur_str,
            'magnitude': avg_mag,
            'w1': 'V' if 'w1' in phases_present else 'X',
            'w2': 'V' if 'w2' in phases_present else 'X',
            'w3': 'V' if 'w3' in phases_present else 'X',
            'cycle_count': len(group),
            'confidence': conf_val,
        })

    result.sort(key=lambda r: r['date'] + r['start'])
    return result


def create_device_activations_detail(activations: List[Dict[str, Any]]) -> str:
    """
    Create detailed device activations tables grouped by device_type.

    Filters to high-confidence detections using minimum duration per type.
    Includes sortable columns, per-type Copy Dates, and a Copy All button.
    """
    if not activations:
        return '<p style="color: #888;">No device activations data available.</p>'

    matched = [a for a in activations if a.get('match_type') == 'matched']
    if not matched:
        return '<p style="color: #888;">No matched device activations found.</p>'

    # Minimum duration (minutes) for high-confidence classification
    MIN_DURATION = {
        'boiler': 15,
        'central_ac': 5,
        'regular_ac': 3,
        'unclassified': 0,
    }

    # Group by device_type, filtering by min duration
    device_groups = {}
    all_copyable = []
    for act in matched:
        parsed = _parse_activation_row(act)
        dtype = parsed['device_type']
        min_dur = MIN_DURATION.get(dtype, 0)
        if parsed['duration'] < min_dur:
            continue
        if dtype not in device_groups:
            device_groups[dtype] = []
        device_groups[dtype].append(parsed)
        if parsed['date'] and parsed['start']:
            all_copyable.append(f"{parsed['date']} {parsed['start']}-{parsed['end']}")

    if not any(device_groups.values()):
        return '<p style="color: #888;">No high-confidence device activations found.</p>'

    display_names = {
        'boiler': 'Water Heater (Boiler)',
        'central_ac': 'Central AC (Multi-phase)',
        'regular_ac': 'Regular AC (Single-phase)',
        'unclassified': 'Unclassified Devices',
    }
    display_order = ['boiler', 'central_ac', 'regular_ac', 'unclassified']

    # Copy All button
    total_count = sum(len(v) for v in device_groups.values())
    all_copyable_text = ', '.join(all_copyable)
    copy_all_html = f'''
    <div style="margin-bottom: 15px;">
        <button onclick="var ta=document.getElementById('all-activations-dates'); ta.style.display = ta.style.display==='none' ? 'block' : 'none';"
                style="padding: 5px 14px; border: 1px solid #667eea; border-radius: 4px; background: #667eea; color: white; cursor: pointer; font-size: 0.85em;">
            Copy All Dates ({total_count})
        </button>
        <textarea id="all-activations-dates" readonly
                  style="display: none; width: 100%; margin-top: 5px; padding: 8px; font-size: 0.85em; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; resize: vertical; min-height: 60px;"
                  onclick="this.select();">{all_copyable_text}</textarea>
    </div>'''

    sections_html = copy_all_html
    section_idx = 0

    for dtype in display_order:
        events = device_groups.get(dtype)
        if not events:
            continue

        name = display_names.get(dtype, dtype)
        min_dur = MIN_DURATION.get(dtype, 0)
        filter_note = f' (≥{min_dur} min)' if min_dur > 0 else ''
        section_id = f'device-detail-{section_idx}'
        section_idx += 1

        _cell = 'padding: 5px 8px; border-bottom: 1px solid #eee;'
        _th = 'padding: 5px 8px;'
        rows = ''
        copyable_dates = []

        if dtype == 'central_ac':
            # --- Central AC: one row per session with V/X per phase ---
            session_rows = _group_central_ac_for_display(events)
            count = len(session_rows)
            for i, r in enumerate(session_rows, 1):
                if r['date'] and r['start']:
                    copyable_dates.append(f"{r['date']} {r['start']}-{r['end']}")
                conf_val = r.get('confidence', 0)
                conf_pct = f'{conf_val:.0%}' if conf_val else '-'
                conf_color = '#48bb78' if conf_val >= 0.8 else '#ecc94b' if conf_val >= 0.6 else '#fc8181' if conf_val > 0 else '#ccc'
                rows += f'''
            <tr>
                <td style="{_cell} text-align: center; color: #aaa; font-size: 0.85em;">{i}</td>
                <td style="{_cell}" data-value="{r['date']}">{r['date']}</td>
                <td style="{_cell} text-align: center;">{r['start']}</td>
                <td style="{_cell} text-align: center;">{r['end']}</td>
                <td style="{_cell} text-align: center;" data-value="{r['duration']}">{r['dur_str']}</td>
                <td style="{_cell} text-align: right;" data-value="{r['magnitude']}">{r['magnitude']:,.0f}W</td>
                <td style="{_cell} text-align: center; color: {"#48bb78" if r["w1"]=="V" else "#e2e8f0"}; font-weight: {"bold" if r["w1"]=="V" else "normal"};">{r['w1']}</td>
                <td style="{_cell} text-align: center; color: {"#48bb78" if r["w2"]=="V" else "#e2e8f0"}; font-weight: {"bold" if r["w2"]=="V" else "normal"};">{r['w2']}</td>
                <td style="{_cell} text-align: center; color: {"#48bb78" if r["w3"]=="V" else "#e2e8f0"}; font-weight: {"bold" if r["w3"]=="V" else "normal"};">{r['w3']}</td>
                <td style="{_cell} text-align: center;">{r['cycle_count']}</td>
                <td style="{_cell} text-align: center; color: {conf_color}; font-weight: 600;" data-value="{conf_val}">{conf_pct}</td>
            </tr>'''

            table_header = f'''
                        <tr style="background: #f8f9fa;">
                            <th style="{_th} text-align: center; width: 35px;">#</th>
                            <th style="{_th} text-align: left; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 1, 'str')">Date &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center;">Start</th>
                            <th style="{_th} text-align: center;">End</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 4, 'num')">Duration &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: right; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 5, 'num')">Avg Power &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center;">w1</th>
                            <th style="{_th} text-align: center;">w2</th>
                            <th style="{_th} text-align: center;">w3</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 9, 'num')">Cycles &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 10, 'num')">Confidence &#x25B4;&#x25BE;</th>
                        </tr>'''
        else:
            # --- Standard table: one row per event ---
            events.sort(key=lambda r: r['date'] + r['start'])
            count = len(events)
            for i, r in enumerate(events, 1):
                if r['date'] and r['start']:
                    copyable_dates.append(f"{r['date']} {r['start']}-{r['end']}")
                conf_val = r.get('confidence', 0)
                conf_pct = f'{conf_val:.0%}' if conf_val else '-'
                conf_color = '#48bb78' if conf_val >= 0.8 else '#ecc94b' if conf_val >= 0.6 else '#fc8181' if conf_val > 0 else '#ccc'
                rows += f'''
            <tr>
                <td style="{_cell} text-align: center; color: #aaa; font-size: 0.85em;">{i}</td>
                <td style="{_cell}" data-value="{r['date']}">{r['date']}</td>
                <td style="{_cell} text-align: center;">{r['start']}</td>
                <td style="{_cell} text-align: center;">{r['end']}</td>
                <td style="{_cell} text-align: center;" data-value="{r['duration']}">{r['dur_str']}</td>
                <td style="{_cell} text-align: right;" data-value="{r['magnitude']}">{r['magnitude']:,.0f}W</td>
                <td style="{_cell} text-align: center;">{r['phase']}</td>
                <td style="{_cell} text-align: center; color: {conf_color}; font-weight: 600;" data-value="{conf_val}">{conf_pct}</td>
            </tr>'''

            table_header = f'''
                        <tr style="background: #f8f9fa;">
                            <th style="{_th} text-align: center; width: 35px;">#</th>
                            <th style="{_th} text-align: left; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 1, 'str')">Date &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center;">Start</th>
                            <th style="{_th} text-align: center;">End</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 4, 'num')">Duration &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: right; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 5, 'num')">Power &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center;">Phase</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 7, 'num')">Confidence &#x25B4;&#x25BE;</th>
                        </tr>'''

        copyable_text = ', '.join(copyable_dates)

        sections_html += f'''
        <div style="margin-bottom: 15px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px; cursor: pointer;"
                 onclick="var tbl=document.getElementById('{section_id}'); tbl.style.display = tbl.style.display==='none' ? 'block' : 'none'; var arrow=this.querySelector('.toggle-arrow'); arrow.textContent = tbl.style.display==='none' ? '\\u25B6' : '\\u25BC';">
                <span class="toggle-arrow" style="font-size: 0.85em;">&#x25BC;</span>
                <strong style="color: #2d3748; font-size: 0.95em;">{name}</strong>
                <span style="background: #667eea; color: white; padding: 1px 8px; border-radius: 10px; font-size: 0.8em;">{count}{filter_note}</span>
            </div>
            <div id="{section_id}">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85em;" id="{section_id}-table">
                    <thead>
                        {table_header}
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
                <div style="margin-top: 5px;">
                    <button onclick="var ta=document.getElementById('{section_id}-dates'); ta.style.display = ta.style.display==='none' ? 'block' : 'none';"
                            style="padding: 3px 10px; border: 1px solid #ccc; border-radius: 4px; background: #f8f9fa; cursor: pointer; font-size: 0.8em;">
                        Copy Dates ({count})
                    </button>
                    <textarea id="{section_id}-dates" readonly
                              style="display: none; width: 100%; margin-top: 4px; padding: 6px; font-size: 0.8em; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; resize: vertical; min-height: 50px;"
                              onclick="this.select();">{copyable_text}</textarea>
                </div>
            </div>
        </div>'''

    return sections_html
