"""
Chart generation for dynamic threshold experiment reports.

Creates visualization sections, each returning an HTML string
with embedded Plotly charts or pure HTML/CSS.

Color scheme (no red):
  Green  = #28a745 (explained / success)
  Gray   = #6c757d (background / baseload)
  Orange = #fd7e14 (unmatched / attention)
  Yellow = #ffc107 (medium efficiency)
"""
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

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
    Create summary section with full explanations.

    Shows Detection Efficiency hero card, 3 decomposition boxes,
    an equation bar, and a plain-language explanation block.
    """
    totals = metrics.get('totals', {})
    phases = metrics.get('phases', {})
    explained_pct = totals.get('explained_pct', 0)
    background_pct = totals.get('background_pct', 0)
    improvable_pct = totals.get('improvable_pct', 0)
    efficiency = totals.get('efficiency', 0)

    explained_kwh = totals.get('explained_kwh', 0)
    background_kwh = totals.get('background_kwh', 0)
    improvable_kwh = totals.get('improvable_kwh', 0)
    total_kwh = totals.get('total_kwh', 0)

    # Average background per minute across phases (watts)
    bg_per_min = [phases.get(p, {}).get('background_per_minute', 0) for p in ['w1', 'w2', 'w3']]
    avg_bg_watts = round(sum(bg_per_min) / 3) if bg_per_min else 0

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

    # Explained color: higher is better
    if explained_pct >= 40:
        exp_color, exp_border, exp_bg, exp_label_color = GREEN, GREEN, LIGHT_GREEN, '#155724'
    elif explained_pct >= 20:
        exp_color, exp_border, exp_bg, exp_label_color = YELLOW, YELLOW, '#fff8e1', '#856404'
    else:
        exp_color, exp_border, exp_bg, exp_label_color = ORANGE, ORANGE, LIGHT_ORANGE, '#856404'

    # Background color: lower is typical; very high = notable
    if background_pct <= 40:
        bg_color, bg_border, bg_bg, bg_label_color = GRAY, GRAY, LIGHT_GRAY, '#495057'
    elif background_pct <= 60:
        bg_color, bg_border, bg_bg, bg_label_color = YELLOW, YELLOW, '#fff8e1', '#856404'
    else:
        bg_color, bg_border, bg_bg, bg_label_color = ORANGE, ORANGE, LIGHT_ORANGE, '#856404'

    # Improvable color: lower is better
    if improvable_pct <= 20:
        imp_color, imp_border, imp_bg, imp_label_color = GREEN, GREEN, LIGHT_GREEN, '#155724'
    elif improvable_pct <= 40:
        imp_color, imp_border, imp_bg, imp_label_color = YELLOW, YELLOW, '#fff8e1', '#856404'
    else:
        imp_color, imp_border, imp_bg, imp_label_color = ORANGE, ORANGE, LIGHT_ORANGE, '#856404'

    # Complementary percentages
    not_explained_pct = 100 - explained_pct
    targetable_pct = 100 - background_pct

    min_threshold = metrics.get('threshold_schedule', [800])[-1]

    return f'''
    <p style="color: #555; margin-bottom: 20px; line-height: 1.7; font-size: 0.95em;">
        The total electricity consumption of this household ({total_kwh} kWh) is broken down into three parts:
        <strong>Explained</strong> &mdash; power attributed to specific device activations that the algorithm detected
        (e.g. boilers, air conditioners);
        <strong>Background</strong> &mdash; constant baseload from always-on appliances;
        and <strong>Unmatched</strong> &mdash; the remaining power above the baseline that was not matched to any device.
        Together these three parts always sum to 100%.
    </p>

    <div style="background: {eff_bg}; border: 3px solid {eff_border}; border-radius: 14px; padding: 25px 30px; text-align: center; margin-bottom: 25px;">
        <div style="font-size: 3.8em; font-weight: bold; color: {eff_color};">{efficiency:.1f}%</div>
        <div style="font-size: 1.2em; font-weight: 700; color: #333; margin-top: 5px;">Detection Efficiency</div>
        <div style="font-size: 0.9em; color: #555; margin-top: 8px; max-width: 750px; margin-left: auto; margin-right: auto; line-height: 1.6;">
            Of the total power that is <em>not</em> background ({targetable_pct:.1f}% of total),
            the algorithm successfully attributed <strong>{efficiency:.1f}%</strong> to specific device events.
            The remaining <strong>{100 - efficiency:.1f}%</strong> was not matched &mdash;
            this includes both devices that could potentially be detected with lower thresholds,
            and complex-pattern appliances (ovens, washing machines, dishwashers) whose variable
            power profiles are not targeted by the current algorithm.
        </div>
    </div>

    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 15px;">
        <div style="background: {exp_bg}; border-left: 5px solid {exp_border}; border-radius: 10px; padding: 20px; text-align: center;">
            <div style="font-size: 2.2em; font-weight: bold; color: {exp_color};">{explained_pct:.1f}%</div>
            <div style="font-size: 1em; color: {exp_label_color}; font-weight: 600; margin-top: 5px;">Explained</div>
            <div style="font-size: 0.85em; color: #666; margin-top: 3px;">{explained_kwh} kWh</div>
            <div style="font-size: 0.8em; color: #888; margin-top: 8px; line-height: 1.5;">
                Power successfully matched to detected ON&rarr;OFF device events
                (boilers, air conditioners, and other high-power appliances with clear
                on/off signatures). The other <strong>{not_explained_pct:.1f}%</strong>
                is either background or unmatched.
            </div>
        </div>
        <div style="background: {bg_bg}; border-left: 5px solid {bg_border}; border-radius: 10px; padding: 20px; text-align: center;">
            <div style="font-size: 2.2em; font-weight: bold; color: {bg_color};">{background_pct:.1f}%</div>
            <div style="font-size: 1em; color: {bg_label_color}; font-weight: 600; margin-top: 5px;">Background</div>
            <div style="font-size: 0.85em; color: #666; margin-top: 3px;">{background_kwh} kWh</div>
            <div style="font-size: 0.8em; color: #888; margin-top: 8px; line-height: 1.5;">
                Constant baseload: always-on appliances (fridge, router, standby devices).
                Estimated as the 5th percentile of power (~{avg_bg_watts}W avg per phase).
                This power is <strong>not targetable</strong> for event-based detection.
            </div>
        </div>
        <div style="background: {imp_bg}; border-left: 5px solid {imp_border}; border-radius: 10px; padding: 20px; text-align: center;">
            <div style="font-size: 2.2em; font-weight: bold; color: {imp_color};">{improvable_pct:.1f}%</div>
            <div style="font-size: 1em; color: {imp_label_color}; font-weight: 600; margin-top: 5px;">Unmatched</div>
            <div style="font-size: 0.85em; color: #666; margin-top: 3px;">{improvable_kwh} kWh</div>
            <div style="font-size: 0.8em; color: #888; margin-top: 8px; line-height: 1.5;">
                Power above background not matched to any device. This includes:
                events below the {min_threshold}W detection threshold,
                complex-pattern appliances (ovens, washing machines, dishwashers)
                whose variable power profiles are not targeted by the algorithm,
                and irregular or noisy consumption.
                <strong>Not all of this is necessarily "improvable".</strong>
            </div>
        </div>
    </div>

    <div style="background: #f8f9fa; border-radius: 8px; padding: 12px 20px; margin-bottom: 10px; text-align: center; font-size: 0.9em; color: #555;">
        <strong>Equation:</strong>
        <span style="color:{exp_color};">Explained ({explained_pct:.1f}%)</span> +
        <span style="color:{bg_color};">Background ({background_pct:.1f}%)</span> +
        <span style="color:{imp_color};">Unmatched ({improvable_pct:.1f}%)</span>
        = 100% of Total ({total_kwh} kWh)
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <strong>Efficiency</strong> = Explained / (Total &minus; Background) =
        {explained_pct:.1f}% / {targetable_pct:.1f}% = <strong style="color:{eff_color};">{efficiency:.1f}%</strong>
    </div>
    '''


def create_power_breakdown_bar(metrics: Dict[str, Any]) -> str:
    """
    Create stacked horizontal bar chart showing power decomposition per phase.

    Each phase gets one bar: [Explained (green) | Background (gray) | Unmatched (orange)]
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
        'name': 'Unmatched',
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
                <th style="padding: 12px 15px; text-align: left;" title="Classified device type based on power, duration, and phase patterns">Device Type</th>
                <th style="padding: 12px 15px; text-align: center;" title="Number of matched ON/OFF activation pairs">Count</th>
                <th style="padding: 12px 15px; text-align: center;" title="Average ON magnitude across all activations of this type">Avg Power</th>
                <th style="padding: 12px 15px; text-align: center;" title="Average total duration from ON start to OFF end">Avg Duration</th>
                <th style="padding: 12px 15px; text-align: left;" title="This device type's energy share of all explained energy (magnitude x duration)">% of Explained Energy</th>
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


def create_device_activations_detail(activations: List[Dict[str, Any]]) -> str:
    """
    Create detailed device activations tables with dates and time ranges.

    Groups matched activations by device_type, showing individual events
    with date, start time, end time, duration, and power.
    Includes a copyable dates section per device type.

    Args:
        activations: List of activation dicts from device_activations JSON

    Returns:
        HTML string with collapsible tables per device type
    """
    if not activations:
        return '<p style="color: #888;">No device activations data available.</p>'

    matched = [a for a in activations if a.get('match_type') == 'matched']
    if not matched:
        return '<p style="color: #888;">No matched device activations found.</p>'

    # Group by device_type
    device_groups = {}
    for act in matched:
        dtype = act.get('device_type') or 'unclassified'
        if dtype not in device_groups:
            device_groups[dtype] = []
        device_groups[dtype].append(act)

    display_names = {
        'boiler': 'Water Heater (Boiler)',
        'central_ac': 'Central AC (Multi-phase)',
        'regular_ac': 'Regular AC (Single-phase)',
        'unclassified': 'Unclassified Devices',
    }

    display_order = ['boiler', 'central_ac', 'regular_ac', 'unclassified']

    sections_html = ''
    section_idx = 0

    for dtype in display_order:
        events = device_groups.get(dtype)
        if not events:
            continue

        # Sort by on_start
        events.sort(key=lambda a: a.get('on_start') or '')

        name = display_names.get(dtype, dtype)
        count = len(events)
        section_id = f'device-detail-{section_idx}'
        section_idx += 1

        # Build rows and copyable dates
        rows = ''
        copyable_dates = []

        for i, act in enumerate(events, 1):
            on_start_raw = act.get('on_start')
            off_end_raw = act.get('off_end') or act.get('off_start')

            # Parse timestamps
            date_str = ''
            on_time_str = ''
            off_time_str = ''

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
            if duration >= 60:
                dur_str = f'{duration / 60:.1f} hr'
            else:
                dur_str = f'{duration:.0f} min'

            magnitude = abs(act.get('on_magnitude', 0) or 0)
            phase = act.get('phase', '')
            tag = act.get('tag', '')
            threshold = act.get('threshold', '')

            if date_str and on_time_str:
                copyable_dates.append(f"{date_str} {on_time_str}-{off_time_str}")

            rows += f'''
            <tr>
                <td style="padding: 8px 12px; border-bottom: 1px solid #eee; text-align: center; color: #888;">{i}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #eee;">{date_str}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #eee; text-align: center;">{on_time_str}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #eee; text-align: center;">{off_time_str}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #eee; text-align: center;">{dur_str}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #eee; text-align: right;">{magnitude:,.0f}W</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #eee; text-align: center;">{phase}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 0.85em; color: #888;">{tag}</td>
            </tr>'''

        copyable_text = ', '.join(copyable_dates)

        sections_html += f'''
        <div style="margin-bottom: 20px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px; cursor: pointer;"
                 onclick="var tbl=document.getElementById('{section_id}'); tbl.style.display = tbl.style.display==='none' ? 'block' : 'none'; var arrow=this.querySelector('.toggle-arrow'); arrow.textContent = tbl.style.display==='none' ? '\\u25B6' : '\\u25BC';">
                <span class="toggle-arrow" style="font-size: 0.9em;">&#x25BC;</span>
                <h3 style="margin: 0; color: #2d3748;">{name}</h3>
                <span style="background: #667eea; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.85em;">{count}</span>
            </div>
            <div id="{section_id}">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 8px 12px; text-align: center; width: 40px;">#</th>
                            <th style="padding: 8px 12px; text-align: left;" title="Date of activation">Date</th>
                            <th style="padding: 8px 12px; text-align: center;" title="Time when device turned ON">Start</th>
                            <th style="padding: 8px 12px; text-align: center;" title="Time when device turned OFF">End</th>
                            <th style="padding: 8px 12px; text-align: center;" title="Total duration from ON to OFF">Duration</th>
                            <th style="padding: 8px 12px; text-align: right;" title="ON event magnitude (watts)">Power</th>
                            <th style="padding: 8px 12px; text-align: center;" title="Phase (w1, w2, w3)">Phase</th>
                            <th style="padding: 8px 12px; text-align: left;" title="Match quality tag: magnitude accuracy + duration category">Tag</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows}
                    </tbody>
                </table>
                <div style="margin-top: 8px;">
                    <button onclick="var ta=document.getElementById('{section_id}-dates'); ta.style.display = ta.style.display==='none' ? 'block' : 'none';"
                            style="padding: 4px 12px; border: 1px solid #ccc; border-radius: 4px; background: #f8f9fa; cursor: pointer; font-size: 0.85em;">
                        Copy Dates ({count})
                    </button>
                    <textarea id="{section_id}-dates" readonly
                              style="display: none; width: 100%; margin-top: 5px; padding: 8px; font-size: 0.85em; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; resize: vertical; min-height: 60px;"
                              onclick="this.select();">{copyable_text}</textarea>
                </div>
            </div>
        </div>'''

    return sections_html
