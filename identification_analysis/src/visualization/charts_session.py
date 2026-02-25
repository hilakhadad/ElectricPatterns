"""
Session and device analysis charts for identification reports.

Extracted from identification_charts.py.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

from visualization.charts_device import _parse_iso, _dur_str

# Color constants (consistent with classification_charts.py)
GREEN = '#28a745'
GRAY = '#6c757d'
ORANGE = '#e67e22'
YELLOW = '#eab308'
PURPLE = '#6f42c1'
BLUE = '#007bff'
RED = '#dc3545'
LIGHT_GREEN = '#d4edda'

DEVICE_COLORS = {
    'boiler': '#007bff',
    'three_phase_device': '#6f42c1',
    'central_ac': '#dc3545',
    'regular_ac': '#e67e22',
    'recurring_pattern': '#17a2b8',
    'unknown': '#6c757d',
    'unclassified': '#6c757d',
}

DEVICE_DISPLAY_NAMES = {
    'boiler': 'Water Heater (Boiler)',
    'three_phase_device': '3-Phase Device (Charger?)',
    'central_ac': 'Central AC (Multi-phase)',
    'regular_ac': 'Regular AC (Single-phase)',
    'recurring_pattern': 'Recurring Pattern (Discovered)',
    'unknown': 'Unclassified',
    'unclassified': 'Unclassified',
}

def create_session_overview(sessions: List[Dict]) -> str:
    """
    Session Overview Dashboard: duration and energy breakdown by device type.
    Focuses on segregated minutes and estimated energy rather than session counts.

    Args:
        sessions: List of session dicts from device_sessions JSON.
    """
    logger.debug("Creating session overview chart")
    if not sessions:
        return '<p style="color: #888;">No session data available.</p>'

    # Minutes and energy by device type
    minutes = defaultdict(float)
    energy = defaultdict(float)  # magnitude * duration (watt-minutes)
    for s in sessions:
        dtype = s.get('device_type', 'unknown')
        dur = s.get('duration_minutes', 0) or 0
        minutes[dtype] += dur
        mag = s.get('avg_cycle_magnitude_w', 0) or 0
        energy[dtype] += mag * dur

    total_minutes = sum(minutes.values())
    classified_minutes = sum(m for dt, m in minutes.items() if dt not in ('unknown', 'unclassified'))
    classified_pct = (classified_minutes / total_minutes * 100) if total_minutes > 0 else 0
    total_energy_kwh = sum(energy.values()) / 60000  # watt-minutes to kWh

    # Avg confidence of classified sessions only
    classified_conf_vals = [
        s.get('confidence', 0)
        for s in sessions
        if s.get('confidence') and s.get('device_type', 'unknown') not in ('unknown', 'unclassified')
    ]
    avg_conf = sum(classified_conf_vals) / len(classified_conf_vals) if classified_conf_vals else 0
    conf_color = GREEN if avg_conf >= 0.8 else ORANGE if avg_conf >= 0.5 else RED

    # Device type count (excluding unknown)
    device_types = set(
        s.get('device_type', 'unknown') for s in sessions
        if s.get('device_type', 'unknown') not in ('unknown', 'unclassified')
    )

    cards_html = f'''
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;">
        <div style="background: #f0f4ff; border: 1px solid #c3d4ff; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; font-weight: 700; color: #2d3748;">{total_minutes:,.0f}</div>
            <div style="font-size: 0.85em; color: #666;">Total Segregated Minutes</div>
        </div>
        <div style="background: {LIGHT_GREEN}; border: 1px solid #c3e6cb; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; font-weight: 700; color: {GREEN};">{classified_pct:.0f}%</div>
            <div style="font-size: 0.85em; color: #666;">Classified (by minutes)</div>
        </div>
        <div style="background: #f0f4ff; border: 1px solid #c3d4ff; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; font-weight: 700; color: {conf_color};">{avg_conf:.0%}</div>
            <div style="font-size: 0.85em; color: #666;">Avg Classified Confidence</div>
        </div>
        <div style="background: #f0f4ff; border: 1px solid #c3d4ff; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; font-weight: 700; color: #2d3748;">{len(device_types)}</div>
            <div style="font-size: 0.85em; color: #666;">Device Types</div>
        </div>
    </div>'''

    # Pie chart — minutes by device type
    display_order = ['boiler', 'three_phase_device', 'central_ac', 'regular_ac', 'recurring_pattern', 'unknown']
    min_labels = []
    min_values = []
    min_colors = []
    for dt in display_order:
        if minutes.get(dt, 0) > 0:
            min_labels.append(DEVICE_DISPLAY_NAMES.get(dt, dt))
            min_values.append(round(minutes[dt], 1))
            min_colors.append(DEVICE_COLORS.get(dt, GRAY))

    min_traces = json.dumps([{
        'type': 'pie',
        'labels': min_labels,
        'values': min_values,
        'marker': {'colors': min_colors},
        'textinfo': 'label+percent',
        'textposition': 'auto',
        'hovertemplate': '%{label}: %{value:.0f} min (%{percent})<extra></extra>',
        'hole': 0.35,
    }])
    min_layout = json.dumps({
        'margin': {'l': 20, 'r': 20, 't': 30, 'b': 20},
        'height': 280,
        'title': {'text': 'Minutes by Device Type', 'font': {'size': 14}},
        'paper_bgcolor': 'white',
        'showlegend': True,
        'legend': {'orientation': 'h', 'y': -0.1},
    })

    # Bar chart — energy by device type
    bar_labels = []
    bar_values = []
    bar_colors = []
    for dt in display_order:
        if energy.get(dt, 0) > 0:
            bar_labels.append(DEVICE_DISPLAY_NAMES.get(dt, dt))
            kwh = energy[dt] / 60000  # watt-minutes to kWh
            bar_values.append(round(kwh, 1))
            bar_colors.append(DEVICE_COLORS.get(dt, GRAY))

    bar_traces = json.dumps([{
        'type': 'bar',
        'x': bar_labels,
        'y': bar_values,
        'marker': {'color': bar_colors},
        'text': [f'{v:.1f}' for v in bar_values],
        'textposition': 'auto',
        'hovertemplate': '%{x}: %{y:.1f} kWh<extra></extra>',
    }])
    bar_layout = json.dumps({
        'margin': {'l': 50, 'r': 20, 't': 30, 'b': 60},
        'height': 280,
        'title': {'text': 'Estimated Energy by Device Type', 'font': {'size': 14}},
        'yaxis': {'title': 'kWh (magnitude \u00d7 duration)'},
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#f8f9fa',
    })

    return f'''
    {cards_html}
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
            <div id="minutes_pie"></div>
            <script>Plotly.newPlot('minutes_pie', {min_traces}, {min_layout}, {{displayModeBar:false}});</script>
        </div>
        <div>
            <div id="session_energy_bar"></div>
            <script>Plotly.newPlot('session_energy_bar', {bar_traces}, {bar_layout}, {{displayModeBar:false}});</script>
        </div>
    </div>'''


def create_boiler_analysis(sessions: List[Dict]) -> str:
    """
    Boiler-specific analysis: daily pattern (hour histogram),
    monthly consistency, magnitude stability, phase info.
    """
    logger.debug("Creating boiler analysis chart")
    boilers = [s for s in sessions if s.get('device_type') == 'boiler']
    if not boilers:
        return '<p style="color: #888;">No boiler sessions detected.</p>'

    # Extract hours, minutes, and months from start timestamps
    hour_minutes = defaultdict(float)  # hour -> total segregated minutes
    month_minutes = defaultdict(float)  # 'YYYY-MM' -> total segregated minutes
    magnitudes = []
    phases = defaultdict(int)

    for s in boilers:
        dt = _parse_iso(s.get('start', ''))
        dur = s.get('duration_minutes', 0) or 0
        if dt:
            hour_minutes[dt.hour] += dur
            month_minutes[dt.strftime('%Y-%m')] += dur
        mag = s.get('avg_cycle_magnitude_w', 0)
        if mag:
            magnitudes.append(mag)
        for ph in (s.get('phases') or []):
            phases[ph] += 1

    # Phase info
    dominant_phase = max(phases, key=phases.get) if phases else 'N/A'
    phase_str = ', '.join(f'{ph}: {cnt}' for ph, cnt in sorted(phases.items()))

    # Hour histogram — segregated minutes per start hour
    hour_values = [round(hour_minutes.get(h, 0), 1) for h in range(24)]

    hour_traces = json.dumps([{
        'type': 'bar',
        'x': list(range(24)),
        'y': hour_values,
        'marker': {'color': BLUE},
        'hovertemplate': 'Hour %{x}: %{y:.0f} min<extra></extra>',
    }])
    hour_layout = json.dumps({
        'margin': {'l': 40, 'r': 10, 't': 30, 'b': 40},
        'height': 220,
        'title': {'text': 'Boiler Usage by Hour of Day', 'font': {'size': 13}},
        'xaxis': {'title': 'Hour', 'dtick': 2, 'range': [-0.5, 23.5]},
        'yaxis': {'title': 'Segregated Minutes'},
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#f8f9fa',
    })

    # Monthly bar chart — segregated minutes per month
    sorted_months = sorted(month_minutes.keys())
    month_labels = sorted_months
    month_values = [round(month_minutes[m], 1) for m in sorted_months]

    month_traces = json.dumps([{
        'type': 'bar',
        'x': month_labels,
        'y': month_values,
        'marker': {'color': BLUE},
        'hovertemplate': '%{x}: %{y:.0f} min<extra></extra>',
    }])
    month_layout = json.dumps({
        'margin': {'l': 40, 'r': 10, 't': 30, 'b': 60},
        'height': 220,
        'title': {'text': 'Boiler Minutes per Month', 'font': {'size': 13}},
        'xaxis': {'title': '', 'tickangle': -45},
        'yaxis': {'title': 'Segregated Minutes'},
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#f8f9fa',
    })

    # Summary stats
    avg_mag = sum(magnitudes) / len(magnitudes) if magnitudes else 0
    avg_dur = sum(s.get('duration_minutes', 0) for s in boilers) / len(boilers)
    mag_cv = (
        (sum((m - avg_mag) ** 2 for m in magnitudes) / len(magnitudes)) ** 0.5 / avg_mag
        if avg_mag > 0 and len(magnitudes) > 1 else 0
    )

    summary_html = f'''
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;">
        <div style="background: #f0f4ff; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.5em; font-weight: 700; color: {BLUE};">{len(boilers)}</div>
            <div style="font-size: 0.8em; color: #666;">Total Sessions</div>
        </div>
        <div style="background: #f0f4ff; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.5em; font-weight: 700; color: {BLUE};">{avg_mag:,.0f}W</div>
            <div style="font-size: 0.8em; color: #666;">Avg Power</div>
        </div>
        <div style="background: #f0f4ff; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.5em; font-weight: 700; color: {BLUE};">{_dur_str(avg_dur)}</div>
            <div style="font-size: 0.8em; color: #666;">Avg Duration</div>
        </div>
        <div style="background: #f0f4ff; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.5em; font-weight: 700; color: {BLUE};">{dominant_phase}</div>
            <div style="font-size: 0.8em; color: #666;">Dominant Phase ({phase_str})</div>
        </div>
    </div>'''

    return f'''
    {summary_html}
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
        <div>
            <div id="boiler_hours"></div>
            <script>Plotly.newPlot('boiler_hours', {hour_traces}, {hour_layout}, {{displayModeBar:false}});</script>
        </div>
        <div>
            <div id="boiler_months"></div>
            <script>Plotly.newPlot('boiler_months', {month_traces}, {month_layout}, {{displayModeBar:false}});</script>
        </div>
    </div>
    <div style="margin-top: 8px; font-size: 0.82em; color: #888;">
        Magnitude CV: {mag_cv:.2f} ({"stable" if mag_cv < 0.15 else "moderate" if mag_cv < 0.30 else "variable"})
    </div>'''


def create_ac_analysis(sessions: List[Dict]) -> str:
    """
    AC analysis: central vs regular comparison, cycle distribution, seasonal pattern.
    """
    logger.debug("Creating AC analysis chart")
    central = [s for s in sessions if s.get('device_type') == 'central_ac']
    regular = [s for s in sessions if s.get('device_type') == 'regular_ac']

    if not central and not regular:
        return '<p style="color: #888;">No AC sessions detected.</p>'

    # Summary cards
    cards_html = f'''
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
        <div style="background: #fff5f5; border: 1px solid #fed7d7; border-radius: 8px; padding: 16px;">
            <div style="font-weight: 600; color: {RED}; margin-bottom: 8px;">Central AC (Multi-phase)</div>
            <div style="font-size: 0.9em; color: #555;">
                Sessions: <strong>{len(central)}</strong><br>
                Avg Duration: <strong>{_dur_str(sum(s.get("duration_minutes", 0) for s in central) / len(central)) if central else "N/A"}</strong><br>
                Phase Combos: <strong>{_count_phase_combos(central)}</strong>
            </div>
        </div>
        <div style="background: #fff8f0; border: 1px solid #feebc8; border-radius: 8px; padding: 16px;">
            <div style="font-weight: 600; color: {ORANGE}; margin-bottom: 8px;">Regular AC (Single-phase)</div>
            <div style="font-size: 0.9em; color: #555;">
                Sessions: <strong>{len(regular)}</strong><br>
                Avg Cycles: <strong>{sum(s.get("cycle_count", 0) for s in regular) / len(regular):.1f}</strong><br>
                Avg Power: <strong>{sum(s.get("avg_cycle_magnitude_w", 0) for s in regular) / len(regular):,.0f}W</strong>
            </div>
        </div>
    </div>''' if central or regular else ''

    parts = [cards_html]

    # Seasonal pattern (monthly minutes for both AC types)
    all_ac = central + regular
    month_central = defaultdict(float)
    month_regular = defaultdict(float)

    for s in central:
        dt = _parse_iso(s.get('start', ''))
        dur = s.get('duration_minutes', 0) or 0
        if dt:
            month_central[dt.strftime('%Y-%m')] += dur

    for s in regular:
        dt = _parse_iso(s.get('start', ''))
        dur = s.get('duration_minutes', 0) or 0
        if dt:
            month_regular[dt.strftime('%Y-%m')] += dur

    all_months = sorted(set(list(month_central.keys()) + list(month_regular.keys())))
    if all_months:
        traces = []
        if central:
            traces.append({
                'type': 'bar', 'name': 'Central AC',
                'x': all_months, 'y': [round(month_central.get(m, 0), 1) for m in all_months],
                'marker': {'color': RED},
                'hovertemplate': '%{x}: %{y:.0f} min<extra></extra>',
            })
        if regular:
            traces.append({
                'type': 'bar', 'name': 'Regular AC',
                'x': all_months, 'y': [round(month_regular.get(m, 0), 1) for m in all_months],
                'marker': {'color': ORANGE},
                'hovertemplate': '%{x}: %{y:.0f} min<extra></extra>',
            })

        seasonal_layout = json.dumps({
            'barmode': 'group',
            'margin': {'l': 40, 'r': 10, 't': 30, 'b': 60},
            'height': 250,
            'title': {'text': 'AC Minutes by Month (Seasonal Pattern)', 'font': {'size': 13}},
            'xaxis': {'tickangle': -45},
            'yaxis': {'title': 'Segregated Minutes'},
            'paper_bgcolor': 'white',
            'plot_bgcolor': '#f8f9fa',
            'legend': {'orientation': 'h', 'y': -0.25},
        })

        parts.append(f'''
        <div id="ac_seasonal"></div>
        <script>Plotly.newPlot('ac_seasonal', {json.dumps(traces)}, {seasonal_layout}, {{displayModeBar:false}});</script>''')

    # Cycle count distribution for regular AC
    if regular:
        cycles = [s.get('cycle_count', 0) for s in regular]
        cycle_traces = json.dumps([{
            'type': 'histogram',
            'x': cycles,
            'marker': {'color': ORANGE},
            'nbinsx': max(len(set(cycles)), 5),
            'hovertemplate': '%{x} cycles: %{y} sessions<extra></extra>',
        }])
        cycle_layout = json.dumps({
            'margin': {'l': 40, 'r': 10, 't': 30, 'b': 40},
            'height': 220,
            'title': {'text': 'Regular AC: Cycle Count Distribution', 'font': {'size': 13}},
            'xaxis': {'title': 'Cycles per Session'},
            'yaxis': {'title': 'Count'},
            'paper_bgcolor': 'white',
            'plot_bgcolor': '#f8f9fa',
        })
        parts.append(f'''
        <div id="ac_cycles" style="margin-top: 16px;"></div>
        <script>Plotly.newPlot('ac_cycles', {cycle_traces}, {cycle_layout}, {{displayModeBar:false}});</script>''')

    return '\n'.join(parts)


def _count_phase_combos(sessions: List[Dict]) -> str:
    """Count unique phase combinations in central AC sessions."""
    combos = defaultdict(int)
    for s in sessions:
        phases = tuple(sorted(s.get('phases', [])))
        combos[phases] += 1
    parts = [f'{"+".join(ph)}: {cnt}' for ph, cnt in sorted(combos.items())]
    return ', '.join(parts) if parts else 'N/A'


def create_temporal_heatmap(sessions: List[Dict]) -> str:
    """
    Temporal Patterns: hour (0-23) x day-of-week heatmap of device activations.
    """
    logger.debug("Creating temporal heatmap chart")
    if not sessions:
        return '<p style="color: #888;">No session data for temporal analysis.</p>'

    # Build matrix: rows=days (Mon-Sun), cols=hours (0-23)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    matrix = [[0] * 24 for _ in range(7)]

    for s in sessions:
        dt = _parse_iso(s.get('start', ''))
        if dt:
            matrix[dt.weekday()][dt.hour] += 1

    # Build separate traces per device type for a stacked view
    # But simpler: just one combined heatmap
    traces = json.dumps([{
        'type': 'heatmap',
        'z': matrix,
        'x': list(range(24)),
        'y': day_names,
        'colorscale': [
            [0, '#f7fafc'],
            [0.25, '#c3dafe'],
            [0.5, '#667eea'],
            [0.75, '#5a67d8'],
            [1, '#2d3748'],
        ],
        'hovertemplate': '%{y}, %{x}:00: %{z} sessions<extra></extra>',
        'showscale': True,
        'colorbar': {'title': 'Sessions', 'titleside': 'right'},
    }])
    layout = json.dumps({
        'margin': {'l': 80, 'r': 60, 't': 30, 'b': 40},
        'height': 280,
        'title': {'text': 'Device Activity Heatmap (All Types)', 'font': {'size': 14}},
        'xaxis': {'title': 'Hour of Day', 'dtick': 2},
        'yaxis': {'title': ''},
        'paper_bgcolor': 'white',
    })

    # Per-device breakdown text
    dtype_hour_peaks = {}
    for dtype in ['boiler', 'three_phase_device', 'central_ac', 'regular_ac', 'recurring_pattern']:
        dtype_sessions = [s for s in sessions if s.get('device_type') == dtype]
        if dtype_sessions:
            hour_counts = defaultdict(int)
            for s in dtype_sessions:
                dt = _parse_iso(s.get('start', ''))
                if dt:
                    hour_counts[dt.hour] += 1
            if hour_counts:
                peak_hour = max(hour_counts, key=hour_counts.get)
                dtype_hour_peaks[dtype] = f'{peak_hour:02d}:00 ({hour_counts[peak_hour]} sessions)'

    peaks_html = ''
    if dtype_hour_peaks:
        peaks_html = '<div style="margin-top: 8px; font-size: 0.85em; color: #555;">'
        for dtype, info in dtype_hour_peaks.items():
            name = DEVICE_DISPLAY_NAMES.get(dtype, dtype)
            peaks_html += f'<span style="margin-right: 16px;"><strong>{name}</strong> peak: {info}</span>'
        peaks_html += '</div>'

    return f'''
    <div id="temporal_heatmap"></div>
    <script>Plotly.newPlot('temporal_heatmap', {traces}, {layout}, {{displayModeBar:false}});</script>
    {peaks_html}'''


def create_unclassified_analysis(sessions: List[Dict]) -> str:
    """
    Analysis of unclassified/unknown sessions: power and duration distributions.
    """
    logger.debug("Creating unclassified analysis chart")
    unknown = [s for s in sessions if s.get('device_type') in ('unknown', 'unclassified')]
    if not unknown:
        return '<p style="color: #888;">No unclassified sessions (all sessions were classified).</p>'

    magnitudes = [s.get('avg_cycle_magnitude_w', 0) for s in unknown if s.get('avg_cycle_magnitude_w')]
    durations = [s.get('duration_minutes', 0) for s in unknown if s.get('duration_minutes')]

    parts = []

    # Summary
    parts.append(f'''
    <div style="background: #f7f7f7; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
        <strong>{len(unknown)}</strong> sessions remain unclassified.
        These could be small appliances, partial detections, or devices not covered by current classification rules.
    </div>''')

    grid_parts = []

    # Power distribution
    if magnitudes:
        mag_traces = json.dumps([{
            'type': 'histogram',
            'x': magnitudes,
            'marker': {'color': GRAY},
            'nbinsx': 15,
            'hovertemplate': '%{x:.0f}W: %{y} sessions<extra></extra>',
        }])
        mag_layout = json.dumps({
            'margin': {'l': 40, 'r': 10, 't': 30, 'b': 40},
            'height': 220,
            'title': {'text': 'Power Distribution', 'font': {'size': 13}},
            'xaxis': {'title': 'Watts'},
            'yaxis': {'title': 'Count'},
            'paper_bgcolor': 'white',
            'plot_bgcolor': '#f8f9fa',
        })
        grid_parts.append(f'''
        <div>
            <div id="unclass_power"></div>
            <script>Plotly.newPlot('unclass_power', {mag_traces}, {mag_layout}, {{displayModeBar:false}});</script>
        </div>''')

    # Duration distribution
    if durations:
        dur_traces = json.dumps([{
            'type': 'histogram',
            'x': durations,
            'marker': {'color': GRAY},
            'nbinsx': 15,
            'hovertemplate': '%{x:.0f} min: %{y} sessions<extra></extra>',
        }])
        dur_layout = json.dumps({
            'margin': {'l': 40, 'r': 10, 't': 30, 'b': 40},
            'height': 220,
            'title': {'text': 'Duration Distribution', 'font': {'size': 13}},
            'xaxis': {'title': 'Minutes'},
            'yaxis': {'title': 'Count'},
            'paper_bgcolor': 'white',
            'plot_bgcolor': '#f8f9fa',
        })
        grid_parts.append(f'''
        <div>
            <div id="unclass_duration"></div>
            <script>Plotly.newPlot('unclass_duration', {dur_traces}, {dur_layout}, {{displayModeBar:false}});</script>
        </div>''')

    if grid_parts:
        parts.append(f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">{"".join(grid_parts)}</div>')

    # Phase distribution
    phase_counts = defaultdict(int)
    for s in unknown:
        for ph in (s.get('phases') or []):
            phase_counts[ph] += 1
    if phase_counts:
        phase_info = ', '.join(f'{ph}: {cnt}' for ph, cnt in sorted(phase_counts.items()))
        parts.append(f'<div style="margin-top: 8px; font-size: 0.85em; color: #666;">Phase distribution: {phase_info}</div>')

    return '\n'.join(parts)


