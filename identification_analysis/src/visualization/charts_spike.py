"""
Spike analysis charts for identification reports.

Extracted from identification_charts.py.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

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

def create_spike_analysis(spike_filter: Dict[str, Any]) -> str:
    """
    Create spike filter analysis section with two charts:
    1. Bar chart: event count — spikes vs. kept, by iteration
    2. Bar chart: total minutes — spikes vs. kept, by iteration
    """
    logger.debug("Creating spike analysis chart")
    if not spike_filter or spike_filter.get('spike_count', 0) == 0:
        kept = spike_filter.get('kept_count', 0) if spike_filter else 0
        threshold = spike_filter.get('min_duration_threshold', 3) if spike_filter else 3
        if kept > 0:
            return (
                f'<p style="color: #888;">No transient events filtered. '
                f'All {kept} matched events have duration &ge; {threshold} min.</p>'
            )
        else:
            return (
                f'<p style="color: #888;">No spike filter data available. '
                f'Re-run the pipeline to generate spike statistics.</p>'
            )

    spike_count = spike_filter['spike_count']
    kept_count = spike_filter['kept_count']
    spike_min = spike_filter['spike_total_minutes']
    kept_min = spike_filter['kept_total_minutes']
    threshold = spike_filter.get('min_duration_threshold', 3)
    total_count = spike_count + kept_count
    total_min = spike_min + kept_min

    spike_pct = (spike_count / total_count * 100) if total_count > 0 else 0
    spike_min_pct = (spike_min / total_min * 100) if total_min > 0 else 0

    # Summary cards
    cards_html = f'''
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;">
        <div style="background: #f8d7da; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.3em; font-weight: 700; color: {RED};">{spike_count}</div>
            <div style="font-size: 0.8em; color: #666;">Spikes Filtered</div>
        </div>
        <div style="background: #d4edda; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.3em; font-weight: 700; color: {GREEN};">{kept_count}</div>
            <div style="font-size: 0.8em; color: #666;">Events Kept</div>
        </div>
        <div style="background: #f8d7da; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.3em; font-weight: 700; color: {RED};">{spike_pct:.1f}%</div>
            <div style="font-size: 0.8em; color: #666;">of Event Count</div>
        </div>
        <div style="background: #d4edda; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.3em; font-weight: 700; color: {GREEN};">{spike_min_pct:.1f}%</div>
            <div style="font-size: 0.8em; color: #666;">of Total Minutes</div>
        </div>
    </div>'''

    # Chart 1: Event count by iteration (stacked bar)
    by_iter = spike_filter.get('by_iteration', {})
    if by_iter:
        iters = sorted(by_iter.keys(), key=lambda x: int(x))
        iter_labels = [f'Iter {i}' for i in iters]
        spike_counts = [by_iter[i]['spike_count'] for i in iters]
        kept_counts = [by_iter[i]['kept_count'] for i in iters]

        count_traces = json.dumps([
            {
                'type': 'bar',
                'x': iter_labels,
                'y': spike_counts,
                'name': f'Spikes (<{threshold} min)',
                'marker': {'color': RED, 'opacity': 0.7},
            },
            {
                'type': 'bar',
                'x': iter_labels,
                'y': kept_counts,
                'name': f'Kept (≥{threshold} min)',
                'marker': {'color': GREEN, 'opacity': 0.7},
            },
        ])
        count_layout = json.dumps({
            'barmode': 'stack',
            'margin': {'l': 45, 'r': 20, 't': 30, 'b': 40},
            'height': 250,
            'title': {'text': 'Event Count by Iteration', 'font': {'size': 13}},
            'xaxis': {'title': 'Iteration'},
            'yaxis': {'title': 'Number of Events'},
            'paper_bgcolor': 'white',
            'plot_bgcolor': '#f8f9fa',
            'legend': {'orientation': 'h', 'y': -0.25},
        })

        # Chart 2: Minutes by iteration (stacked bar)
        spike_minutes = [by_iter[i]['spike_minutes'] for i in iters]
        kept_minutes = [by_iter[i]['kept_minutes'] for i in iters]

        min_traces = json.dumps([
            {
                'type': 'bar',
                'x': iter_labels,
                'y': spike_minutes,
                'name': f'Spike minutes (<{threshold} min)',
                'marker': {'color': RED, 'opacity': 0.7},
            },
            {
                'type': 'bar',
                'x': iter_labels,
                'y': kept_minutes,
                'name': f'Kept minutes (≥{threshold} min)',
                'marker': {'color': GREEN, 'opacity': 0.7},
            },
        ])
        min_layout = json.dumps({
            'barmode': 'stack',
            'margin': {'l': 55, 'r': 20, 't': 30, 'b': 40},
            'height': 250,
            'title': {'text': 'Total Minutes by Iteration', 'font': {'size': 13}},
            'xaxis': {'title': 'Iteration'},
            'yaxis': {'title': 'Total Minutes'},
            'paper_bgcolor': 'white',
            'plot_bgcolor': '#f8f9fa',
            'legend': {'orientation': 'h', 'y': -0.25},
        })

        charts_html = f'''
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div>
                <div id="spike_count_chart"></div>
                <script>Plotly.newPlot('spike_count_chart', {count_traces}, {count_layout}, {{displayModeBar:false}});</script>
            </div>
            <div>
                <div id="spike_min_chart"></div>
                <script>Plotly.newPlot('spike_min_chart', {min_traces}, {min_layout}, {{displayModeBar:false}});</script>
            </div>
        </div>'''
    else:
        charts_html = ''

    # Pie chart: Event breakdown by duration category
    # Categories: Spikes (<2 min), Short (2-15 min), Long (>=15 min)
    short_count = spike_filter.get('short_count', 0)
    long_count = spike_filter.get('long_count', 0)
    short_min = spike_filter.get('short_minutes', 0)
    long_min = spike_filter.get('long_minutes', 0)
    long_threshold = spike_filter.get('long_duration_threshold', 15)

    # If short/long aren't in spike_filter, estimate from kept events
    if short_count == 0 and long_count == 0 and kept_count > 0:
        # Fallback: we can't distinguish short/long without per-event data
        short_count = kept_count
        short_min = kept_min

    pie_labels = []
    pie_values_count = []
    pie_values_min = []
    pie_colors = []

    if spike_count > 0:
        pie_labels.append(f'Spikes (<{threshold} min)')
        pie_values_count.append(spike_count)
        pie_values_min.append(spike_min)
        pie_colors.append(RED)
    if short_count > 0:
        pie_labels.append(f'Short ({threshold}-{long_threshold} min)')
        pie_values_count.append(short_count)
        pie_values_min.append(short_min)
        pie_colors.append(ORANGE)
    if long_count > 0:
        pie_labels.append(f'Long (>={long_threshold} min)')
        pie_values_count.append(long_count)
        pie_values_min.append(long_min)
        pie_colors.append(GREEN)

    pie_html = ''
    if len(pie_labels) >= 2:
        count_pie = json.dumps([{
            'type': 'pie', 'labels': pie_labels, 'values': pie_values_count,
            'marker': {'colors': pie_colors}, 'textinfo': 'label+percent+value',
            'textposition': 'auto', 'hole': 0.35,
            'hovertemplate': '%{label}: %{value} events (%{percent})<extra></extra>',
        }])
        count_pie_layout = json.dumps({
            'margin': {'l': 10, 'r': 10, 't': 30, 'b': 10}, 'height': 250,
            'title': {'text': 'By Event Count', 'font': {'size': 13}},
            'paper_bgcolor': 'white', 'showlegend': False,
        })
        min_pie = json.dumps([{
            'type': 'pie', 'labels': pie_labels, 'values': pie_values_min,
            'marker': {'colors': pie_colors}, 'textinfo': 'label+percent+value',
            'textposition': 'auto', 'hole': 0.35,
            'hovertemplate': '%{label}: %{value:.0f} min (%{percent})<extra></extra>',
        }])
        min_pie_layout = json.dumps({
            'margin': {'l': 10, 'r': 10, 't': 30, 'b': 10}, 'height': 250,
            'title': {'text': 'By Total Minutes', 'font': {'size': 13}},
            'paper_bgcolor': 'white', 'showlegend': False,
        })
        pie_html = f'''
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px;">
            <div>
                <div id="spike_pie_count"></div>
                <script>Plotly.newPlot('spike_pie_count', {count_pie}, {count_pie_layout}, {{displayModeBar:false}});</script>
            </div>
            <div>
                <div id="spike_pie_min"></div>
                <script>Plotly.newPlot('spike_pie_min', {min_pie}, {min_pie_layout}, {{displayModeBar:false}});</script>
            </div>
        </div>'''

    # Explanation with clarifications about Unknown and Confidence
    explanation = f'''
    <div style="margin-top: 12px; padding: 12px; background: #fff3cd; border-radius: 6px; font-size: 0.85em; color: #856404;">
        <strong>Why filter spikes?</strong> With 1-minute resolution data and a purely unsupervised approach
        (no ML, only power magnitude), this project focuses on devices with consistent usage patterns.
        Events shorter than {threshold} minutes (microwave, oven, motor starts) cannot be reliably
        identified at this resolution &mdash; no classification rule accepts them
        (boiler &ge;25 min, AC cycle &ge;3 min).
    </div>
    <div style="margin-top: 8px; padding: 12px; background: #e8f4fd; border-radius: 6px; font-size: 0.85em; color: #0c5460;">
        <strong>Spikes vs Unknown:</strong> Spikes are removed <em>before</em> session grouping and classification.
        They are <strong>not</strong> part of the Unknown category. Unknown sessions are events that passed the duration
        filter (&ge;{threshold} min) but did not match any device classification rule.<br>
        <strong>Confidence scope:</strong> Confidence scores are calculated only for <em>classified</em> sessions
        (boiler, central AC, regular AC). Unknown sessions do not have confidence scores and are excluded from confidence statistics.
    </div>'''

    return f'{cards_html}{charts_html}{pie_html}{explanation}'


