"""
Confidence distribution and overview charts.

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

def create_confidence_overview(sessions: List[Dict]) -> str:
    """
    Confidence distribution overview with three histograms:
    1. Classified sessions — confidence score weighted by minutes
    2. Unknown sessions — "Not Boiler" exclusion confidence
    3. Unknown sessions — "Not AC" exclusion confidence
    """
    logger.debug("Creating confidence overview chart")
    if not sessions:
        return '<p style="color: #888;">No session data available.</p>'

    classified = [s for s in sessions if s.get('device_type') not in ('unknown', 'unclassified')]
    unknowns = [s for s in sessions if s.get('device_type') in ('unknown', 'unclassified')]

    if not classified and not unknowns:
        return '<p style="color: #888;">No sessions with confidence scores.</p>'

    parts = []

    # --- Classified confidence (weighted by minutes) ---
    if classified:
        # Tier cards weighted by minutes
        high_min = sum(
            (s.get('duration_minutes', 0) or 0)
            for s in classified if (s.get('confidence', 0) or 0) >= 0.8
        )
        medium_min = sum(
            (s.get('duration_minutes', 0) or 0)
            for s in classified if 0.4 <= (s.get('confidence', 0) or 0) < 0.8
        )
        low_min = sum(
            (s.get('duration_minutes', 0) or 0)
            for s in classified if (s.get('confidence', 0) or 0) < 0.4
        )
        total_classified_min = high_min + medium_min + low_min
        h_pct = (high_min / total_classified_min * 100) if total_classified_min > 0 else 0
        m_pct = (medium_min / total_classified_min * 100) if total_classified_min > 0 else 0
        l_pct = (low_min / total_classified_min * 100) if total_classified_min > 0 else 0

        cards_html = f'''
        <h3 style="color: #2a4365; margin-bottom: 10px;">Classified Sessions</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 16px;">
            <div style="background: #d4edda; border-radius: 6px; padding: 12px; text-align: center;">
                <div style="font-size: 1.3em; font-weight: 700; color: {GREEN};">{h_pct:.0f}%</div>
                <div style="font-size: 0.8em; color: #666;">High (\u22650.80) — {high_min:.0f} min</div>
            </div>
            <div style="background: #fef3cd; border-radius: 6px; padding: 12px; text-align: center;">
                <div style="font-size: 1.3em; font-weight: 700; color: {ORANGE};">{m_pct:.0f}%</div>
                <div style="font-size: 0.8em; color: #666;">Medium (0.40\u20130.80) — {medium_min:.0f} min</div>
            </div>
            <div style="background: #f8d7da; border-radius: 6px; padding: 12px; text-align: center;">
                <div style="font-size: 1.3em; font-weight: 700; color: {RED};">{l_pct:.0f}%</div>
                <div style="font-size: 0.8em; color: #666;">Low (&lt;0.40) — {low_min:.0f} min</div>
            </div>
        </div>'''

        confidences = [s.get('confidence', 0) for s in classified]
        traces = json.dumps([
            {
                'type': 'histogram',
                'x': [c for c in confidences if c >= 0.8],
                'name': 'High',
                'marker': {'color': GREEN},
                'xbins': {'start': 0, 'end': 1, 'size': 0.05},
            },
            {
                'type': 'histogram',
                'x': [c for c in confidences if 0.4 <= c < 0.8],
                'name': 'Medium',
                'marker': {'color': ORANGE},
                'xbins': {'start': 0, 'end': 1, 'size': 0.05},
            },
            {
                'type': 'histogram',
                'x': [c for c in confidences if c < 0.4],
                'name': 'Low',
                'marker': {'color': RED},
                'xbins': {'start': 0, 'end': 1, 'size': 0.05},
            },
        ])
        layout = json.dumps({
            'barmode': 'stack',
            'margin': {'l': 40, 'r': 20, 't': 10, 'b': 40},
            'height': 220,
            'xaxis': {'title': 'Confidence Score', 'range': [0, 1]},
            'yaxis': {'title': 'Sessions'},
            'paper_bgcolor': 'white',
            'plot_bgcolor': '#f8f9fa',
            'legend': {'orientation': 'h', 'y': -0.2},
            'shapes': [
                {'type': 'line', 'x0': 0.8, 'x1': 0.8, 'y0': 0, 'y1': 1, 'yref': 'paper',
                 'line': {'color': '#888', 'dash': 'dot', 'width': 1}},
                {'type': 'line', 'x0': 0.4, 'x1': 0.4, 'y0': 0, 'y1': 1, 'yref': 'paper',
                 'line': {'color': '#888', 'dash': 'dot', 'width': 1}},
            ],
        })
        parts.append(f'''
        {cards_html}
        <div id="confidence_hist"></div>
        <script>Plotly.newPlot('confidence_hist', {traces}, {layout}, {{displayModeBar:false}});</script>''')

    # --- Unknown sessions: exclusion confidence histograms ---
    if unknowns:
        not_boiler_scores = []
        not_ac_scores = []
        for s in unknowns:
            bd = s.get('confidence_breakdown', {}) or {}
            nb = bd.get('not_boiler')
            na = bd.get('not_ac')
            if nb is not None:
                not_boiler_scores.append(nb)
            if na is not None:
                not_ac_scores.append(na)

        if not_boiler_scores or not_ac_scores:
            parts.append(f'''
            <h3 style="color: #2a4365; margin: 24px 0 8px 0;">Unknown Sessions — Exclusion Confidence</h3>
            <p style="color: #666; font-size: 0.85em; margin-bottom: 12px;">
                How confidently each unknown session was excluded from known device types.
                High "Not Boiler" = clearly not a boiler. Low = borderline, might be a boiler missed by rules.
            </p>''')

            hist_parts = []

            if not_boiler_scores:
                nb_traces = json.dumps([{
                    'type': 'histogram',
                    'x': not_boiler_scores,
                    'marker': {'color': '#6c757d'},
                    'xbins': {'start': 0, 'end': 1, 'size': 0.05},
                    'hovertemplate': 'Score %{{x:.2f}}: %{{y}} sessions<extra></extra>',
                }])
                nb_layout = json.dumps({
                    'margin': {'l': 40, 'r': 20, 't': 10, 'b': 40},
                    'height': 200,
                    'xaxis': {'title': 'Not Boiler Score', 'range': [0, 1]},
                    'yaxis': {'title': 'Sessions'},
                    'paper_bgcolor': 'white',
                    'plot_bgcolor': '#f8f9fa',
                })
                hist_parts.append(f'''
                <div>
                    <div id="not_boiler_hist"></div>
                    <script>Plotly.newPlot('not_boiler_hist', {nb_traces}, {nb_layout}, {{displayModeBar:false}});</script>
                </div>''')

            if not_ac_scores:
                na_traces = json.dumps([{
                    'type': 'histogram',
                    'x': not_ac_scores,
                    'marker': {'color': '#6c757d'},
                    'xbins': {'start': 0, 'end': 1, 'size': 0.05},
                    'hovertemplate': 'Score %{{x:.2f}}: %{{y}} sessions<extra></extra>',
                }])
                na_layout = json.dumps({
                    'margin': {'l': 40, 'r': 20, 't': 10, 'b': 40},
                    'height': 200,
                    'xaxis': {'title': 'Not AC Score', 'range': [0, 1]},
                    'yaxis': {'title': 'Sessions'},
                    'paper_bgcolor': 'white',
                    'plot_bgcolor': '#f8f9fa',
                })
                hist_parts.append(f'''
                <div>
                    <div id="not_ac_hist"></div>
                    <script>Plotly.newPlot('not_ac_hist', {na_traces}, {na_layout}, {{displayModeBar:false}});</script>
                </div>''')

            cols = len(hist_parts)
            parts.append(f'<div style="display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 20px;">')
            parts.extend(hist_parts)
            parts.append('</div>')

    if not parts:
        return '<p style="color: #888;">No confidence data available.</p>'

    return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Spike Filter Analysis
# ---------------------------------------------------------------------------

