"""
Device activations detail and summary table charts.

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

def _parse_iso(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp string, returning None on failure."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts))
    except (ValueError, TypeError):
        return None


def _dur_str(minutes: float) -> str:
    if minutes >= 60:
        return f'{minutes / 60:.1f} hr'
    return f'{minutes:.0f} min'


# ---------------------------------------------------------------------------
# Existing M2 functions (from dynamic_report_charts.py)
# ---------------------------------------------------------------------------

def create_device_summary_table(metrics: Dict[str, Any]) -> str:
    """
    Create HTML table summarizing detected devices.

    Columns: Device Type | Count | Avg Power | Avg Duration | % of Explained
    """
    logger.debug("Creating device summary table chart")
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

    rows = ''
    for dtype in ['boiler', 'three_phase_device', 'central_ac', 'regular_ac', 'recurring_pattern', 'unclassified']:
        info = types.get(dtype)
        if info is None:
            continue

        name = DEVICE_DISPLAY_NAMES.get(dtype, dtype)
        count = info.get('count', 0)
        avg_mag = info.get('avg_magnitude', 0)
        avg_dur = info.get('avg_duration', 0)
        energy_pct = info.get('energy_pct', 0)
        dur_s = _dur_str(avg_dur)
        bar_width = min(energy_pct, 100)

        rows += f'''
        <tr>
            <td style="padding: 12px 15px; border-bottom: 1px solid #eee; font-weight: 600;">{name}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid #eee; text-align: center;">{count}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid #eee; text-align: center;">{avg_mag:,}W</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid #eee; text-align: center;">{dur_s}</td>
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


def _parse_session_row(session: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a session (from device_sessions.json) into display-ready fields."""
    start_raw = session.get('start', '')
    end_raw = session.get('end', '')

    date_str = start_str = end_str = ''

    if start_raw:
        dt = _parse_iso(str(start_raw))
        if dt:
            date_str = dt.strftime('%Y-%m-%d')
            start_str = dt.strftime('%H:%M')
        else:
            date_str = str(start_raw)[:10]
            start_str = str(start_raw)[11:16] if len(str(start_raw)) > 16 else ''

    if end_raw:
        dt = _parse_iso(str(end_raw))
        if dt:
            end_str = dt.strftime('%H:%M')
        else:
            end_str = str(end_raw)[11:16] if len(str(end_raw)) > 16 else ''

    duration = session.get('duration_minutes', 0) or 0
    magnitude = session.get('avg_cycle_magnitude_w', 0) or 0
    confidence = session.get('confidence', 0) or 0
    cycle_count = session.get('cycle_count', 0) or 0
    phases = session.get('phases', [])
    phase_presence = session.get('phase_presence', {})
    phase_magnitudes = session.get('phase_magnitudes', {})

    # Single phase string for boiler/regular_ac display
    phase = phases[0] if len(phases) == 1 else ', '.join(phases)

    # Confidence breakdown (used for unknown sessions: not_boiler, not_ac)
    breakdown = session.get('confidence_breakdown', {})
    reason = session.get('classification_reason', '')

    return {
        'date': date_str, 'start': start_str, 'end': end_str,
        'duration': duration, 'dur_str': _dur_str(duration),
        'magnitude': magnitude, 'phase': phase,
        'device_type': session.get('device_type', 'unknown'),
        'confidence': confidence,
        'confidence_breakdown': breakdown,
        'classification_reason': reason,
        'cycle_count': cycle_count,
        'session_id': session.get('session_id', ''),
        'on_start_iso': str(start_raw) if start_raw else '',
        'off_end_iso': str(end_raw) if end_raw else '',
        'w1': phase_presence.get('w1', 'X'),
        'w2': phase_presence.get('w2', 'X'),
        'w3': phase_presence.get('w3', 'X'),
        'phase_magnitudes': phase_magnitudes,
        'constituent_events': session.get('constituent_events', []),
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
            dt = _parse_iso(earliest)
            if dt:
                date_str = dt.strftime('%Y-%m-%d')
                start_str = dt.strftime('%H:%M')
            else:
                date_str = earliest[:10]
                start_str = earliest[11:16] if len(earliest) > 16 else ''

        if iso_ends:
            latest = max(iso_ends)
            dt = _parse_iso(latest)
            if dt:
                end_str = dt.strftime('%H:%M')
            else:
                end_str = latest[11:16] if len(latest) > 16 else ''

        if iso_starts and iso_ends:
            t_start = _parse_iso(min(iso_starts))
            t_end = _parse_iso(max(iso_ends))
            if t_start and t_end:
                session_dur = (t_end - t_start).total_seconds() / 60

        if not date_str:
            date_str = group[0].get('date', '')
        if not start_str:
            start_str = group[0].get('start', '')
        if not end_str:
            end_str = group[0].get('end', '')

        dur_s = _dur_str(session_dur)
        avg_mag = sum(e['magnitude'] for e in group) / len(group) if group else 0
        conf_val = max((e.get('confidence', 0) for e in group), default=0)

        result.append({
            'date': date_str, 'start': start_str, 'end': end_str,
            'duration': session_dur, 'dur_str': dur_s,
            'magnitude': avg_mag,
            'w1': 'V' if 'w1' in phases_present else 'X',
            'w2': 'V' if 'w2' in phases_present else 'X',
            'w3': 'V' if 'w3' in phases_present else 'X',
            'cycle_count': len(group),
            'confidence': conf_val,
            'on_start_iso': min(iso_starts) if iso_starts else '',
            'off_end_iso': max(iso_ends) if iso_ends else '',
        })

    result.sort(key=lambda r: r['date'] + r['start'])
    return result


def create_device_activations_detail(sessions: List[Dict[str, Any]], house_id: str = '',
                                     summarized_data=None,
                                     all_match_intervals: Optional[Dict[str, list]] = None) -> str:
    """
    Create detailed device sessions tables grouped by device_type.

    Each row represents a full session (not individual ON→OFF pairs).
    Includes sortable columns, per-type Copy Dates, and a Copy All button.
    When summarized_data is provided, each row is clickable to expand a 3x3
    power chart grid (original/remaining/segregated x w1/w2/w3).
    all_match_intervals: {phase: [(start, end, magnitude, duration), ...]}
        for rendering individual match rectangles in the segregated chart.
    """
    logger.debug("Creating device activations detail chart")
    if not sessions:
        return '<p style="color: #888;">No session data available.</p>'

    MIN_DURATION = {
        'boiler': 15, 'central_ac': 5, 'regular_ac': 3, 'recurring_pattern': 0, 'unknown': 0,
    }

    device_groups = {}
    all_copyable = []
    for session in sessions:
        parsed = _parse_session_row(session)
        dtype = parsed['device_type']
        min_dur = MIN_DURATION.get(dtype, 0)
        if parsed['duration'] < min_dur:
            continue
        device_groups.setdefault(dtype, []).append(parsed)
        # Only collect dates for classified (non-unknown) device types
        if dtype != 'unknown' and parsed['date'] and parsed['start']:
            all_copyable.append(f"{parsed['date']} {parsed['start']}-{parsed['end']}")

    if not any(device_groups.values()):
        return '<p style="color: #888;">No high-confidence device activations found.</p>'

    # Chart expansion support
    has_charts = (summarized_data is not None and hasattr(summarized_data, 'empty')
                  and not summarized_data.empty)
    if has_charts:
        import pandas as pd
        if not pd.api.types.is_datetime64_any_dtype(summarized_data['timestamp']):
            summarized_data = summarized_data.copy()
            summarized_data['timestamp'] = pd.to_datetime(summarized_data['timestamp'])
        summarized_data = summarized_data.sort_values('timestamp').reset_index(drop=True)
    all_chart_data = {}
    global_act_idx = 0

    # Display all device types including unknown
    display_order = ['boiler', 'three_phase_device', 'central_ac', 'regular_ac', 'recurring_pattern', 'unknown']

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

        name = DEVICE_DISPLAY_NAMES.get(dtype, dtype)
        min_dur = MIN_DURATION.get(dtype, 0)
        filter_note = f' (\u2265{min_dur} min)' if min_dur > 0 else ''
        section_id = f'device-detail-{section_idx}'
        section_idx += 1

        _cell = 'padding: 5px 8px; border-bottom: 1px solid #eee;'
        _th = 'padding: 5px 8px;'
        rows = ''
        copyable_dates = []

        if dtype == 'central_ac':
            events.sort(key=lambda r: r['date'] + r['start'])
            count = len(events)
            for r in events:
                if r['date'] and r['start']:
                    copyable_dates.append(f"{r['date']} {r['start']}-{r['end']}")
            for i, r in enumerate(events, 1):
                act_idx = global_act_idx
                global_act_idx += 1
                conf_val = r.get('confidence', 0)
                conf_pct = f'{conf_val:.0%}' if conf_val else '-'
                conf_color = '#48bb78' if conf_val >= 0.8 else '#ecc94b' if conf_val >= 0.6 else '#fc8181' if conf_val > 0 else '#ccc'
                click_attr = f' style="cursor:pointer;" onclick="toggleActChart({act_idx})"' if has_charts else ''
                rows += f'''
            <tr{click_attr}>
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
                if has_charts:
                    rows += _build_chart_row_html(act_idx, 11)
                    session_phases = r.get('phase_magnitudes', {})
                    if not session_phases:
                        mag = r.get('magnitude', 0)
                        for ph in ['w1', 'w2', 'w3']:
                            if r.get(ph) == 'V':
                                session_phases[ph] = mag
                    cd = _extract_chart_window(
                        summarized_data, r.get('on_start_iso', ''),
                        r.get('off_end_iso', ''), session_phases, dtype,
                        all_match_intervals=all_match_intervals,
                        constituent_events=r.get('constituent_events', []),
                    )
                    if cd:
                        all_chart_data[str(act_idx)] = cd

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
        elif dtype == 'unknown':
            # Unknown sessions: show Not Boiler / Not AC instead of single Confidence
            # Split into short (<5 min) and normal (>=5 min) — short ones get a summary line
            SHORT_THRESHOLD = 5  # minutes
            short_events = [r for r in events if r['duration'] < SHORT_THRESHOLD]
            normal_events = [r for r in events if r['duration'] >= SHORT_THRESHOLD]
            normal_events.sort(key=lambda r: r['date'] + r['start'])
            count = len(normal_events)  # table row count (excluding short)
            if short_events:
                filter_note = f' (\u2265{SHORT_THRESHOLD} min, +{len(short_events)} short)'
            for r in normal_events:
                if r['date'] and r['start']:
                    copyable_dates.append(f"{r['date']} {r['start']}-{r['end']}")

            # Summary line for short events
            if short_events:
                n_short = len(short_events)
                total_dur = sum(r['duration'] for r in short_events)
                total_energy_wh = sum(
                    r['magnitude'] * r['duration'] / 60 for r in short_events
                )
                total_energy_kwh = total_energy_wh / 1000
                rows += f'''
            <tr style="background: #f7f7f7; font-style: italic; color: #888;">
                <td colspan="10" style="{_cell}">
                    {n_short} unclassified events shorter than {SHORT_THRESHOLD} min
                    &mdash; total duration: {total_dur:.0f} min,
                    total energy: {total_energy_kwh:.1f} kWh
                </td>
            </tr>'''

            for i, r in enumerate(normal_events, 1):
                act_idx = global_act_idx
                global_act_idx += 1
                bd = r.get('confidence_breakdown', {})
                not_boiler = bd.get('not_boiler')
                not_ac = bd.get('not_ac')
                reason = r.get('classification_reason', '')
                # Format not_boiler / not_ac cells
                if not_boiler is not None:
                    nb_pct = f'{not_boiler:.0%}'
                    nb_color = '#48bb78' if not_boiler >= 0.7 else '#ecc94b' if not_boiler >= 0.4 else '#fc8181'
                else:
                    nb_pct = '-'
                    nb_color = '#ccc'
                if not_ac is not None:
                    na_pct = f'{not_ac:.0%}'
                    na_color = '#48bb78' if not_ac >= 0.7 else '#ecc94b' if not_ac >= 0.4 else '#fc8181'
                else:
                    na_pct = '-'
                    na_color = '#ccc'
                click_attr = f' style="cursor:pointer;" onclick="toggleActChart({act_idx})"' if has_charts else ''
                reason_escaped = reason.replace('"', '&quot;')
                rows += f'''
            <tr{click_attr} title="{reason_escaped}">
                <td style="{_cell} text-align: center; color: #aaa; font-size: 0.85em;">{i}</td>
                <td style="{_cell}" data-value="{r['date']}">{r['date']}</td>
                <td style="{_cell} text-align: center;">{r['start']}</td>
                <td style="{_cell} text-align: center;">{r['end']}</td>
                <td style="{_cell} text-align: center;" data-value="{r['duration']}">{r['dur_str']}</td>
                <td style="{_cell} text-align: right;" data-value="{r['magnitude']}">{r['magnitude']:,.0f}W</td>
                <td style="{_cell} text-align: center;">{r['phase']}</td>
                <td style="{_cell} text-align: center;">{r['cycle_count']}</td>
                <td style="{_cell} text-align: center; color: {nb_color}; font-weight: 600;" data-value="{not_boiler or 0}">{nb_pct}</td>
                <td style="{_cell} text-align: center; color: {na_color}; font-weight: 600;" data-value="{not_ac or 0}">{na_pct}</td>
            </tr>'''
                if has_charts:
                    rows += _build_chart_row_html(act_idx, 10)
                    session_phases = r.get('phase_magnitudes', {})
                    if not session_phases and r.get('phase'):
                        session_phases = {r['phase']: r['magnitude']}
                    cd = _extract_chart_window(
                        summarized_data, r.get('on_start_iso', ''), r.get('off_end_iso', ''),
                        session_phases, dtype,
                        all_match_intervals=all_match_intervals,
                        constituent_events=r.get('constituent_events', []),
                    )
                    if cd:
                        all_chart_data[str(act_idx)] = cd

            table_header = f'''
                        <tr style="background: #f8f9fa;">
                            <th style="{_th} text-align: center; width: 35px;">#</th>
                            <th style="{_th} text-align: left; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 1, 'str')">Date &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center;">Start</th>
                            <th style="{_th} text-align: center;">End</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 4, 'num')">Duration &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: right; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 5, 'num')">Power &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center;">Phase</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 7, 'num')">Cycles &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 8, 'num')">Not Boiler &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 9, 'num')">Not AC &#x25B4;&#x25BE;</th>
                        </tr>'''
        elif dtype == 'three_phase_device':
            # 3-Phase Device (Charger?) — same layout as central_ac with w1/w2/w3 columns
            events.sort(key=lambda r: r['date'] + r['start'])
            count = len(events)
            for r in events:
                if r['date'] and r['start']:
                    copyable_dates.append(f"{r['date']} {r['start']}-{r['end']}")
            for i, r in enumerate(events, 1):
                act_idx = global_act_idx
                global_act_idx += 1
                conf_val = r.get('confidence', 0)
                conf_pct = f'{conf_val:.0%}' if conf_val else '-'
                conf_color = '#48bb78' if conf_val >= 0.8 else '#ecc94b' if conf_val >= 0.6 else '#fc8181' if conf_val > 0 else '#ccc'
                click_attr = f' style="cursor:pointer;" onclick="toggleActChart({act_idx})"' if has_charts else ''
                rows += f'''
            <tr{click_attr}>
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
                if has_charts:
                    rows += _build_chart_row_html(act_idx, 11)
                    session_phases = r.get('phase_magnitudes', {})
                    if not session_phases:
                        mag = r.get('magnitude', 0)
                        for ph in ['w1', 'w2', 'w3']:
                            if r.get(ph) == 'V':
                                session_phases[ph] = mag
                    cd = _extract_chart_window(
                        summarized_data, r.get('on_start_iso', ''),
                        r.get('off_end_iso', ''), session_phases, dtype,
                        all_match_intervals=all_match_intervals,
                        constituent_events=r.get('constituent_events', []),
                    )
                    if cd:
                        all_chart_data[str(act_idx)] = cd

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
        elif dtype == 'recurring_pattern':
            # Recurring patterns — separate table per pattern_id
            # Group events by pattern_id from confidence_breakdown
            pattern_groups = {}
            for r in events:
                pid = r.get('confidence_breakdown', {}).get('pattern_id', 0)
                pattern_groups.setdefault(pid, []).append(r)

            for pid in sorted(pattern_groups.keys()):
                p_events = pattern_groups[pid]
                p_events.sort(key=lambda r: r['date'] + r['start'])
                p_count = len(p_events)
                p_section_id = f'device-detail-{section_idx}'
                section_idx += 1

                # Extract cluster characteristics from first session's breakdown
                bd = p_events[0].get('confidence_breakdown', {})
                avg_mag = bd.get('avg_magnitude_w', 0)
                avg_dur = bd.get('avg_duration_min', 0)
                cluster_conf = p_events[0].get('confidence', 0)
                phase_set = set(r['phase'] for r in p_events if r.get('phase'))
                phase_str = ', '.join(sorted(phase_set)) if phase_set else '?'

                global_name = bd.get('global_pattern_name')
                if global_name:
                    p_name = f'{global_name} — Recurring Pattern #{pid}'
                else:
                    p_name = f'Recurring Pattern #{pid}'
                global_desc = bd.get('global_descriptive_name', '')
                p_subtitle = f'~{avg_mag:,.0f}W, ~{avg_dur:.0f}min, phase {phase_str}, confidence {cluster_conf:.0%}'
                if global_desc:
                    p_subtitle = f'{global_desc} | {p_subtitle}'

                p_rows = ''
                p_copyable = []
                for r in p_events:
                    if r['date'] and r['start']:
                        p_copyable.append(f"{r['date']} {r['start']}-{r['end']}")
                        all_copyable.append(f"{r['date']} {r['start']}-{r['end']}")
                for i, r in enumerate(p_events, 1):
                    act_idx = global_act_idx
                    global_act_idx += 1
                    conf_val = r.get('confidence', 0)
                    conf_pct = f'{conf_val:.0%}' if conf_val else '-'
                    conf_color = '#48bb78' if conf_val >= 0.8 else '#ecc94b' if conf_val >= 0.6 else '#fc8181' if conf_val > 0 else '#ccc'
                    click_attr = f' style="cursor:pointer;" onclick="toggleActChart({act_idx})"' if has_charts else ''
                    p_rows += f'''
            <tr{click_attr}>
                <td style="{_cell} text-align: center; color: #aaa; font-size: 0.85em;">{i}</td>
                <td style="{_cell}" data-value="{r['date']}">{r['date']}</td>
                <td style="{_cell} text-align: center;">{r['start']}</td>
                <td style="{_cell} text-align: center;">{r['end']}</td>
                <td style="{_cell} text-align: center;" data-value="{r['duration']}">{r['dur_str']}</td>
                <td style="{_cell} text-align: right;" data-value="{r['magnitude']}">{r['magnitude']:,.0f}W</td>
                <td style="{_cell} text-align: center;">{r['phase']}</td>
                <td style="{_cell} text-align: center;">{r['cycle_count']}</td>
                <td style="{_cell} text-align: center; color: {conf_color}; font-weight: 600;" data-value="{conf_val}">{conf_pct}</td>
            </tr>'''
                    if has_charts:
                        p_rows += _build_chart_row_html(act_idx, 9)
                        session_phases = r.get('phase_magnitudes', {})
                        if not session_phases and r.get('phase'):
                            session_phases = {r['phase']: r['magnitude']}
                        cd = _extract_chart_window(
                            summarized_data, r.get('on_start_iso', ''), r.get('off_end_iso', ''),
                            session_phases, dtype,
                            all_match_intervals=all_match_intervals,
                            constituent_events=r.get('constituent_events', []),
                        )
                        if cd:
                            all_chart_data[str(act_idx)] = cd

                p_header = f'''
                        <tr style="background: #f8f9fa;">
                            <th style="{_th} text-align: center; width: 35px;">#</th>
                            <th style="{_th} text-align: left; cursor: pointer;" onclick="sortDeviceTable('{p_section_id}-table', 1, 'str')">Date &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center;">Start</th>
                            <th style="{_th} text-align: center;">End</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{p_section_id}-table', 4, 'num')">Duration &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: right; cursor: pointer;" onclick="sortDeviceTable('{p_section_id}-table', 5, 'num')">Power &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center;">Phase</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{p_section_id}-table', 7, 'num')">Cycles &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{p_section_id}-table', 8, 'num')">Confidence &#x25B4;&#x25BE;</th>
                        </tr>'''

                p_copyable_text = ', '.join(p_copyable)
                sections_html += f'''
        <div style="margin-bottom: 15px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px; cursor: pointer;"
                 onclick="var tbl=document.getElementById('{p_section_id}'); tbl.style.display = tbl.style.display==='none' ? 'block' : 'none'; var arrow=this.querySelector('.toggle-arrow'); arrow.textContent = tbl.style.display==='none' ? '\\u25B6' : '\\u25BC';">
                <span class="toggle-arrow" style="font-size: 0.85em;">&#x25BC;</span>
                <strong style="color: #17a2b8; font-size: 0.95em;">{p_name}</strong>
                <span style="background: #17a2b8; color: white; padding: 1px 8px; border-radius: 10px; font-size: 0.8em;">{p_count}</span>
            </div>
            <div style="margin-left: 20px; margin-bottom: 6px; font-size: 0.82em; color: #666;">{p_subtitle}</div>
            <div id="{p_section_id}">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.85em;" id="{p_section_id}-table">
                    <thead>
                        {p_header}
                    </thead>
                    <tbody>
                        {p_rows}
                    </tbody>
                </table>
                <div style="margin-top: 5px;">
                    <button onclick="var ta=document.getElementById('{p_section_id}-dates'); ta.style.display = ta.style.display==='none' ? 'block' : 'none';"
                            style="padding: 3px 10px; border: 1px solid #ccc; border-radius: 4px; background: #f8f9fa; cursor: pointer; font-size: 0.8em;">
                        Copy Dates ({p_count})
                    </button>
                    <textarea id="{p_section_id}-dates" readonly
                              style="display: none; width: 100%; margin-top: 4px; padding: 6px; font-size: 0.8em; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; resize: vertical; min-height: 50px;"
                              onclick="this.select();">{p_copyable_text}</textarea>
                </div>
            </div>
        </div>'''

            # All sub-sections built directly — skip standard section assembly
            continue

        else:
            # Boiler / Regular AC
            events.sort(key=lambda r: r['date'] + r['start'])
            count = len(events)
            for r in events:
                if r['date'] and r['start']:
                    copyable_dates.append(f"{r['date']} {r['start']}-{r['end']}")
            for i, r in enumerate(events, 1):
                act_idx = global_act_idx
                global_act_idx += 1
                conf_val = r.get('confidence', 0)
                conf_pct = f'{conf_val:.0%}' if conf_val else '-'
                conf_color = '#48bb78' if conf_val >= 0.8 else '#ecc94b' if conf_val >= 0.6 else '#fc8181' if conf_val > 0 else '#ccc'
                click_attr = f' style="cursor:pointer;" onclick="toggleActChart({act_idx})"' if has_charts else ''
                rows += f'''
            <tr{click_attr}>
                <td style="{_cell} text-align: center; color: #aaa; font-size: 0.85em;">{i}</td>
                <td style="{_cell}" data-value="{r['date']}">{r['date']}</td>
                <td style="{_cell} text-align: center;">{r['start']}</td>
                <td style="{_cell} text-align: center;">{r['end']}</td>
                <td style="{_cell} text-align: center;" data-value="{r['duration']}">{r['dur_str']}</td>
                <td style="{_cell} text-align: right;" data-value="{r['magnitude']}">{r['magnitude']:,.0f}W</td>
                <td style="{_cell} text-align: center;">{r['phase']}</td>
                <td style="{_cell} text-align: center;">{r['cycle_count']}</td>
                <td style="{_cell} text-align: center; color: {conf_color}; font-weight: 600;" data-value="{conf_val}">{conf_pct}</td>
            </tr>'''
                if has_charts:
                    rows += _build_chart_row_html(act_idx, 9)
                    session_phases = r.get('phase_magnitudes', {})
                    if not session_phases and r.get('phase'):
                        session_phases = {r['phase']: r['magnitude']}
                    cd = _extract_chart_window(
                        summarized_data, r.get('on_start_iso', ''), r.get('off_end_iso', ''),
                        session_phases, dtype,
                        all_match_intervals=all_match_intervals,
                        constituent_events=r.get('constituent_events', []),
                    )
                    if cd:
                        all_chart_data[str(act_idx)] = cd

            table_header = f'''
                        <tr style="background: #f8f9fa;">
                            <th style="{_th} text-align: center; width: 35px;">#</th>
                            <th style="{_th} text-align: left; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 1, 'str')">Date &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center;">Start</th>
                            <th style="{_th} text-align: center;">End</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 4, 'num')">Duration &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: right; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 5, 'num')">Power &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center;">Phase</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 7, 'num')">Cycles &#x25B4;&#x25BE;</th>
                            <th style="{_th} text-align: center; cursor: pointer;" onclick="sortDeviceTable('{section_id}-table', 8, 'num')">Confidence &#x25B4;&#x25BE;</th>
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

    # Add expandable chart JavaScript if chart data is available
    if has_charts and all_chart_data:
        sections_html += _build_activation_charts_script(all_chart_data)
        sections_html += '''
        <div style="margin-top: 8px; padding: 8px 12px; background: #e8f4fd; border: 1px solid #bee3f8; border-radius: 6px; font-size: 0.82em; color: #2a4365;">
            <strong>Tip:</strong> Click any activation row to expand a 3&times;3 power chart grid
            showing Original / Remaining / Segregated power per phase (w1, w2, w3).
        </div>'''

    return sections_html


# ---------------------------------------------------------------------------
# NEW session-based charts for the M2 Identification Report
# ---------------------------------------------------------------------------



def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


# ---------------------------------------------------------------------------
# Expandable 3x3 power charts per activation row
# ---------------------------------------------------------------------------

def _extract_chart_window(summarized_data, on_start_iso: str, off_end_iso: str,
                          session_phases: dict, device_type: str,
                          margin_minutes: int = 60,
                          all_match_intervals: Optional[Dict[str, list]] = None,
                          constituent_events: Optional[list] = None) -> Optional[dict]:
    """Extract power data window for a single activation's expandable charts.

    Args:
        summarized_data: DataFrame with timestamp, original_wN, remaining_wN
        on_start_iso: ISO start timestamp
        off_end_iso: ISO end timestamp
        session_phases: dict mapping phase -> magnitude (e.g. {'w2': 2500})
        device_type: device type string for color lookup
        margin_minutes: minutes of context before/after
        all_match_intervals: {phase: [(start, end, magnitude, duration), ...]}
            All matches from all iterations for building match rectangles.
        constituent_events: List of session's own events for identifying
            which matches belong to this session.

    Returns:
        Compact dict with chart data, or None if no data available.
    """
    import pandas as pd

    if summarized_data is None or not on_start_iso or not off_end_iso:
        return None

    try:
        start = pd.Timestamp(on_start_iso)
        end = pd.Timestamp(off_end_iso)
    except (ValueError, TypeError):
        return None

    margin = pd.Timedelta(minutes=margin_minutes)
    mask = ((summarized_data['timestamp'] >= start - margin) &
            (summarized_data['timestamp'] <= end + margin))
    window = summarized_data[mask]

    if window.empty or len(window) < 3:
        return None

    result = {
        'ts': window['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        'ss': start.strftime('%Y-%m-%d %H:%M'),
        'se': end.strftime('%Y-%m-%d %H:%M'),
        'sp': session_phases,
        'dc': DEVICE_COLORS.get(device_type, GRAY),
    }

    for phase in ['w1', 'w2', 'w3']:
        orig_col = f'original_{phase}'
        remain_col = f'remaining_{phase}'
        if orig_col in window.columns:
            result[f'o_{phase}'] = [
                int(round(v)) if pd.notna(v) else None
                for v in window[orig_col]
            ]
        if remain_col in window.columns:
            result[f'r_{phase}'] = [
                int(round(v)) if pd.notna(v) else None
                for v in window[remain_col]
            ]

    # Add match rectangles for each phase (individual match shapes)
    if all_match_intervals:
        # Build lookup dicts from session events for classification and profiles
        ses_keys = set()
        ses_profiles = {}  # (phase, normalized_start) -> power_profile
        for ce in (constituent_events or []):
            ce_start = ce.get('on_start', '')
            ce_phase = ce.get('phase', '')
            if ce_start and ce_phase:
                try:
                    normalized = pd.Timestamp(ce_start).strftime('%Y-%m-%dT%H:%M')
                    ses_keys.add((ce_phase, normalized))
                    # Store power profile if available
                    pp = ce.get('power_profile')
                    if pp and pp.get('values'):
                        ses_profiles[(ce_phase, normalized)] = pp
                except (ValueError, TypeError):
                    pass

        min_duration_threshold = 3  # spike threshold in minutes
        win_start = start - margin
        win_end = end + margin

        for phase in ['w1', 'w2', 'w3']:
            phase_matches = all_match_intervals.get(phase, [])
            if not phase_matches:
                continue
            rects = []
            for interval in phase_matches:
                # Support both old 4-tuple and new 6-tuple format
                if len(interval) >= 6:
                    m_start, m_end, m_mag, m_dur, m_on_end, m_off_start = interval[:6]
                else:
                    m_start, m_end, m_mag, m_dur = interval[:4]
                    m_on_end, m_off_start = m_start, m_end
                try:
                    ts_s = pd.Timestamp(m_start)
                    ts_e = pd.Timestamp(m_end)
                except (ValueError, TypeError):
                    continue
                # Check if within chart window
                if ts_e < win_start or ts_s > win_end:
                    continue
                # Classify: session match, spike, or other device
                normalized_start = ts_s.strftime('%Y-%m-%dT%H:%M')
                if (phase, normalized_start) in ses_keys:
                    cat = 'ses'
                elif m_dur < min_duration_threshold:
                    cat = 'spk'
                else:
                    cat = 'oth'
                rect = {
                    's': ts_s.strftime('%Y-%m-%d %H:%M'),
                    'e': ts_e.strftime('%Y-%m-%d %H:%M'),
                    'm': m_mag,
                    'c': cat,
                }
                # Add per-minute profile if available (session events)
                pp = ses_profiles.get((phase, normalized_start))
                if pp:
                    # Normalize timestamps to chart format
                    try:
                        rect['pt'] = [
                            pd.Timestamp(t).strftime('%Y-%m-%d %H:%M')
                            for t in pp['timestamps']
                        ]
                    except (ValueError, TypeError):
                        rect['pt'] = pp['timestamps']
                    rect['pp'] = pp['values']
                else:
                    # Add on_end/off_start for trapezoidal fallback
                    try:
                        rect['oe'] = pd.Timestamp(m_on_end).strftime('%Y-%m-%d %H:%M')
                        rect['os'] = pd.Timestamp(m_off_start).strftime('%Y-%m-%d %H:%M')
                    except (ValueError, TypeError):
                        pass
                rects.append(rect)
            if rects:
                result[f'mt_{phase}'] = rects

    return result


def _build_chart_row_html(act_idx: int, colspan: int) -> str:
    """Build the hidden chart row HTML with 3x3 grid placeholders."""
    divs = []
    for row_key, row_label in [('o', 'Original'), ('r', 'Remaining'), ('s', 'Segregated')]:
        divs.append(
            f'<div style="grid-column:1/-1;font-weight:600;color:#555;font-size:0.82em;'
            f'margin:{"8" if row_key != "o" else "0"}px 0 2px 4px;">{row_label} Power</div>'
        )
        for phase in ['w1', 'w2', 'w3']:
            divs.append(f'<div id="act-c-{act_idx}-{row_key}-{phase}"></div>')

    grid_content = '\n                        '.join(divs)
    return f'''
            <tr id="chart-row-{act_idx}" style="display:none;">
                <td colspan="{colspan}" style="padding:0;border-bottom:2px solid #667eea;">
                    <div id="chart-grid-{act_idx}" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px;padding:12px;background:#f8f9fa;">
                        {grid_content}
                    </div>
                </td>
            </tr>'''


def _build_activation_charts_script(all_chart_data: dict) -> str:
    """Build JavaScript for expandable activation charts with lazy Plotly rendering."""
    data_json = json.dumps(all_chart_data, separators=(',', ':'))

    return f'''
    <script>
    var _actCD = {data_json};

    function toggleActChart(idx) {{
        var row = document.getElementById('chart-row-' + idx);
        if (!row) return;
        if (row.style.display === 'none' || row.style.display === '') {{
            row.style.display = 'table-row';
            if (!row.dataset.rendered) {{
                _renderActCharts(idx);
                row.dataset.rendered = '1';
            }}
        }} else {{
            row.style.display = 'none';
        }}
    }}

    function _hexRgba(hex, a) {{
        var r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
        return 'rgba('+r+','+g+','+b+','+a+')';
    }}

    function _sessShapes(d) {{
        return [
            {{type:'line',x0:d.ss,x1:d.ss,y0:0,y1:1,yref:'paper',
              line:{{color:'rgba(0,0,0,0.4)',width:1.5,dash:'dash'}}}},
            {{type:'line',x0:d.se,x1:d.se,y0:0,y1:1,yref:'paper',
              line:{{color:'rgba(0,0,0,0.4)',width:1.5,dash:'dash'}}}},
            {{type:'rect',x0:d.ss,x1:d.se,y0:0,y1:1,yref:'paper',
              fillcolor:'rgba(0,0,0,0.04)',line:{{width:0}}}}
        ];
    }}

    function _renderActCharts(idx) {{
        var d = _actCD[idx];
        if (!d) return;
        var phases = ['w1','w2','w3'];
        var pC = {{w1:'#007bff',w2:'#dc3545',w3:'#28a745'}};
        var shapes = _sessShapes(d);

        // Compute global Y max across all phases and all rows (original/remaining/segregated)
        var gMax = 0;
        phases.forEach(function(ph) {{
            (d['o_'+ph]||[]).forEach(function(v){{ if(v>gMax) gMax=v; }});
        }});
        gMax = Math.ceil(gMax * 1.08 / 100) * 100;  // 8% headroom, round to nearest 100
        if (gMax < 100) gMax = 100;

        var xRange = [d.ts[0], d.ts[d.ts.length-1]];
        var bLay = {{
            margin:{{l:50,r:10,t:28,b:30}},height:200,
            xaxis:{{tickangle:-30,tickfont:{{size:9}},range:xRange}},
            yaxis:{{title:'W',titlefont:{{size:10}},range:[0,gMax]}},
            plot_bgcolor:'#fafafa',paper_bgcolor:'white',
            shapes:shapes,hovermode:'x unified'
        }};
        var cfg = {{displayModeBar:false,responsive:true}};

        phases.forEach(function(ph) {{
            var orig = d['o_'+ph]||[];
            var rem = d['r_'+ph]||[];
            var ts = d.ts;
            if (!ts || !orig.length) return;

            // Row 1: Original
            Plotly.newPlot('act-c-'+idx+'-o-'+ph, [{{
                x:ts,y:orig,type:'scatter',mode:'lines',
                line:{{color:pC[ph],width:2}},
                hovertemplate:'%{{y:.0f}}W<extra></extra>'
            }}], Object.assign({{}},bLay,{{
                title:{{text:ph.toUpperCase()+' — Original',font:{{size:12}}}}
            }}), cfg);

            // Row 2: Remaining
            Plotly.newPlot('act-c-'+idx+'-r-'+ph, [{{
                x:ts,y:rem,type:'scatter',mode:'lines',
                line:{{color:pC[ph],width:2}},
                hovertemplate:'%{{y:.0f}}W<extra></extra>'
            }}], Object.assign({{}},bLay,{{
                title:{{text:ph.toUpperCase()+' — Remaining',font:{{size:12}}}}
            }}), cfg);

            // Row 3: Segregated — individual match rectangles as toself polygons
            var segTraces = [];
            var matchRects = d['mt_'+ph]||[];
            var sesX=[],sesY=[],spkX=[],spkY=[],othX=[],othY=[];
            matchRects.forEach(function(r) {{
                var bx,by;
                if (r.c==='ses') {{ bx=sesX; by=sesY; }}
                else if (r.c==='spk') {{ bx=spkX; by=spkY; }}
                else {{ bx=othX; by=othY; }}

                if (r.pp && r.pt && r.pp.length>0) {{
                    // Per-minute profile: draw actual extraction shape
                    for (var i=0;i<r.pt.length;i++) {{
                        bx.push(r.pt[i]);
                        by.push(r.pp[i]);
                    }}
                    // Close polygon back to zero
                    bx.push(r.pt[r.pt.length-1],r.pt[0],null);
                    by.push(0,0,null);
                }} else if (r.oe && r.os && r.oe!==r.s && r.os!==r.e) {{
                    // Trapezoidal fallback (ramp up → stable → ramp down)
                    bx.push(r.s,r.oe,r.os,r.e,r.s,null);
                    by.push(0,r.m,r.m,0,0,null);
                }} else {{
                    // Flat rectangle fallback
                    bx.push(r.s,r.s,r.e,r.e,r.s,null);
                    by.push(0,r.m,r.m,0,0,null);
                }}
            }});

            if (sesX.length) segTraces.push({{
                x:sesX,y:sesY,type:'scatter',mode:'lines',
                fill:'toself',fillcolor:_hexRgba(d.dc,0.55),
                line:{{color:d.dc,width:1}},
                name:'This session',legendgroup:'ses',showlegend:true,
                hovertemplate:'Session: %{{y:.0f}}W<extra></extra>'
            }});
            if (spkX.length) segTraces.push({{
                x:spkX,y:spkY,type:'scatter',mode:'lines',
                fill:'toself',fillcolor:'rgba(255,165,0,0.4)',
                line:{{color:'#e67e22',width:1}},
                name:'Filtered (<2 min)',legendgroup:'spk',showlegend:true,
                hovertemplate:'Spike: %{{y:.0f}}W<extra></extra>'
            }});
            if (othX.length) segTraces.push({{
                x:othX,y:othY,type:'scatter',mode:'lines',
                fill:'toself',fillcolor:'rgba(176,176,176,0.4)',
                line:{{color:'#aaa',width:1}},
                name:'Other devices',legendgroup:'oth',showlegend:true,
                hovertemplate:'Other: %{{y:.0f}}W<extra></extra>'
            }});
            if (!segTraces.length) segTraces.push({{
                x:ts,y:ts.map(function(){{return null;}}),
                type:'scatter',mode:'lines',line:{{width:0}},
                showlegend:false,hoverinfo:'skip'
            }});

            Plotly.newPlot('act-c-'+idx+'-s-'+ph, segTraces,
            Object.assign({{}},bLay,{{
                title:{{text:ph.toUpperCase()+' — Segregated',font:{{size:12}}}},
                margin:{{l:50,r:10,t:28,b:55}},
                showlegend:true,
                legend:{{orientation:'h',y:-0.3,font:{{size:10}}}}
            }}), cfg);
        }});
    }}
    </script>'''
