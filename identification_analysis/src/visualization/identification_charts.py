"""
Chart generation for device identification (Module 2) reports.

Contains:
  - Existing charts moved from dynamic_report_charts.py:
    create_device_summary_table, create_device_activations_detail
  - New session-based charts:
    create_session_overview, create_boiler_analysis, create_ac_analysis,
    create_temporal_heatmap, create_unclassified_analysis
"""
import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List, Optional

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
    'central_ac': '#dc3545',
    'regular_ac': '#e67e22',
    'unknown': '#6c757d',
    'unclassified': '#6c757d',
}

DEVICE_DISPLAY_NAMES = {
    'boiler': 'Water Heater (Boiler)',
    'central_ac': 'Central AC (Multi-phase)',
    'regular_ac': 'Regular AC (Single-phase)',
    'unknown': 'Unclassified',
    'unclassified': 'Unclassified',
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

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
    for dtype in ['boiler', 'central_ac', 'regular_ac', 'unclassified']:
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
    dur_s = _dur_str(duration)
    magnitude = abs(act.get('on_magnitude', 0) or 0)
    confidence = act.get('confidence', 0) or 0

    return {
        'date': date_str, 'start': on_time_str, 'end': off_time_str,
        'duration': duration, 'dur_str': dur_s,
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
        })

    result.sort(key=lambda r: r['date'] + r['start'])
    return result


def create_device_activations_detail(activations: List[Dict[str, Any]], house_id: str = '') -> str:
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

    MIN_DURATION = {
        'boiler': 15, 'central_ac': 5, 'regular_ac': 3, 'unclassified': 0,
    }

    device_groups = {}
    all_copyable = []
    for act in matched:
        parsed = _parse_activation_row(act)
        dtype = parsed['device_type']
        min_dur = MIN_DURATION.get(dtype, 0)
        if parsed['duration'] < min_dur:
            continue
        device_groups.setdefault(dtype, []).append(parsed)
        # Only collect dates for classified (non-unknown) device types
        if dtype != 'unclassified' and parsed['date'] and parsed['start']:
            all_copyable.append(f"{parsed['date']} {parsed['start']}-{parsed['end']}")

    if not any(device_groups.values()):
        return '<p style="color: #888;">No high-confidence device activations found.</p>'

    # Exclude unclassified/unknown from display (user requested)
    display_order = ['boiler', 'central_ac', 'regular_ac']

    # Limit to MAX_DISPLAY_PER_TYPE activations per device type
    MAX_DISPLAY_PER_TYPE = 15

    total_count = sum(len(v) for k, v in device_groups.items() if k != 'unclassified')
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
            session_rows = _group_central_ac_for_display(events)
            count = len(session_rows)
            # Collect ALL dates for copy button
            for r in session_rows:
                if r['date'] and r['start']:
                    copyable_dates.append(f"{r['date']} {r['start']}-{r['end']}")
            display_rows = session_rows[:MAX_DISPLAY_PER_TYPE]
            truncated = count > MAX_DISPLAY_PER_TYPE
            for i, r in enumerate(display_rows, 1):
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
            events.sort(key=lambda r: r['date'] + r['start'])
            count = len(events)
            # Collect ALL dates for copy button
            for r in events:
                if r['date'] and r['start']:
                    copyable_dates.append(f"{r['date']} {r['start']}-{r['end']}")
            display_events = events[:MAX_DISPLAY_PER_TYPE]
            truncated = count > MAX_DISPLAY_PER_TYPE
            for i, r in enumerate(display_events, 1):
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

        truncation_notice = ''
        if truncated:
            truncation_notice = f'''
                <div style="padding: 6px 12px; margin-top: 4px; background: #f0f4ff; border: 1px solid #d0d8f0; border-radius: 4px; font-size: 0.82em; color: #4a5568;">
                    Showing {MAX_DISPLAY_PER_TYPE} of {count} activations.
                    Full data available in <code>device_sessions_{house_id}.json</code>
                </div>'''

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
                </table>{truncation_notice}
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


# ---------------------------------------------------------------------------
# NEW session-based charts for the M2 Identification Report
# ---------------------------------------------------------------------------

def create_session_overview(sessions: List[Dict]) -> str:
    """
    Session Overview Dashboard: pie chart of sessions by device type +
    summary cards (total sessions, classified %, energy attributed).

    Args:
        sessions: List of session dicts from device_sessions JSON.
    """
    if not sessions:
        return '<p style="color: #888;">No session data available.</p>'

    # Count and energy by device type
    counts = defaultdict(int)
    energy = defaultdict(float)  # magnitude * duration (watt-minutes)
    for s in sessions:
        dtype = s.get('device_type', 'unknown')
        counts[dtype] += 1
        mag = s.get('avg_cycle_magnitude_w', 0) or 0
        dur = s.get('duration_minutes', 0) or 0
        energy[dtype] += mag * dur

    total = sum(counts.values())
    classified = sum(c for dt, c in counts.items() if dt not in ('unknown', 'unclassified'))
    classified_pct = (classified / total * 100) if total > 0 else 0
    total_energy = sum(energy.values())

    # Summary cards
    avg_conf_vals = [s.get('confidence', 0) for s in sessions if s.get('confidence')]
    avg_conf = sum(avg_conf_vals) / len(avg_conf_vals) if avg_conf_vals else 0
    conf_color = GREEN if avg_conf >= 0.7 else ORANGE if avg_conf >= 0.5 else RED

    cards_html = f'''
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;">
        <div style="background: #f0f4ff; border: 1px solid #c3d4ff; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; font-weight: 700; color: #2d3748;">{total}</div>
            <div style="font-size: 0.85em; color: #666;">Total Sessions</div>
        </div>
        <div style="background: {LIGHT_GREEN}; border: 1px solid #c3e6cb; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; font-weight: 700; color: {GREEN};">{classified_pct:.0f}%</div>
            <div style="font-size: 0.85em; color: #666;">Classified</div>
        </div>
        <div style="background: #f0f4ff; border: 1px solid #c3d4ff; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; font-weight: 700; color: {conf_color};">{avg_conf:.0%}</div>
            <div style="font-size: 0.85em; color: #666;">Avg Confidence</div>
        </div>
        <div style="background: #f0f4ff; border: 1px solid #c3d4ff; border-radius: 8px; padding: 16px; text-align: center;">
            <div style="font-size: 2em; font-weight: 700; color: #2d3748;">{len(counts)}</div>
            <div style="font-size: 0.85em; color: #666;">Device Types</div>
        </div>
    </div>'''

    # Pie chart — sessions by device type
    display_order = ['boiler', 'central_ac', 'regular_ac', 'unknown']
    pie_labels = []
    pie_values = []
    pie_colors = []
    for dt in display_order:
        if counts.get(dt, 0) > 0:
            pie_labels.append(DEVICE_DISPLAY_NAMES.get(dt, dt))
            pie_values.append(counts[dt])
            pie_colors.append(DEVICE_COLORS.get(dt, GRAY))

    pie_traces = json.dumps([{
        'type': 'pie',
        'labels': pie_labels,
        'values': pie_values,
        'marker': {'colors': pie_colors},
        'textinfo': 'label+percent',
        'textposition': 'auto',
        'hovertemplate': '%{label}: %{value} sessions (%{percent})<extra></extra>',
        'hole': 0.35,
    }])
    pie_layout = json.dumps({
        'margin': {'l': 20, 'r': 20, 't': 30, 'b': 20},
        'height': 280,
        'title': {'text': 'Sessions by Device Type', 'font': {'size': 14}},
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
            <div id="session_pie"></div>
            <script>Plotly.newPlot('session_pie', {pie_traces}, {pie_layout}, {{displayModeBar:false}});</script>
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
    boilers = [s for s in sessions if s.get('device_type') == 'boiler']
    if not boilers:
        return '<p style="color: #888;">No boiler sessions detected.</p>'

    # Extract hours and months from start timestamps
    hours = []
    months = defaultdict(int)  # 'YYYY-MM' -> count
    magnitudes = []
    phases = defaultdict(int)

    for s in boilers:
        dt = _parse_iso(s.get('start', ''))
        if dt:
            hours.append(dt.hour)
            months[dt.strftime('%Y-%m')] += 1
        mag = s.get('avg_cycle_magnitude_w', 0)
        if mag:
            magnitudes.append(mag)
        for ph in (s.get('phases') or []):
            phases[ph] += 1

    # Phase info
    dominant_phase = max(phases, key=phases.get) if phases else 'N/A'
    phase_str = ', '.join(f'{ph}: {cnt}' for ph, cnt in sorted(phases.items()))

    # Hour histogram
    hour_counts = [0] * 24
    for h in hours:
        hour_counts[h] += 1

    hour_traces = json.dumps([{
        'type': 'bar',
        'x': list(range(24)),
        'y': hour_counts,
        'marker': {'color': BLUE},
        'hovertemplate': 'Hour %{x}: %{y} sessions<extra></extra>',
    }])
    hour_layout = json.dumps({
        'margin': {'l': 40, 'r': 10, 't': 30, 'b': 40},
        'height': 220,
        'title': {'text': 'Boiler Usage by Hour of Day', 'font': {'size': 13}},
        'xaxis': {'title': 'Hour', 'dtick': 2, 'range': [-0.5, 23.5]},
        'yaxis': {'title': 'Sessions'},
        'paper_bgcolor': 'white',
        'plot_bgcolor': '#f8f9fa',
    })

    # Monthly bar chart
    sorted_months = sorted(months.keys())
    month_labels = sorted_months
    month_values = [months[m] for m in sorted_months]

    month_traces = json.dumps([{
        'type': 'bar',
        'x': month_labels,
        'y': month_values,
        'marker': {'color': BLUE},
        'hovertemplate': '%{x}: %{y} sessions<extra></extra>',
    }])
    month_layout = json.dumps({
        'margin': {'l': 40, 'r': 10, 't': 30, 'b': 60},
        'height': 220,
        'title': {'text': 'Boiler Sessions per Month', 'font': {'size': 13}},
        'xaxis': {'title': '', 'tickangle': -45},
        'yaxis': {'title': 'Sessions'},
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

    # Seasonal pattern (monthly counts for both AC types)
    all_ac = central + regular
    month_central = defaultdict(int)
    month_regular = defaultdict(int)

    for s in central:
        dt = _parse_iso(s.get('start', ''))
        if dt:
            month_central[dt.strftime('%Y-%m')] += 1

    for s in regular:
        dt = _parse_iso(s.get('start', ''))
        if dt:
            month_regular[dt.strftime('%Y-%m')] += 1

    all_months = sorted(set(list(month_central.keys()) + list(month_regular.keys())))
    if all_months:
        traces = []
        if central:
            traces.append({
                'type': 'bar', 'name': 'Central AC',
                'x': all_months, 'y': [month_central.get(m, 0) for m in all_months],
                'marker': {'color': RED},
            })
        if regular:
            traces.append({
                'type': 'bar', 'name': 'Regular AC',
                'x': all_months, 'y': [month_regular.get(m, 0) for m in all_months],
                'marker': {'color': ORANGE},
            })

        seasonal_layout = json.dumps({
            'barmode': 'group',
            'margin': {'l': 40, 'r': 10, 't': 30, 'b': 60},
            'height': 250,
            'title': {'text': 'AC Sessions by Month (Seasonal Pattern)', 'font': {'size': 13}},
            'xaxis': {'tickangle': -45},
            'yaxis': {'title': 'Sessions'},
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
    for dtype in ['boiler', 'central_ac', 'regular_ac']:
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


def create_confidence_overview(sessions: List[Dict]) -> str:
    """
    Confidence distribution overview: histogram of confidence scores across all sessions,
    broken down by tier (high/medium/low).
    """
    if not sessions:
        return '<p style="color: #888;">No session data available.</p>'

    classified = [s for s in sessions if s.get('device_type') not in ('unknown', 'unclassified')]
    if not classified:
        return '<p style="color: #888;">No classified sessions with confidence scores.</p>'

    confidences = [s.get('confidence', 0) for s in classified]
    high = sum(1 for c in confidences if c >= 0.7)
    medium = sum(1 for c in confidences if 0.4 <= c < 0.7)
    low = sum(1 for c in confidences if c < 0.4)
    total = len(confidences)

    # Summary cards
    cards_html = f'''
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 16px;">
        <div style="background: #d4edda; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.3em; font-weight: 700; color: {GREEN};">{high} ({high/total:.0%})</div>
            <div style="font-size: 0.8em; color: #666;">High (\u22650.70)</div>
        </div>
        <div style="background: #fef3cd; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.3em; font-weight: 700; color: {ORANGE};">{medium} ({medium/total:.0%})</div>
            <div style="font-size: 0.8em; color: #666;">Medium (0.40\u20130.70)</div>
        </div>
        <div style="background: #f8d7da; border-radius: 6px; padding: 12px; text-align: center;">
            <div style="font-size: 1.3em; font-weight: 700; color: {RED};">{low} ({low/total:.0%})</div>
            <div style="font-size: 0.8em; color: #666;">Low (&lt;0.40)</div>
        </div>
    </div>'''

    # Histogram with tier coloring
    traces = json.dumps([
        {
            'type': 'histogram',
            'x': [c for c in confidences if c >= 0.7],
            'name': 'High',
            'marker': {'color': GREEN},
            'xbins': {'start': 0, 'end': 1, 'size': 0.05},
        },
        {
            'type': 'histogram',
            'x': [c for c in confidences if 0.4 <= c < 0.7],
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
            {'type': 'line', 'x0': 0.7, 'x1': 0.7, 'y0': 0, 'y1': 1, 'yref': 'paper',
             'line': {'color': '#888', 'dash': 'dot', 'width': 1}},
            {'type': 'line', 'x0': 0.4, 'x1': 0.4, 'y0': 0, 'y1': 1, 'yref': 'paper',
             'line': {'color': '#888', 'dash': 'dot', 'width': 1}},
        ],
    })

    return f'''
    {cards_html}
    <div id="confidence_hist"></div>
    <script>Plotly.newPlot('confidence_hist', {traces}, {layout}, {{displayModeBar:false}});</script>'''


# ---------------------------------------------------------------------------
# Spike Filter Analysis
# ---------------------------------------------------------------------------

def create_spike_analysis(spike_filter: Dict[str, Any]) -> str:
    """
    Create spike filter analysis section with two charts:
    1. Bar chart: event count — spikes vs. kept, by iteration
    2. Bar chart: total minutes — spikes vs. kept, by iteration
    """
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
    # Categories: Spikes (<3 min), Short (3-25 min), Long (>=25 min)
    short_count = spike_filter.get('short_count', 0)
    long_count = spike_filter.get('long_count', 0)
    short_min = spike_filter.get('short_minutes', 0)
    long_min = spike_filter.get('long_minutes', 0)

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
        pie_labels.append(f'Short ({threshold}-25 min)')
        pie_values_count.append(short_count)
        pie_values_min.append(short_min)
        pie_colors.append(ORANGE)
    if long_count > 0:
        pie_labels.append('Long (>=25 min)')
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


# ---------------------------------------------------------------------------
# Per-session power visualization (original / remaining / segregated)
# ---------------------------------------------------------------------------

def create_session_power_plots(
    sessions: List[Dict],
    summarized_data,  # pd.DataFrame with timestamp, original_wN, remaining_wN columns
    margin_minutes: int = 60,
    max_sessions: int = 50,
) -> str:
    """Create per-session power charts showing original, remaining, and segregated power.

    For each classified session, renders a Plotly chart with:
      - Original power (blue) — what the sensor read
      - Remaining power (gray) — after all device extraction
      - Segregated power (green fill) — what was attributed to devices

    Args:
        sessions: List of session dicts from device_sessions JSON
        summarized_data: DataFrame with columns: timestamp, original_wN, remaining_wN
        margin_minutes: Minutes to show before and after session (default: 60)
        max_sessions: Maximum number of session charts to generate

    Returns:
        HTML string with all session charts
    """
    import pandas as pd

    if summarized_data is None or summarized_data.empty or not sessions:
        return '<p style="color: #888;">No power data available for session visualization.</p>'

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(summarized_data['timestamp']):
        summarized_data = summarized_data.copy()
        summarized_data['timestamp'] = pd.to_datetime(summarized_data['timestamp'])
    summarized_data = summarized_data.sort_values('timestamp').reset_index(drop=True)

    # Filter to classified sessions only, sorted by confidence (highest first)
    classified = [s for s in sessions if s.get('device_type') not in ('unknown', None)]
    classified.sort(key=lambda s: s.get('confidence', 0), reverse=True)

    if not classified:
        return '<p style="color: #888;">No classified sessions to visualize.</p>'

    classified = classified[:max_sessions]

    # Group by device type for organized display
    by_type = {}
    for s in classified:
        dtype = s.get('device_type', 'unknown')
        by_type.setdefault(dtype, []).append(s)

    type_order = ['boiler', 'central_ac', 'regular_ac']
    type_labels = {'boiler': 'Boiler', 'central_ac': 'Central AC', 'regular_ac': 'Regular AC'}

    charts_html_parts = []
    chart_id = 0

    for dtype in type_order:
        if dtype not in by_type:
            continue
        dtype_sessions = by_type[dtype]
        color = DEVICE_COLORS.get(dtype, GRAY)
        label = type_labels.get(dtype, dtype)

        charts_html_parts.append(
            f'<h3 style="color:{color}; margin-top:20px; margin-bottom:10px;">'
            f'{label} ({len(dtype_sessions)} sessions)</h3>'
        )

        for s in dtype_sessions:
            chart_html = _render_session_chart(
                s, summarized_data, margin_minutes, chart_id, color
            )
            charts_html_parts.append(chart_html)
            chart_id += 1

    if not charts_html_parts:
        return '<p style="color: #888;">No session charts generated.</p>'

    return '\n'.join(charts_html_parts)


def _render_session_chart(
    session: Dict,
    summarized_data,  # pd.DataFrame
    margin_minutes: int,
    chart_id: int,
    device_color: str,
) -> str:
    """Render a single session's power chart."""
    import pandas as pd

    start_str = session.get('start', '')
    end_str = session.get('end', '')
    if not start_str or not end_str:
        return ''

    session_start = pd.Timestamp(start_str)
    session_end = pd.Timestamp(end_str)
    margin = pd.Timedelta(minutes=margin_minutes)

    window_start = session_start - margin
    window_end = session_end + margin

    # Slice data to window
    mask = (summarized_data['timestamp'] >= window_start) & (summarized_data['timestamp'] <= window_end)
    window_data = summarized_data[mask]

    if window_data.empty:
        return ''

    # Determine phases for this session
    phases = session.get('phases', [])
    if not phases:
        # Try to infer from constituent events
        events = session.get('constituent_events', [])
        phases = list(set(e.get('phase', '') for e in events if e.get('phase')))
    if not phases:
        return ''

    # Session metadata
    dtype = session.get('device_type', 'unknown')
    confidence = session.get('confidence', 0)
    duration = session.get('duration_minutes', 0)
    magnitude = session.get('avg_cycle_magnitude_w', 0)
    cycles = session.get('cycle_count', 0)

    label_map = {'boiler': 'Boiler', 'central_ac': 'Central AC', 'regular_ac': 'Regular AC'}
    dtype_label = label_map.get(dtype, dtype)

    # Build chart: one subplot per phase
    timestamps = window_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()

    traces = []
    phase_colors = {'w1': '#007bff', 'w2': '#dc3545', 'w3': '#28a745'}

    for phase in phases:
        orig_col = f'original_{phase}'
        remain_col = f'remaining_{phase}'

        if orig_col not in window_data.columns or remain_col not in window_data.columns:
            continue

        original = window_data[orig_col].tolist()
        remaining = window_data[remain_col].tolist()
        segregated = [(o - r) if (o is not None and r is not None) else 0
                      for o, r in zip(original, remaining)]

        ph_color = phase_colors.get(phase, '#333')
        phase_label = phase.upper()

        # Original power line
        traces.append({
            'x': timestamps, 'y': original, 'type': 'scatter', 'mode': 'lines',
            'name': f'{phase_label} Original',
            'line': {'color': ph_color, 'width': 2},
            'hovertemplate': f'{phase_label} Original: %{{y:.0f}}W<br>%{{x}}<extra></extra>',
        })

        # Remaining power line
        traces.append({
            'x': timestamps, 'y': remaining, 'type': 'scatter', 'mode': 'lines',
            'name': f'{phase_label} Remaining',
            'line': {'color': ph_color, 'width': 1.5, 'dash': 'dash'},
            'hovertemplate': f'{phase_label} Remaining: %{{y:.0f}}W<br>%{{x}}<extra></extra>',
        })

        # Segregated power (filled area)
        traces.append({
            'x': timestamps, 'y': segregated, 'type': 'scatter', 'mode': 'lines',
            'name': f'{phase_label} Segregated',
            'fill': 'tozeroy',
            'fillcolor': ph_color.replace(')', ',0.2)').replace('rgb', 'rgba') if 'rgb' in ph_color
                        else _hex_to_rgba(ph_color, 0.2),
            'line': {'color': ph_color, 'width': 1, 'dash': 'dot'},
            'hovertemplate': f'{phase_label} Segregated: %{{y:.0f}}W<br>%{{x}}<extra></extra>',
        })

    if not traces:
        return ''

    # Session time range markers
    start_idx = session_start.strftime('%Y-%m-%d %H:%M')
    end_idx = session_end.strftime('%Y-%m-%d %H:%M')

    shapes = [
        # Session start vertical line
        {'type': 'line', 'x0': start_idx, 'x1': start_idx, 'y0': 0, 'y1': 1,
         'yref': 'paper', 'line': {'color': device_color, 'width': 2, 'dash': 'dash'}},
        # Session end vertical line
        {'type': 'line', 'x0': end_idx, 'x1': end_idx, 'y0': 0, 'y1': 1,
         'yref': 'paper', 'line': {'color': device_color, 'width': 2, 'dash': 'dash'}},
        # Session highlight rectangle
        {'type': 'rect', 'x0': start_idx, 'x1': end_idx, 'y0': 0, 'y1': 1,
         'yref': 'paper', 'fillcolor': device_color, 'opacity': 0.07, 'line': {'width': 0}},
    ]

    layout = {
        'height': 300,
        'margin': {'l': 50, 'r': 20, 't': 10, 'b': 40},
        'xaxis': {'title': '', 'tickangle': -30, 'tickfont': {'size': 10}},
        'yaxis': {'title': 'Power (W)', 'titlefont': {'size': 11}},
        'legend': {'orientation': 'h', 'y': -0.3, 'x': 0, 'font': {'size': 10}},
        'plot_bgcolor': '#fafafa',
        'paper_bgcolor': 'white',
        'shapes': shapes,
        'hovermode': 'x unified',
    }

    traces_json = json.dumps(traces)
    layout_json = json.dumps(layout)
    div_id = f'session_power_{chart_id}'

    # Header with session info
    phases_str = ', '.join(p.upper() for p in phases)
    header = (
        f'<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px; flex-wrap:wrap;">'
        f'<span style="font-weight:600; color:#333;">'
        f'{session_start.strftime("%Y-%m-%d %H:%M")} — {session_end.strftime("%H:%M")}</span>'
        f'<span style="background:{device_color}; color:white; padding:1px 8px; border-radius:3px; font-size:0.82em;">'
        f'{dtype_label}</span>'
        f'<span style="font-size:0.85em; color:#666;">Confidence: {confidence:.2f} | '
        f'Duration: {duration:.0f}min | Magnitude: {magnitude:.0f}W | '
        f'Cycles: {cycles} | Phase: {phases_str}</span>'
        f'</div>'
    )

    return f'''
    <div style="margin-bottom:16px; border:1px solid #e2e8f0; border-radius:8px; padding:12px; background:white;">
        {header}
        <div id="{div_id}"></div>
        <script>Plotly.newPlot('{div_id}', {traces_json}, {layout_json}, {{displayModeBar:false, responsive:true}});</script>
    </div>'''


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'
