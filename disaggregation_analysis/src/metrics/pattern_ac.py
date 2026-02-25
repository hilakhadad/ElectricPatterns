"""
AC pattern detection and validation.

Extracted from patterns.py -- contains AC session grouping,
session validation, and AC pattern detection.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


def _group_activations_into_sessions(activations: List[Dict],
                                       gap_threshold_minutes: int = 60) -> List[Dict]:
    """
    Group individual activations into sessions based on time gaps.

    AC compressors cycle multiple times during a cooling session (4-5 min ON/OFF).
    These cycles should be grouped into a single "session" - from first event
    until last event before a gap > threshold.

    Args:
        activations: List of activation dicts with 'date', 'on_time', 'off_time',
                    'duration_minutes', 'magnitude', etc.
        gap_threshold_minutes: Gap between activations to start a new session (default 60 min)

    Returns:
        List of session dicts with aggregated data
    """
    if not activations:
        return []

    # Sort activations by datetime
    def get_datetime(act):
        try:
            date_str = act.get('date', '')
            time_str = act.get('on_time', '00:00')
            return pd.to_datetime(f"{date_str} {time_str}")
        except (ValueError, TypeError):
            return pd.NaT

    sorted_acts = sorted(activations, key=lambda x: get_datetime(x))

    sessions = []
    current_session = {
        'activations': [sorted_acts[0]],
        'start_datetime': get_datetime(sorted_acts[0]),
    }

    gap_threshold = pd.Timedelta(minutes=gap_threshold_minutes)

    for i in range(1, len(sorted_acts)):
        act = sorted_acts[i]
        act_datetime = get_datetime(act)

        # Get the end time of the last activation in current session
        last_act = current_session['activations'][-1]
        last_end_time = last_act.get('off_time', last_act.get('on_time', '00:00'))
        last_date = last_act.get('date', '')

        try:
            last_datetime = pd.to_datetime(f"{last_date} {last_end_time}")
        except (ValueError, TypeError):
            last_datetime = get_datetime(last_act)

        # Check if this activation starts more than gap_threshold after last ended
        if pd.notna(act_datetime) and pd.notna(last_datetime):
            time_gap = act_datetime - last_datetime

            if time_gap > gap_threshold:
                # Start a new session - save current one first
                sessions.append(_summarize_session(current_session))
                current_session = {
                    'activations': [act],
                    'start_datetime': act_datetime,
                }
            else:
                # Continue current session
                current_session['activations'].append(act)
        else:
            # Can't determine gap - add to current session
            current_session['activations'].append(act)

    # Don't forget the last session
    sessions.append(_summarize_session(current_session))

    return sessions


def _summarize_session(session: Dict) -> Dict:
    """
    Summarize a session of activations into a single record.

    Args:
        session: Dict with 'activations' list and 'start_datetime'

    Returns:
        Session summary dict
    """
    activations = session['activations']
    if not activations:
        return {}

    first_act = activations[0]
    last_act = activations[-1]

    # Calculate total duration from first ON to last OFF
    first_date = first_act.get('date', '')
    first_time = first_act.get('on_time', '00:00')
    last_date = last_act.get('date', '')
    last_off_time = last_act.get('off_time', last_act.get('on_time', '00:00'))

    try:
        start_dt = pd.to_datetime(f"{first_date} {first_time}")
        end_dt = pd.to_datetime(f"{last_date} {last_off_time}")
        total_duration = (end_dt - start_dt).total_seconds() / 60
    except (ValueError, TypeError):
        # Fallback: sum individual durations
        total_duration = sum(a.get('duration_minutes', 0) for a in activations)

    # Get magnitude (use max or average)
    magnitudes = []
    for act in activations:
        if 'magnitude' in act:
            magnitudes.append(act['magnitude'])
        elif 'total_magnitude' in act:
            magnitudes.append(act['total_magnitude'])

    avg_magnitude = int(sum(magnitudes) / len(magnitudes)) if magnitudes else 0
    max_magnitude = max(magnitudes) if magnitudes else 0

    # Preserve phase info
    phase = first_act.get('phase', '')
    phases = first_act.get('phases', [phase] if phase else [])
    phase_magnitudes = first_act.get('phase_magnitudes', {})

    return {
        'date': first_date,
        'on_time': first_time,
        'off_time': last_off_time,
        'duration_minutes': int(round(total_duration)),
        'magnitude': avg_magnitude,
        'max_magnitude': max_magnitude,
        'total_magnitude': sum(magnitudes) if magnitudes else 0,
        'cycle_count': len(activations),  # Number of compressor cycles in this session
        'phase': phase,
        'phases': phases,
        'phase_magnitudes': phase_magnitudes,
        '_raw_activations': activations,  # Keep raw data for validation
    }


def _is_valid_ac_session(session: Dict, raw_activations: List[Dict],
                          min_first_duration: int = 15,
                          min_following_cycles: int = 3,
                          max_magnitude_std_pct: float = 0.20) -> bool:
    """
    Validate if a session looks like real AC compressor cycling.

    AC pattern: a long initial compressor run (>=15 min) followed by
    at least 3 more ON/OFF cycles (compressor cycling). The session
    spans from the start of the first event to the end of the last.

    Args:
        session: Session summary dict with cycle_count, duration_minutes, etc.
        raw_activations: List of individual activations in this session
        min_first_duration: Minimum duration of the first activation in minutes (default 15)
        min_following_cycles: Minimum number of cycles after the first (default 3)
        max_magnitude_std_pct: Maximum magnitude std as percentage of mean (default 20%)

    Returns:
        True if session looks like AC, False otherwise
    """
    cycle_count = session.get('cycle_count', 1)

    # Need 1 long first + N following cycles
    if cycle_count < (1 + min_following_cycles):
        return False

    # First activation must be >= min_first_duration (initial compressor run)
    if raw_activations:
        first_duration = raw_activations[0].get('duration_minutes', 0) or 0
        if first_duration < min_first_duration:
            return False
    else:
        return False

    # Check magnitude consistency across all cycles (std < 20% of mean)
    if len(raw_activations) >= 2:
        magnitudes = [a.get('magnitude', 0) for a in raw_activations]
        if magnitudes:
            mean_mag = np.mean(magnitudes)
            std_mag = np.std(magnitudes)
            if mean_mag > 0 and (std_mag / mean_mag) > max_magnitude_std_pct:
                return False

    return True


def detect_ac_patterns(experiment_dir: Path, house_id: str,
                       run_number: int = 0,
                       time_tolerance_minutes: int = 10,
                       damaged_phases: List[str] = None,
                       preloaded: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Detect AC patterns - both central AC (multi-phase) and regular AC (single phase).

    Central AC: Events that occur on all 3 phases (or 2 if one is damaged) at similar times.
    Regular AC: High-power events only on w1 phase.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number to analyze
        time_tolerance_minutes: Max time difference between phases to be considered synchronized
        damaged_phases: List of damaged phase names (e.g., ['w3'])
        preloaded: Optional dict with pre-loaded DataFrames ('matches')

    Returns:
        Dictionary with central_ac and regular_ac activation lists
    """
    logger.debug("detect_ac_patterns: house_id=%s, run_number=%d, preloaded=%s",
                 house_id, run_number, bool(preloaded))
    result = {
        'central_ac': {
            'activations': [],
            'total_count': 0,
            'phases_used': [],
        },
        'regular_ac': {
            'activations': [],
            'total_count': 0,
        },
        'has_central_ac': False,
        'has_regular_ac': False,
    }

    if preloaded:
        matches_df = preloaded.get('matches')
        if matches_df is not None:
            matches_df = matches_df.copy()
    else:
        from metrics.patterns import _get_house_dir, _load_monthly_files
        house_dir = _get_house_dir(experiment_dir, house_id, run_number)
        if not house_dir.exists():
            return result
        matches_df = _load_monthly_files(house_dir, "matches", f"matches_{house_id}_*.pkl")

    if matches_df is None or len(matches_df) == 0:
        logger.warning("detect_ac_patterns: No matches data found for house %s run %d",
                       house_id, run_number)
        return result

    # Parse timestamps (skip if already datetime)
    for col in ['on_start', 'on_end', 'off_start', 'off_end']:
        if col in matches_df.columns and not pd.api.types.is_datetime64_any_dtype(matches_df[col]):
            matches_df[col] = pd.to_datetime(matches_df[col], format='mixed', dayfirst=True, errors='coerce')

    # Determine which phases to check
    all_phases = ['w1', 'w2', 'w3']
    damaged_phases = damaged_phases or []
    active_phases = [p for p in all_phases if p not in damaged_phases]

    # Need at least 2 phases for central AC
    if len(active_phases) < 2:
        return result

    tolerance = pd.Timedelta(minutes=time_tolerance_minutes)

    # Group matches by phase
    phase_matches = {}
    for phase in active_phases:
        phase_df = matches_df[matches_df['phase'] == phase].copy()
        if not phase_df.empty:
            phase_matches[phase] = phase_df

    # Find central AC activations (synchronized across phases)
    # Optimized: use merge_asof instead of iterrows
    central_activations = []
    used_match_ids = set()

    # Use w1 as reference if available, otherwise use first available phase
    reference_phase = 'w1' if 'w1' in phase_matches else active_phases[0]
    if reference_phase not in phase_matches:
        return result

    ref_matches = phase_matches[reference_phase].copy()
    ref_matches = ref_matches.sort_values('on_start').reset_index(drop=True)

    # Use merge_asof to find synchronized events on other phases
    merged = ref_matches[['on_start', 'on_event_id', 'on_magnitude', 'off_end', 'off_start', 'duration']].copy()
    merged = merged.rename(columns={
        'on_event_id': f'id_{reference_phase}',
        'on_magnitude': f'mag_{reference_phase}',
        'off_end': f'off_end_{reference_phase}',
        'off_start': f'off_start_{reference_phase}',
        'duration': f'dur_{reference_phase}'
    })

    other_phases_in_merge = []
    for other_phase in active_phases:
        if other_phase == reference_phase or other_phase not in phase_matches:
            continue

        other_df = phase_matches[other_phase][['on_start', 'on_event_id', 'on_magnitude', 'off_end', 'off_start', 'duration']].copy()
        other_df = other_df.sort_values('on_start').reset_index(drop=True)
        other_df = other_df.rename(columns={
            'on_start': f'on_start_{other_phase}',
            'on_event_id': f'id_{other_phase}',
            'on_magnitude': f'mag_{other_phase}',
            'off_end': f'off_end_{other_phase}',
            'off_start': f'off_start_{other_phase}',
            'duration': f'dur_{other_phase}'
        })

        # Merge with tolerance - ensure both DataFrames have reset index
        merged = merged.sort_values('on_start').reset_index(drop=True)
        other_df = other_df.sort_values(f'on_start_{other_phase}').reset_index(drop=True)

        merged = pd.merge_asof(
            merged,
            other_df,
            left_on='on_start',
            right_on=f'on_start_{other_phase}',
            tolerance=tolerance,
            direction='nearest'
        )
        other_phases_in_merge.append(other_phase)

    # Filter rows where ALL phases have matches (central AC)
    if other_phases_in_merge:
        all_matched_mask = merged[f'id_{reference_phase}'].notna()
        for other_phase in other_phases_in_merge:
            all_matched_mask &= merged[f'id_{other_phase}'].notna()

        central_rows = merged[all_matched_mask]

        for _, row in central_rows.iterrows():
            ref_id = row[f'id_{reference_phase}']
            if ref_id in used_match_ids:
                continue

            # Check if any other phase ID is already used
            skip = False
            for other_phase in other_phases_in_merge:
                if row[f'id_{other_phase}'] in used_match_ids:
                    skip = True
                    break
            if skip:
                continue

            # Mark all IDs as used
            used_match_ids.add(ref_id)
            for other_phase in other_phases_in_merge:
                used_match_ids.add(row[f'id_{other_phase}'])

            # Calculate totals
            total_magnitude = abs(row.get(f'mag_{reference_phase}', 0) or 0)
            durations = [row.get(f'dur_{reference_phase}', 0) or 0]
            off_times = []

            ref_off = row.get(f'off_end_{reference_phase}') or row.get(f'off_start_{reference_phase}')
            if pd.notna(ref_off):
                off_times.append(ref_off)

            phase_magnitudes = {reference_phase: int(abs(row.get(f'mag_{reference_phase}', 0) or 0))}

            for other_phase in other_phases_in_merge:
                mag = row.get(f'mag_{other_phase}', 0) or 0
                total_magnitude += abs(mag)
                phase_magnitudes[other_phase] = int(abs(mag))

                dur = row.get(f'dur_{other_phase}', 0)
                if dur:
                    durations.append(dur)

                off = row.get(f'off_end_{other_phase}') or row.get(f'off_start_{other_phase}')
                if pd.notna(off):
                    off_times.append(off)

            avg_duration = np.mean(durations) if durations else 0
            off_end = max(off_times) if off_times else ref_off
            ref_on_start = row['on_start']

            activation = {
                'date': ref_on_start.strftime('%Y-%m-%d'),
                'on_time': ref_on_start.strftime('%H:%M'),
                'off_time': off_end.strftime('%H:%M') if pd.notna(off_end) else '',
                'duration_minutes': int(round(avg_duration)),
                'total_magnitude': int(total_magnitude),
                'phases': [reference_phase] + other_phases_in_merge,
                'phase_magnitudes': phase_magnitudes,
            }
            central_activations.append(activation)

    # Sort by date and time
    central_activations.sort(key=lambda x: (x['date'], x['on_time']))

    # Group into sessions (compressor cycles within 1 hour = same session)
    central_sessions = _group_activations_into_sessions(central_activations, gap_threshold_minutes=60)

    result['central_ac']['activations'] = central_sessions  # Sessions, not individual cycles
    result['central_ac']['raw_activations'] = central_activations  # Keep raw data for reference
    result['central_ac']['total_count'] = len(central_sessions)  # Session count
    result['central_ac']['total_cycles'] = len(central_activations)  # Individual cycle count
    result['central_ac']['phases_used'] = active_phases
    result['has_central_ac'] = len(central_sessions) >= 3  # Need at least 3 sessions

    # Find regular AC on ALL phases (not just w1), not part of central AC
    # AC detection: two pools of candidates combined into sessions:
    # - "Initial run" candidates: 800W+, >= 15 min (no upper limit)
    # - "Cycling" candidates: 800W+, 3-30 min
    # Sessions are validated by _is_valid_ac_session (first >= 15 min, then 3+ more)
    regular_ac_by_phase = {}

    # AC cycle duration bounds (minutes)
    MIN_CYCLE_DURATION = 3
    MAX_CYCLE_DURATION = 30
    MIN_INITIAL_DURATION = 15  # First event in session must be >= this

    for phase in active_phases:
        if phase not in phase_matches:
            continue

        phase_df = phase_matches[phase].copy()

        # Two pools: cycling events (3-30 min) + initial run candidates (>= 15 min)
        # Combined with OR to allow long initial compressor runs
        not_used = ~phase_df['on_event_id'].isin(used_match_ids)
        high_power = phase_df['on_magnitude'].abs() >= 800
        is_cycle = (phase_df['duration'] >= MIN_CYCLE_DURATION) & (phase_df['duration'] <= MAX_CYCLE_DURATION)
        is_initial = phase_df['duration'] >= MIN_INITIAL_DURATION

        filtered = phase_df[
            not_used & high_power & (is_cycle | is_initial)
        ].copy()

        if filtered.empty:
            continue

        # Vectorized: prepare off_time column
        filtered['off_time_val'] = filtered['off_end'].fillna(filtered['off_start'])

        # Build activations list (still need a loop for dict creation, but much smaller dataset now)
        phase_activations = []
        for _, row in filtered.iterrows():
            on_start = row['on_start']
            off_end = row['off_time_val']
            activation = {
                'date': on_start.strftime('%Y-%m-%d'),
                'on_time': on_start.strftime('%H:%M'),
                'off_time': off_end.strftime('%H:%M') if pd.notna(off_end) else '',
                'duration_minutes': int(round(row['duration'])),
                'magnitude': int(abs(row['on_magnitude'])),
                'phase': phase,
            }
            phase_activations.append(activation)

        if phase_activations:
            # Sort by date and time
            phase_activations.sort(key=lambda x: (x['date'], x['on_time']))

            # Group into sessions (compressor cycles within 1 hour = same session)
            phase_sessions = _group_activations_into_sessions(phase_activations, gap_threshold_minutes=60)

            # Filter sessions using AC validation criteria
            valid_sessions = []
            valid_raw_activations = []
            for session in phase_sessions:
                raw_acts = session.get('_raw_activations', [])
                if _is_valid_ac_session(session, raw_acts):
                    # Remove internal field before storing
                    session_copy = {k: v for k, v in session.items() if not k.startswith('_')}
                    valid_sessions.append(session_copy)
                    valid_raw_activations.extend(raw_acts)

            if valid_sessions:
                regular_ac_by_phase[phase] = {
                    'activations': valid_sessions,  # Sessions, not individual cycles
                    'raw_activations': valid_raw_activations,  # Keep raw data for reference
                    'total_count': len(valid_sessions),  # Session count
                    'total_cycles': len(valid_raw_activations),  # Individual cycle count
                }

    # Keep backwards compatibility with old format (combine all into regular_ac)
    all_regular_activations = []
    all_raw_activations = []
    for phase_data in regular_ac_by_phase.values():
        all_regular_activations.extend(phase_data['activations'])
        all_raw_activations.extend(phase_data.get('raw_activations', []))
    all_regular_activations.sort(key=lambda x: (x['date'], x['on_time']))
    all_raw_activations.sort(key=lambda x: (x['date'], x['on_time']))

    result['regular_ac']['activations'] = all_regular_activations  # Sessions
    result['regular_ac']['raw_activations'] = all_raw_activations  # Individual cycles
    result['regular_ac']['total_count'] = len(all_regular_activations)  # Session count
    result['regular_ac']['total_cycles'] = len(all_raw_activations)  # Individual cycle count
    result['has_regular_ac'] = len(all_regular_activations) >= 3  # Need at least 3 sessions

    # New format: grouped by phase
    result['regular_ac_by_phase'] = regular_ac_by_phase

    return result


def analyze_device_usage_patterns(ac_detection: Dict[str, Any],
                                   boiler_detection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze device usage by season (winter/summer) and time of day (day/night).

    Args:
        ac_detection: Result from detect_ac_patterns
        boiler_detection: Result from detect_boiler_patterns

    Returns:
        Dictionary with seasonal and time-of-day breakdown for each device type
    """
    logger.debug("analyze_device_usage_patterns: ac_detection=%s, boiler_detection=%s",
                 bool(ac_detection), bool(boiler_detection))
    result = {
        'central_ac': {'seasonal': {}, 'time_of_day': {}},
        'regular_ac': {'seasonal': {}, 'time_of_day': {}},
        'boiler': {'seasonal': {}, 'time_of_day': {}},
    }

    # Define seasons (Northern Hemisphere, Israel)
    # Winter: Dec, Jan, Feb (months 12, 1, 2)
    # Summer: Jun, Jul, Aug (months 6, 7, 8)
    # Spring/Fall: Mar, Apr, May, Sep, Oct, Nov
    def get_season(date_str):
        """Get season from date string (YYYY-MM-DD)."""
        try:
            month = int(date_str.split('-')[1])
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [6, 7, 8]:
                return 'summer'
            elif month in [3, 4, 5]:
                return 'spring'
            else:
                return 'fall'
        except (ValueError, IndexError):
            return 'unknown'

    def get_time_of_day(time_str):
        """Get time of day from time string (HH:MM)."""
        try:
            hour = int(time_str.split(':')[0])
            if 6 <= hour < 18:
                return 'day'
            else:
                return 'night'
        except (ValueError, IndexError):
            return 'unknown'

    def analyze_activations(activations):
        """Analyze a list of activations by season and time of day."""
        seasonal = {'winter': 0, 'spring': 0, 'summer': 0, 'fall': 0}
        time_of_day = {'day': 0, 'night': 0}

        for act in activations:
            date = act.get('date', '')
            time = act.get('on_time', '')

            season = get_season(date)
            tod = get_time_of_day(time)

            if season in seasonal:
                seasonal[season] += 1
            if tod in time_of_day:
                time_of_day[tod] += 1

        return {'seasonal': seasonal, 'time_of_day': time_of_day}

    # Analyze Central AC
    if ac_detection:
        central_activations = ac_detection.get('central_ac', {}).get('activations', [])
        if central_activations:
            analysis = analyze_activations(central_activations)
            result['central_ac'] = analysis

        # Analyze Regular AC (all phases combined)
        regular_activations = ac_detection.get('regular_ac', {}).get('activations', [])
        if regular_activations:
            analysis = analyze_activations(regular_activations)
            result['regular_ac'] = analysis

    # Analyze Boiler
    if boiler_detection:
        boiler_activations = boiler_detection.get('boiler', {}).get('activations', [])
        if boiler_activations:
            analysis = analyze_activations(boiler_activations)
            result['boiler'] = analysis

    return result


