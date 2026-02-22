"""
Guided cycle recovery — finds missed compressor cycles in the remaining signal.

After all detection iterations complete, this step:
1. Loads all matched events and groups them into sessions
2. Identifies AC-like sessions with enough matched cycles to form a template
3. Searches the remaining signal within those session windows at a lower threshold
4. Matches and extracts any found cycles, updating the remaining signal

This is conservative: requires >= 3 existing matched cycles in a session to
establish a template, and only searches within that session's time range.

Disabled by default (use_guided_recovery=False) until validated on real data.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================

def process_guided_recovery(
    output_path: str,
    house_id: str,
    threshold_schedule: List[int],
    config,
    run_logger,
) -> dict:
    """
    Guided recovery pass for missed compressor cycles.

    Runs after all detection iterations. Uses matched AC cycles as templates
    to find missed cycles in the remaining signal at a lower threshold.

    Args:
        output_path: Base experiment output path
        house_id: House identifier
        threshold_schedule: List of thresholds used in iterations
        config: ExperimentConfig with recovery parameters
        run_logger: Logger instance from pipeline runner

    Returns:
        Dict with recovery statistics: {'recovered': int, 'sessions_searched': int}
    """
    from identification.session_grouper import load_all_matches

    experiment_dir = Path(output_path)
    last_run = len(threshold_schedule) - 1
    recovery_factor = getattr(config, 'guided_recovery_threshold_factor', 0.6)
    min_cycles = getattr(config, 'guided_recovery_min_cycles', 3)

    # 1. Load all matches from all iterations
    all_matches = load_all_matches(experiment_dir, house_id, threshold_schedule)
    if all_matches.empty:
        run_logger.info("Recovery: no matches found, skipping")
        return {'recovered': 0, 'sessions_searched': 0}

    # 2. Group matches into sessions and find templates
    sessions = group_matches_into_sessions(all_matches, gap_minutes=30)
    templates = find_recovery_templates(
        sessions, min_cycles=min_cycles, recovery_factor=recovery_factor,
    )

    if not templates:
        run_logger.info("Recovery: no qualifying AC sessions found for recovery")
        return {'recovered': 0, 'sessions_searched': 0}

    run_logger.info(
        f"Recovery: found {len(templates)} qualifying sessions "
        f"({sum(t['cycle_count'] for t in templates)} total cycles)"
    )

    # 3. Load remaining data from last iteration and search for missed cycles
    total_recovered = 0
    sessions_searched = 0

    for template in templates:
        recovered = _recover_cycles_for_template(
            template, experiment_dir, house_id, last_run, run_logger,
        )
        total_recovered += recovered
        sessions_searched += 1

    run_logger.info(f"Recovery: recovered {total_recovered} cycles from {sessions_searched} sessions")
    return {'recovered': total_recovered, 'sessions_searched': sessions_searched}


# ============================================================================
# Session grouping (simplified version for recovery — no classification needed)
# ============================================================================

def group_matches_into_sessions(
    matches: pd.DataFrame,
    gap_minutes: int = 30,
) -> List[dict]:
    """
    Group matched events into sessions by phase and time proximity.

    A session is a group of matches on the same phase where consecutive
    matches are separated by less than gap_minutes.

    Args:
        matches: DataFrame with matched events (on_start, off_end, phase, on_magnitude)
        gap_minutes: Maximum gap between consecutive matches to group them

    Returns:
        List of session dicts with keys: phase, events, start, end,
        avg_magnitude, avg_duration, cycle_count
    """
    if matches.empty:
        return []

    sessions = []
    gap_td = pd.Timedelta(minutes=gap_minutes)

    for phase in ['w1', 'w2', 'w3']:
        phase_matches = matches[matches['phase'] == phase].copy()
        if phase_matches.empty:
            continue

        # Sort by on_start
        phase_matches = phase_matches.sort_values('on_start').reset_index(drop=True)

        # Group by time gap
        current_events = [phase_matches.iloc[0]]

        for i in range(1, len(phase_matches)):
            row = phase_matches.iloc[i]
            prev = current_events[-1]

            # Gap between previous match end and current match start
            prev_end = prev['off_end']
            curr_start = row['on_start']

            if (curr_start - prev_end) <= gap_td:
                current_events.append(row)
            else:
                # Finalize current session
                sessions.append(_build_session(phase, current_events))
                current_events = [row]

        # Don't forget the last session
        sessions.append(_build_session(phase, current_events))

    return sessions


def _build_session(phase: str, events: list) -> dict:
    """Build a session dict from a list of match rows."""
    magnitudes = [abs(float(e['on_magnitude'])) for e in events]
    durations = [float(e['duration']) for e in events]

    return {
        'phase': phase,
        'events': events,
        'start': events[0]['on_start'],
        'end': events[-1]['off_end'],
        'avg_magnitude': np.mean(magnitudes),
        'avg_duration': np.mean(durations),
        'cycle_count': len(events),
        'magnitude_cv': np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 1.0,
    }


# ============================================================================
# Template identification
# ============================================================================

def find_recovery_templates(
    sessions: List[dict],
    min_cycles: int = 3,
    recovery_factor: float = 0.6,
    min_magnitude: float = 500,
    max_magnitude: float = 3000,
    min_duration: float = 3,
    max_duration: float = 30,
    max_cv: float = 0.5,
) -> List[dict]:
    """
    Identify sessions that qualify as AC templates for cycle recovery.

    A qualifying session has:
    - At least min_cycles matched cycles
    - Average magnitude in AC range (500-3000W)
    - Average duration in compressor cycle range (3-30 min)
    - Consistent magnitudes (CV < 0.5)

    Args:
        sessions: List of session dicts from group_matches_into_sessions
        min_cycles: Minimum number of matched cycles required
        recovery_factor: Factor to compute recovery threshold (e.g., 0.6 = 60% of avg)
        min_magnitude: Minimum average magnitude for AC classification
        max_magnitude: Maximum average magnitude for AC classification
        min_duration: Minimum average duration (minutes) for compressor cycles
        max_duration: Maximum average duration (minutes) for compressor cycles
        max_cv: Maximum coefficient of variation for consistent magnitudes

    Returns:
        List of template dicts with recovery_threshold added
    """
    templates = []

    for session in sessions:
        if session['cycle_count'] < min_cycles:
            continue

        avg_mag = session['avg_magnitude']
        avg_dur = session['avg_duration']
        cv = session['magnitude_cv']

        if not (min_magnitude <= avg_mag <= max_magnitude):
            continue
        if not (min_duration <= avg_dur <= max_duration):
            continue
        if cv > max_cv:
            continue

        templates.append({
            'phase': session['phase'],
            'start': session['start'],
            'end': session['end'],
            'avg_magnitude': avg_mag,
            'avg_duration': avg_dur,
            'recovery_threshold': avg_mag * recovery_factor,
            'cycle_count': session['cycle_count'],
        })

    return templates


# ============================================================================
# Cycle recovery
# ============================================================================

def recover_cycles_from_remaining(
    remaining_data: pd.DataFrame,
    phase: str,
    recovery_threshold: float,
    session_start: pd.Timestamp,
    session_end: pd.Timestamp,
) -> List[dict]:
    """
    Search remaining signal for ON/OFF events above recovery threshold.

    Runs a simplified sharp detection on the remaining signal within the
    session time window. Does not use gradual detection or expansion —
    recovery is conservative and only catches clear missed cycles.

    Args:
        remaining_data: DataFrame with timestamp index and remaining power columns
        phase: Phase column name (w1, w2, w3)
        recovery_threshold: Minimum magnitude for detection
        session_start: Start of the search window
        session_end: End of the search window

    Returns:
        List of matched cycle dicts (on_start, on_end, off_start, off_end,
        magnitude, phase) for successfully matched ON→OFF pairs
    """
    remaining_col = f'remaining_power_{phase}'
    if remaining_col not in remaining_data.columns:
        return []

    # Window the data to session time range (with 10-min buffer)
    buffer = pd.Timedelta(minutes=10)
    window_start = session_start - buffer
    window_end = session_end + buffer

    windowed = remaining_data.loc[window_start:window_end].copy()
    if len(windowed) < 5:
        return []

    # Compute diffs on remaining signal
    remaining_values = windowed[remaining_col]
    diffs = remaining_values.diff()

    # Find ON events (positive jumps >= recovery_threshold)
    on_mask = diffs >= recovery_threshold
    on_times = diffs[on_mask].index.tolist()

    # Find OFF events (negative drops <= -recovery_threshold)
    off_mask = diffs <= -recovery_threshold
    off_times = diffs[off_mask].index.tolist()

    if not on_times or not off_times:
        return []

    # Simple greedy matching: for each ON, find the closest following OFF
    # with similar magnitude
    matched_cycles = []
    used_off = set()

    for on_ts in on_times:
        on_mag = float(diffs.loc[on_ts])

        best_off = None
        best_diff = float('inf')

        for off_ts in off_times:
            if off_ts in used_off:
                continue
            if off_ts <= on_ts:
                continue

            # Duration check: compressor cycles are 3-30 minutes
            duration = (off_ts - on_ts).total_seconds() / 60
            if duration < 2 or duration > 45:
                continue

            off_mag = abs(float(diffs.loc[off_ts]))
            mag_diff = abs(on_mag - off_mag)

            # Magnitude similarity check (within 350W, same as Stage 1)
            if mag_diff <= 350 and mag_diff < best_diff:
                best_diff = mag_diff
                best_off = off_ts

        if best_off is not None:
            used_off.add(best_off)
            matched_cycles.append({
                'on_start': on_ts,
                'on_end': on_ts,
                'off_start': best_off,
                'off_end': best_off,
                'magnitude': float(diffs.loc[on_ts]),
                'phase': phase,
            })

    return matched_cycles


# ============================================================================
# Internal helpers
# ============================================================================

def _recover_cycles_for_template(
    template: dict,
    experiment_dir: Path,
    house_id: str,
    last_run: int,
    run_logger,
) -> int:
    """
    Search for missed cycles in the remaining signal for one template session.

    Loads summarized data from the last iteration and runs simplified detection
    within the session's time window at the recovery threshold.

    Returns:
        Number of cycles recovered
    """
    from core import load_power_data

    phase = template['phase']
    recovery_threshold = template['recovery_threshold']

    # Find summarized data from last run
    summarized_dir = experiment_dir / f"run_{last_run}" / f"house_{house_id}" / "summarized"
    if not summarized_dir.exists():
        run_logger.warning(f"Recovery: summarized dir not found for run {last_run}")
        return 0

    summarized_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.pkl"))
    if not summarized_files:
        run_logger.warning(f"Recovery: no summarized files found in {summarized_dir}")
        return 0

    total_recovered = 0

    for summ_file in summarized_files:
        try:
            data = load_power_data(summ_file)
        except Exception as e:
            run_logger.warning(f"Recovery: failed to load {summ_file}: {e}")
            continue

        if data.empty:
            continue

        data_indexed = data.set_index('timestamp')

        # Search for missed cycles
        cycles = recover_cycles_from_remaining(
            data_indexed, phase, recovery_threshold,
            template['start'], template['end'],
        )

        if cycles:
            run_logger.info(
                f"Recovery: found {len(cycles)} missed cycles on {phase} "
                f"(threshold={recovery_threshold:.0f}W, session has "
                f"{template['cycle_count']} existing cycles)"
            )
            total_recovered += len(cycles)

            # Extract recovered cycles from remaining
            remaining_col = f'remaining_power_{phase}'
            for cycle in cycles:
                _extract_cycle_from_remaining(
                    data_indexed, remaining_col, cycle, run_logger,
                )

            # Save updated remaining data
            data_updated = data_indexed.reset_index()
            data_updated.to_pickle(summ_file)

    return total_recovered


def _extract_cycle_from_remaining(
    data_indexed: pd.DataFrame,
    remaining_col: str,
    cycle: dict,
    run_logger,
) -> None:
    """
    Extract a recovered cycle from the remaining signal.

    Uses a simple approach: subtract the cycle magnitude from the remaining
    signal for the duration of the cycle. Clips to zero to prevent negatives.

    Args:
        data_indexed: DataFrame with timestamp index (modified in place)
        remaining_col: Name of the remaining power column
        cycle: Cycle dict with on_start, off_end, magnitude
        run_logger: Logger instance
    """
    on_start = cycle['on_start']
    off_end = cycle['off_end']
    magnitude = abs(cycle['magnitude'])

    # Find the range of timestamps for this cycle
    mask = (data_indexed.index >= on_start) & (data_indexed.index <= off_end)

    if mask.sum() == 0:
        return

    # Subtract magnitude from remaining, clip to zero
    data_indexed.loc[mask, remaining_col] = (
        data_indexed.loc[mask, remaining_col] - magnitude
    ).clip(lower=0)

    duration = (off_end - on_start).total_seconds() / 60
    run_logger.info(
        f"  Extracted cycle: {on_start} → {off_end} "
        f"({duration:.0f}min, {magnitude:.0f}W)"
    )
