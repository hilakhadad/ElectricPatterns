"""
AC device classifier.

Identifies regular AC compressor cycling patterns from remaining events
after boiler identification. Handles anchor detection, cycling sequence
building, and regular AC confidence scoring.
"""
import logging
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ..config import (
    AC_MIN_MAGNITUDE,
    AC_MIN_CYCLE_DURATION,
    AC_MAX_CYCLE_DURATION,
    AC_MIN_INITIAL_DURATION,
    AC_MIN_FOLLOWING_CYCLES,
    AC_MAX_MAGNITUDE_CV,
    DEFAULT_SESSION_GAP_MINUTES,
    PHASES,
)
from ..session_grouper import Session
from .scoring_utils import _linear_score, _inverse_linear_score

logger = logging.getLogger(__name__)


def _identify_ac_sessions(
    remaining: pd.DataFrame,
) -> tuple:
    """Find AC cycling patterns in remaining events, then check cross-phase overlap.

    Algorithm:
        1. For each phase, find cycling sequences (anchor + following cycles)
        2. Validate each sequence passes AC criteria
        3. Cross-phase check: independently-identified AC sessions overlapping?
           -> central_ac vs regular_ac
        4. Remove classified events from pool

    Returns:
        (central_ac_classified, regular_ac_classified, remaining_matches)
    """
    from .central_ac_classifier import _check_central_ac_overlap

    if remaining.empty:
        return [], [], remaining

    # Ensure timestamps
    for col in ['on_start', 'off_end', 'off_start']:
        if col in remaining.columns and not pd.api.types.is_datetime64_any_dtype(remaining[col]):
            remaining[col] = pd.to_datetime(remaining[col])

    # Find cycling sequences per phase
    ac_sessions_by_phase: Dict[str, List[Session]] = {}
    consumed_indices: Set[int] = set()

    for phase in PHASES:
        phase_df = remaining[remaining['phase'] == phase]
        if phase_df.empty:
            continue

        sessions, used_indices = _find_cycling_sequences(phase_df, phase)
        if sessions:
            ac_sessions_by_phase[phase] = sessions
            consumed_indices.update(used_indices)

    # Cross-phase overlap check for central AC
    central_list, regular_list = _check_central_ac_overlap(ac_sessions_by_phase, remaining)

    # Collect all consumed event indices from classified AC sessions
    from ..session_grouper import MultiPhaseSession
    ac_consumed = set()
    for cs in central_list + regular_list:
        session = cs.session
        if isinstance(session, MultiPhaseSession):
            for ps in session.phase_sessions.values():
                ac_consumed.update(_find_event_indices(ps.events, remaining))
        else:
            ac_consumed.update(_find_event_indices(session.events, remaining))

    remaining_out = remaining.loc[~remaining.index.isin(ac_consumed)].reset_index(drop=True)
    return central_list, regular_list, remaining_out


def _find_event_indices(events: List[dict], matches: pd.DataFrame) -> Set[int]:
    """Find DataFrame indices corresponding to event dicts (by on_start timestamp)."""
    indices = set()
    for ev in events:
        ts = pd.Timestamp(ev['on_start'])
        phase = ev.get('phase', '')
        mask = (matches['on_start'] == ts) & (matches['phase'] == phase)
        indices.update(matches.index[mask].tolist())
    return indices


def _find_cycling_sequences(
    phase_df: pd.DataFrame,
    phase: str,
) -> Tuple[List[Session], Set[int]]:
    """Find AC cycling patterns within a single phase's events.

    Algorithm:
        - Sort events by on_start
        - Find anchor events (duration >= AC_MIN_INITIAL_DURATION, magnitude >= AC_MIN_MAGNITUDE)
        - For each anchor, collect following events:
          * duration 3-30 min
          * gap from previous event < DEFAULT_SESSION_GAP_MINUTES
          * magnitude within reasonable range
        - Require at least AC_MIN_FOLLOWING_CYCLES following events
        - Build Session objects from qualifying sequences

    Returns:
        (sessions, consumed_indices)
    """
    from .central_ac_classifier import _has_ac_pattern_in_dominant_iteration

    sorted_df = phase_df.sort_values('on_start')
    used_indices: Set[int] = set()
    sessions: List[Session] = []

    indices = sorted_df.index.tolist()
    rows = [sorted_df.loc[i] for i in indices]

    for i, (idx, row) in enumerate(zip(indices, rows)):
        if idx in used_indices:
            continue

        duration = row.get('duration', 0) or 0
        magnitude = abs(row.get('on_magnitude', 0) or 0)

        # Try this event as an anchor
        if duration < AC_MIN_INITIAL_DURATION or magnitude < AC_MIN_MAGNITUDE:
            continue

        # Collect following events
        sequence_indices = [idx]
        sequence_events = [row.to_dict()]
        prev_end = pd.Timestamp(row.get('off_end') or row.get('off_start') or row['on_start'])
        magnitudes = [magnitude]

        for j in range(i + 1, len(indices)):
            next_idx = indices[j]
            if next_idx in used_indices:
                continue

            next_row = rows[j]
            next_start = pd.Timestamp(next_row['on_start'])
            next_dur = next_row.get('duration', 0) or 0
            next_mag = abs(next_row.get('on_magnitude', 0) or 0)

            # Gap check
            gap = (next_start - prev_end).total_seconds() / 60
            if gap > DEFAULT_SESSION_GAP_MINUTES:
                break

            # Duration in AC cycle range
            if next_dur < AC_MIN_CYCLE_DURATION or next_dur > AC_MAX_CYCLE_DURATION:
                continue

            # Magnitude check: reasonable magnitude
            if next_mag < AC_MIN_MAGNITUDE:
                continue

            sequence_indices.append(next_idx)
            sequence_events.append(next_row.to_dict())
            magnitudes.append(next_mag)
            prev_end = pd.Timestamp(next_row.get('off_end') or next_row.get('off_start') or next_row['on_start'])

        # Check if we have enough following cycles (excluding anchor)
        following_count = len(sequence_events) - 1
        if following_count < AC_MIN_FOLLOWING_CYCLES:
            continue

        # Magnitude CV check
        if len(magnitudes) >= 2 and np.mean(magnitudes) > 0:
            cv = float(np.std(magnitudes) / np.mean(magnitudes))
            if cv > AC_MAX_MAGNITUDE_CV:
                # Fall back: check dominant iteration
                session_temp = _build_session_from_events(sequence_events, phase)
                if not _has_ac_pattern_in_dominant_iteration(session_temp):
                    continue

        # Build session
        session = _build_session_from_events(sequence_events, phase)
        sessions.append(session)
        used_indices.update(sequence_indices)

    return sessions, used_indices


def _build_session_from_events(events: List[dict], phase: str) -> Session:
    """Build a Session from a list of event dicts."""
    from ..session_grouper import _build_session_from_dicts
    return _build_session_from_dicts(events, phase)


def _session_ac_likeness(session: Session) -> float:
    """Score how AC-like a single-phase session is (0.0-1.0).

    Used as a component of central AC confidence.
    Considers: cycle count, magnitude, cycle duration range, magnitude
    consistency, and monotonicity of magnitude trend.
    """
    from ..config import (
        CENTRAL_AC_MIN_CYCLES,
        CENTRAL_AC_MIN_MAGNITUDE,
        CENTRAL_AC_MIN_CYCLE_DURATION,
        CENTRAL_AC_MAX_CYCLE_DURATION,
        CENTRAL_AC_MAX_MAGNITUDE_CV,
        CENTRAL_AC_MAX_DURATION_CV,
        CENTRAL_AC_MAX_GAP_CV,
    )
    from .scoring_utils import _magnitude_monotonicity, _cycling_regularity

    scores = []

    # Cycle count: 4 -> 0.5, >=10 -> 1.0
    scores.append(_linear_score(session.cycle_count, CENTRAL_AC_MIN_CYCLES, 10))

    # Magnitude: 800W -> 0.5, >=1500W -> 1.0
    scores.append(_linear_score(session.avg_magnitude, CENTRAL_AC_MIN_MAGNITUDE, 1500))

    # Magnitude CV: lower is better. CV=0.30 -> 0.5, CV=0 -> 1.0
    scores.append(_inverse_linear_score(session.magnitude_cv, 0, CENTRAL_AC_MAX_MAGNITUDE_CV))

    # Fraction of events with duration in AC range (3-30 min)
    if session.events:
        durations = [
            e.get('duration', 0) or 0
            for e in session.events
            if (e.get('duration') or 0) > 0
        ]
        if durations:
            in_range = sum(
                1 for d in durations
                if CENTRAL_AC_MIN_CYCLE_DURATION <= d <= CENTRAL_AC_MAX_CYCLE_DURATION
            )
            frac_in_range = in_range / len(durations)
            scores.append(max(0.5, frac_in_range))
        else:
            scores.append(0.5)
    else:
        scores.append(0.5)

    # Monotonicity: AC magnitudes trend in one direction (not zig-zag)
    mono = _magnitude_monotonicity(session)
    scores.append(_linear_score(mono, 0.5, 0.85))

    # Cycling regularity: consistent durations and gaps
    dur_cv, gap_cv = _cycling_regularity(session)
    scores.append(_inverse_linear_score(dur_cv, 0, CENTRAL_AC_MAX_DURATION_CV))
    scores.append(_inverse_linear_score(gap_cv, 0, CENTRAL_AC_MAX_GAP_CV))

    return float(np.mean(scores))


def _find_anchor_duration(session: Session) -> float:
    """Return the duration of the first event with duration >= AC_MIN_INITIAL_DURATION.

    Falls back to the longest event duration if no event meets the threshold.
    """
    if not session.events:
        return 0
    sorted_events = sorted(session.events, key=lambda e: e['on_start'])
    for e in sorted_events:
        dur = e.get('duration', 0) or 0
        if dur >= AC_MIN_INITIAL_DURATION:
            return dur
    # No anchor found -- return longest event
    return max((e.get('duration', 0) or 0) for e in sorted_events)


def _regular_ac_confidence(session: Session) -> tuple:
    """Compute confidence score for a regular AC session.

    Factors:
    - cycle_count: 4 cycles -> 0.5, >=10 cycles -> 1.0
    - magnitude_cv: CV=0.20 -> 0.5, CV<=0.05 -> 1.0
    - initial_duration: anchor event length (>=15min -> 0.5, >=30min -> 1.0)
    - magnitude: 800W -> 0.5, >=1500W -> 1.0
    """
    breakdown = {}

    # Cycle count -- more cycles = stronger pattern
    breakdown['cycle_count'] = round(_linear_score(
        session.cycle_count, 1 + AC_MIN_FOLLOWING_CYCLES, 10
    ), 2)

    # Magnitude CV -- lower = more consistent
    breakdown['magnitude_cv'] = round(_inverse_linear_score(
        session.magnitude_cv, 0.05, AC_MAX_MAGNITUDE_CV
    ), 2)

    # Initial duration -- anchor event (first long activation, not necessarily first event)
    anchor_dur = _find_anchor_duration(session)
    breakdown['initial_duration'] = round(_linear_score(
        anchor_dur, AC_MIN_INITIAL_DURATION, AC_MIN_INITIAL_DURATION * 2
    ), 2)

    # Magnitude -- higher magnitude = clearer signal
    breakdown['magnitude'] = round(_linear_score(
        session.avg_magnitude, AC_MIN_MAGNITUDE, AC_MIN_MAGNITUDE * 2
    ), 2)

    confidence = round(float(np.mean(list(breakdown.values()))), 2)
    return confidence, breakdown


def _is_regular_ac_session(session: Session) -> bool:
    """Check if a session qualifies as a regular AC compressor pattern.

    Criteria:
    - Average magnitude >= 800 W
    - 4+ cycles (1 initial >= 15 min + 3 following)
    - Magnitude CV <= 20% (overall or dominant iteration)
    """
    from .central_ac_classifier import _has_ac_pattern_in_dominant_iteration

    if session.avg_magnitude < AC_MIN_MAGNITUDE:
        return False

    if session.cycle_count < (1 + AC_MIN_FOLLOWING_CYCLES):
        return False

    if not session.events:
        return False
    sorted_events = sorted(session.events, key=lambda e: e['on_start'])
    anchor_idx = None
    for i, e in enumerate(sorted_events):
        dur = e.get('duration', 0) or 0
        if dur >= AC_MIN_INITIAL_DURATION:
            anchor_idx = i
            break

    if anchor_idx is not None:
        following_events = sorted_events[anchor_idx:]
        if len(following_events) < (1 + AC_MIN_FOLLOWING_CYCLES):
            return False

        mags = [abs(float(e.get('on_magnitude', 0))) for e in following_events]
        if not mags or np.mean(mags) <= 0:
            return False
        anchor_cv = float(np.std(mags) / np.mean(mags))

        if anchor_cv <= AC_MAX_MAGNITUDE_CV:
            return True

    return _has_ac_pattern_in_dominant_iteration(session)
