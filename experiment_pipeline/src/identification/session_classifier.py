"""
Session-level device classifier.

Classifies *sessions* (not individual ON→OFF events) into device types.

Priority order:
  1. **Boiler** — single long high-power event, single-phase, isolated
  2. **Central AC** — multi-phase synchronised compressor sessions
  3. **Regular AC** — single-phase compressor cycling (4+ cycles)
  4. **Unknown** — everything else

Adapted from the per-event logic in ``device_classifier.py`` to operate
on :class:`Session` and :class:`MultiPhaseSession` objects.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd

from .config import (
    BOILER_MIN_DURATION,
    BOILER_MIN_MAGNITUDE,
    BOILER_ISOLATION_WINDOW,
    AC_MIN_MAGNITUDE,
    AC_MIN_CYCLE_DURATION,
    AC_MAX_CYCLE_DURATION,
    AC_MIN_INITIAL_DURATION,
    AC_MIN_FOLLOWING_CYCLES,
    AC_MAX_MAGNITUDE_CV,
    AC_FILTER_WINDOW,
    AC_FILTER_MIN_CYCLES,
    AC_FILTER_MIN_CYCLE_MAG,
    AC_FILTER_MAG_RATIO,
    MULTI_PHASE_WINDOW,
    PHASES,
)
from .session_grouper import Session, MultiPhaseSession

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ClassifiedSession:
    """A session with device classification attached."""
    session: Union[Session, MultiPhaseSession]
    device_type: str          # boiler | central_ac | regular_ac | unknown
    reason: str               # human-readable classification reason


# ============================================================================
# Public API
# ============================================================================

def classify_sessions(
    sessions: List[Session],
    all_matches: pd.DataFrame,
) -> Dict[str, List[ClassifiedSession]]:
    """Classify all sessions into device types.

    Args:
        sessions: Single-phase sessions from :func:`group_into_sessions`.
        all_matches: Full (deduplicated) match DataFrame — used for isolation
                     and compressor-cycle checks around boiler candidates.

    Returns:
        Dict mapping device_type → list of :class:`ClassifiedSession`.
    """
    from .session_grouper import detect_phase_synchronized_groups

    result: Dict[str, List[ClassifiedSession]] = {
        'boiler': [],
        'central_ac': [],
        'regular_ac': [],
        'unknown': [],
    }

    classified_session_ids: set = set()

    # --- Step 1: Boiler ---------------------------------------------------
    for s in sessions:
        if _is_boiler_session(s, all_matches):
            result['boiler'].append(ClassifiedSession(
                session=s,
                device_type='boiler',
                reason=f"Single event ≥{BOILER_MIN_DURATION}min, ≥{BOILER_MIN_MAGNITUDE}W, isolated",
            ))
            classified_session_ids.add(s.session_id)

    n_boiler = len(result['boiler'])
    if n_boiler:
        logger.info(f"  Classified {n_boiler} boiler sessions")

    # --- Step 2: Central AC -----------------------------------------------
    unclassified_sessions = [s for s in sessions if s.session_id not in classified_session_ids]
    multi_phase = detect_phase_synchronized_groups(unclassified_sessions)

    for mp in multi_phase:
        result['central_ac'].append(ClassifiedSession(
            session=mp,
            device_type='central_ac',
            reason=f"Synchronised across {len(mp.phases)} phases",
        ))
        # Mark all constituent single-phase sessions
        for ps in mp.phase_sessions.values():
            classified_session_ids.add(ps.session_id)

    n_central = len(result['central_ac'])
    if n_central:
        logger.info(f"  Classified {n_central} central AC sessions")

    # --- Step 3: Regular AC -----------------------------------------------
    for s in sessions:
        if s.session_id in classified_session_ids:
            continue
        if _is_regular_ac_session(s):
            result['regular_ac'].append(ClassifiedSession(
                session=s,
                device_type='regular_ac',
                reason=f"{s.cycle_count} cycles, CV={s.magnitude_cv:.2f}",
            ))
            classified_session_ids.add(s.session_id)

    n_regular = len(result['regular_ac'])
    if n_regular:
        logger.info(f"  Classified {n_regular} regular AC sessions")

    # --- Step 4: Unknown --------------------------------------------------
    for s in sessions:
        if s.session_id not in classified_session_ids:
            result['unknown'].append(ClassifiedSession(
                session=s,
                device_type='unknown',
                reason="No pattern match",
            ))

    n_unknown = len(result['unknown'])
    logger.info(
        f"Classification: {n_boiler} boiler, {n_central} central_ac, "
        f"{n_regular} regular_ac, {n_unknown} unknown"
    )

    return result


# ============================================================================
# Boiler detection
# ============================================================================

def _is_boiler_session(session: Session, all_matches: pd.DataFrame) -> bool:
    """Check if a session qualifies as a boiler session.

    A boiler session is typically a single long, high-power, isolated event.
    """
    # Boiler sessions should be a single (or very few) long events
    if session.cycle_count > 2:
        return False

    # Duration and magnitude
    if session.avg_cycle_duration < BOILER_MIN_DURATION:
        return False
    if session.avg_magnitude < BOILER_MIN_MAGNITUDE:
        return False

    # Isolation check — no medium-duration events nearby on same phase
    phase_matches = all_matches[all_matches['phase'] == session.phase]
    if not phase_matches.empty:
        medium = phase_matches[
            (phase_matches['duration'] >= AC_MIN_CYCLE_DURATION) &
            (phase_matches['duration'] <= AC_MAX_CYCLE_DURATION)
        ]
        if not medium.empty:
            for event in session.events:
                event_start = pd.Timestamp(event['on_start'])
                time_diffs = (medium['on_start'] - event_start).abs()
                if hasattr(time_diffs.iloc[0], 'total_seconds'):
                    nearby_mask = (time_diffs.dt.total_seconds() / 60) <= BOILER_ISOLATION_WINDOW
                else:
                    nearby_mask = time_diffs <= BOILER_ISOLATION_WINDOW
                if nearby_mask.any():
                    return False

    # Multi-phase check — boilers are single-phase
    for event in session.events:
        event_start = pd.Timestamp(event['on_start'])
        for other_phase in PHASES:
            if other_phase == session.phase:
                continue
            other = all_matches[all_matches['phase'] == other_phase]
            if other.empty:
                continue
            time_diffs = (other['on_start'] - event_start).abs()
            if hasattr(time_diffs.iloc[0], 'total_seconds'):
                nearby_mask = (time_diffs.dt.total_seconds() / 60) <= MULTI_PHASE_WINDOW
            else:
                nearby_mask = time_diffs <= MULTI_PHASE_WINDOW
            if nearby_mask.any():
                return False

    # AC filter — check for compressor cycling nearby
    for event in session.events:
        if _has_nearby_compressor_cycles(event, all_matches, session.phase, session.avg_magnitude):
            return False

    return True


def _has_nearby_compressor_cycles(
    event: dict,
    all_matches: pd.DataFrame,
    phase: str,
    boiler_magnitude: float,
) -> bool:
    """Check if compressor cycles exist near a boiler candidate event."""
    phase_matches = all_matches[all_matches['phase'] == phase]
    if phase_matches.empty:
        return False

    event_start = pd.Timestamp(event['on_start'])
    event_end = pd.Timestamp(event.get('off_end') or event.get('off_start') or event['on_start'])

    window_start = event_start - pd.Timedelta(minutes=AC_FILTER_WINDOW)
    window_end = event_end + pd.Timedelta(minutes=AC_FILTER_WINDOW)

    nearby = phase_matches[
        (phase_matches['on_start'] >= window_start) &
        (phase_matches['on_start'] <= window_end)
    ]

    cycles = nearby[
        (nearby['duration'] >= AC_MIN_CYCLE_DURATION) &
        (nearby['duration'] <= AC_MAX_CYCLE_DURATION) &
        (nearby['on_magnitude'].abs() >= AC_FILTER_MIN_CYCLE_MAG) &
        (nearby['on_magnitude'].abs() >= boiler_magnitude * AC_FILTER_MAG_RATIO)
    ]

    return len(cycles) >= AC_FILTER_MIN_CYCLES


# ============================================================================
# Regular AC detection
# ============================================================================

def _is_regular_ac_session(session: Session) -> bool:
    """Check if a session qualifies as a regular AC compressor pattern.

    Criteria:
    - Average magnitude ≥ 800 W
    - 4+ cycles (1 initial ≥ 15 min + 3 following)
    - Magnitude CV ≤ 20%
    """
    if session.avg_magnitude < AC_MIN_MAGNITUDE:
        return False

    if session.cycle_count < (1 + AC_MIN_FOLLOWING_CYCLES):
        return False

    # First event must be the initial long run
    if not session.events:
        return False
    sorted_events = sorted(session.events, key=lambda e: e['on_start'])
    first_duration = sorted_events[0].get('duration', 0) or 0
    if first_duration < AC_MIN_INITIAL_DURATION:
        return False

    # Magnitude consistency
    if session.magnitude_cv > AC_MAX_MAGNITUDE_CV:
        return False

    return True
