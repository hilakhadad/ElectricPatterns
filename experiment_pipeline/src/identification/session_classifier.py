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
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
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
    CENTRAL_AC_SYNC_TOLERANCE,
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
    confidence: float = 0.0   # 0–1 overall confidence score
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)


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
            conf, breakdown = _boiler_confidence(s, all_matches)
            result['boiler'].append(ClassifiedSession(
                session=s,
                device_type='boiler',
                reason=f"Single event ≥{BOILER_MIN_DURATION}min, ≥{BOILER_MIN_MAGNITUDE}W, isolated",
                confidence=conf,
                confidence_breakdown=breakdown,
            ))
            classified_session_ids.add(s.session_id)

    # Boiler phase exclusivity: a household has ONE boiler on ONE phase.
    # Keep only the dominant phase, demote the rest to unknown (Step 4).
    result['boiler'], demoted = _enforce_boiler_phase_exclusivity(result['boiler'])
    classified_session_ids = {cs.session.session_id for cs in result['boiler']}

    n_boiler = len(result['boiler'])
    if n_boiler:
        logger.info(f"  Classified {n_boiler} boiler sessions (demoted {len(demoted)} from other phases)")

    # --- Step 2: Central AC -----------------------------------------------
    unclassified_sessions = [s for s in sessions if s.session_id not in classified_session_ids]
    multi_phase = detect_phase_synchronized_groups(unclassified_sessions)

    for mp in multi_phase:
        conf, breakdown = _central_ac_confidence(mp)
        result['central_ac'].append(ClassifiedSession(
            session=mp,
            device_type='central_ac',
            reason=f"Synchronised across {len(mp.phases)} phases",
            confidence=conf,
            confidence_breakdown=breakdown,
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
            conf, breakdown = _regular_ac_confidence(s)
            result['regular_ac'].append(ClassifiedSession(
                session=s,
                device_type='regular_ac',
                reason=f"{s.cycle_count} cycles, CV={s.magnitude_cv:.2f}",
                confidence=conf,
                confidence_breakdown=breakdown,
            ))
            classified_session_ids.add(s.session_id)

    n_regular = len(result['regular_ac'])
    if n_regular:
        logger.info(f"  Classified {n_regular} regular AC sessions")

    # --- Step 4: Unknown --------------------------------------------------
    # Add demoted boiler sessions from non-dominant phases
    for cs in demoted:
        result['unknown'].append(ClassifiedSession(
            session=cs.session,
            device_type='unknown',
            reason=f"Boiler-like but on non-dominant phase ({cs.session.phase})",
        ))
        classified_session_ids.add(cs.session.session_id)

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
# Boiler phase exclusivity
# ============================================================================

def _enforce_boiler_phase_exclusivity(
    boiler_sessions: List[ClassifiedSession],
) -> tuple:
    """A household has ONE boiler connected to ONE phase.

    If boiler sessions appear on multiple phases, keep only those on the
    dominant phase (most sessions).  Sessions on other phases are returned
    as ``demoted`` so the caller can reclassify them as unknown.

    Returns:
        (kept, demoted) — two lists of ClassifiedSession.
    """
    if not boiler_sessions:
        return boiler_sessions, []

    # Count sessions per phase
    phase_counts = Counter(cs.session.phase for cs in boiler_sessions)

    if len(phase_counts) <= 1:
        return boiler_sessions, []

    dominant_phase = phase_counts.most_common(1)[0][0]
    kept = [cs for cs in boiler_sessions if cs.session.phase == dominant_phase]
    demoted = [cs for cs in boiler_sessions if cs.session.phase != dominant_phase]

    if demoted:
        logger.info(
            f"  Boiler phase exclusivity: dominant={dominant_phase} "
            f"({len(kept)} sessions), demoted {len(demoted)} from "
            f"{sorted(set(cs.session.phase for cs in demoted))}"
        )

    return kept, demoted


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
# Confidence scoring
# ============================================================================

def _linear_score(value: float, low: float, high: float) -> float:
    """Map value from [low, high] → [0.5, 1.0]. Below low → 0.5, above high → 1.0."""
    if value >= high:
        return 1.0
    if value <= low:
        return 0.5
    return 0.5 + 0.5 * (value - low) / (high - low)


def _inverse_linear_score(value: float, low: float, high: float) -> float:
    """Map value from [low, high] → [1.0, 0.5]. Lower is better."""
    if value <= low:
        return 1.0
    if value >= high:
        return 0.5
    return 1.0 - 0.5 * (value - low) / (high - low)


def _boiler_confidence(
    session: Session,
    all_matches: pd.DataFrame,
) -> tuple:
    """Compute confidence score for a boiler session.

    Factors:
    - duration: 25min → 0.5, ≥50min → 1.0
    - magnitude: 1500W → 0.5, ≥3000W → 1.0
    - isolation: distance to nearest medium event (farther = higher)
    - cycle_count: 1 event → 1.0, 2 events → 0.6
    """
    breakdown = {}

    # Duration — how far above minimum
    breakdown['duration'] = round(_linear_score(
        session.avg_cycle_duration, BOILER_MIN_DURATION, BOILER_MIN_DURATION * 2
    ), 2)

    # Magnitude — how far above minimum
    breakdown['magnitude'] = round(_linear_score(
        session.avg_magnitude, BOILER_MIN_MAGNITUDE, BOILER_MIN_MAGNITUDE * 2
    ), 2)

    # Cycle count — single event is most boiler-like
    breakdown['cycle_count'] = 1.0 if session.cycle_count == 1 else 0.6

    # Isolation — minimum distance to nearest medium-duration event
    min_dist = _min_distance_to_medium_events(session, all_matches)
    if min_dist is None:
        breakdown['isolation'] = 1.0  # no medium events at all — perfect isolation
    else:
        # 0 min distance → 0.5, ≥BOILER_ISOLATION_WINDOW → 1.0
        breakdown['isolation'] = round(_linear_score(
            min_dist, 0, BOILER_ISOLATION_WINDOW
        ), 2)

    confidence = round(float(np.mean(list(breakdown.values()))), 2)
    return confidence, breakdown


def _min_distance_to_medium_events(
    session: Session,
    all_matches: pd.DataFrame,
) -> Optional[float]:
    """Return minimum distance (minutes) to nearest medium-duration event on same phase."""
    phase_matches = all_matches[all_matches['phase'] == session.phase]
    if phase_matches.empty:
        return None

    medium = phase_matches[
        (phase_matches['duration'] >= AC_MIN_CYCLE_DURATION) &
        (phase_matches['duration'] <= AC_MAX_CYCLE_DURATION)
    ]
    if medium.empty:
        return None

    min_dist = float('inf')
    for event in session.events:
        event_start = pd.Timestamp(event['on_start'])
        time_diffs = (medium['on_start'] - event_start).abs()
        if hasattr(time_diffs.iloc[0], 'total_seconds'):
            distances = time_diffs.dt.total_seconds() / 60
        else:
            distances = time_diffs
        min_dist = min(min_dist, float(distances.min()))

    return min_dist if min_dist != float('inf') else None


def _central_ac_confidence(mp: MultiPhaseSession) -> tuple:
    """Compute confidence score for a central AC session.

    Factors:
    - phase_count: 2 phases → 0.7, 3 phases → 1.0
    - sync_quality: how tightly the phases overlap in time
    - magnitude_balance: how similar the magnitudes are across phases
    """
    breakdown = {}

    # Phase count — 3 phases is ideal for Israeli 3-phase systems
    n_phases = len(mp.phases)
    breakdown['phase_count'] = 1.0 if n_phases >= 3 else 0.7

    # Sync quality — how close the start times are across phases
    starts = [ps.start for ps in mp.phase_sessions.values()]
    if len(starts) >= 2:
        start_spread = (max(starts) - min(starts)).total_seconds() / 60
        # 0 min spread → 1.0, ≥CENTRAL_AC_SYNC_TOLERANCE → 0.5
        breakdown['sync_quality'] = round(_inverse_linear_score(
            start_spread, 0, CENTRAL_AC_SYNC_TOLERANCE
        ), 2)
    else:
        breakdown['sync_quality'] = 0.5

    # Magnitude balance — CV of magnitudes across phases (lower = more balanced)
    mags = [ps.avg_magnitude for ps in mp.phase_sessions.values()]
    if len(mags) >= 2 and np.mean(mags) > 0:
        mag_cv = float(np.std(mags) / np.mean(mags))
        # CV=0 → 1.0, CV≥0.5 → 0.5
        breakdown['magnitude_balance'] = round(_inverse_linear_score(mag_cv, 0, 0.5), 2)
    else:
        breakdown['magnitude_balance'] = 0.5

    confidence = round(float(np.mean(list(breakdown.values()))), 2)
    return confidence, breakdown


def _regular_ac_confidence(session: Session) -> tuple:
    """Compute confidence score for a regular AC session.

    Factors:
    - cycle_count: 4 cycles → 0.5, ≥10 cycles → 1.0
    - magnitude_cv: CV=0.20 → 0.5, CV≤0.05 → 1.0
    - initial_duration: 15min → 0.5, ≥30min → 1.0
    - magnitude: 800W → 0.5, ≥1500W → 1.0
    """
    breakdown = {}

    # Cycle count — more cycles = stronger pattern
    breakdown['cycle_count'] = round(_linear_score(
        session.cycle_count, 1 + AC_MIN_FOLLOWING_CYCLES, 10
    ), 2)

    # Magnitude CV — lower = more consistent
    breakdown['magnitude_cv'] = round(_inverse_linear_score(
        session.magnitude_cv, 0.05, AC_MAX_MAGNITUDE_CV
    ), 2)

    # Initial duration — first activation length
    sorted_events = sorted(session.events, key=lambda e: e['on_start'])
    first_dur = sorted_events[0].get('duration', 0) or 0 if sorted_events else 0
    breakdown['initial_duration'] = round(_linear_score(
        first_dur, AC_MIN_INITIAL_DURATION, AC_MIN_INITIAL_DURATION * 2
    ), 2)

    # Magnitude — higher magnitude = clearer signal
    breakdown['magnitude'] = round(_linear_score(
        session.avg_magnitude, AC_MIN_MAGNITUDE, AC_MIN_MAGNITUDE * 2
    ), 2)

    confidence = round(float(np.mean(list(breakdown.values()))), 2)
    return confidence, breakdown


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
