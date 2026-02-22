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
    CENTRAL_AC_MIN_MAGNITUDE,
    CENTRAL_AC_MIN_CYCLES,
    CENTRAL_AC_MIN_CYCLE_DURATION,
    CENTRAL_AC_MAX_CYCLE_DURATION,
    CENTRAL_AC_MAX_MAGNITUDE_CV,
    CENTRAL_AC_MAX_DURATION_CV,
    CENTRAL_AC_MAX_GAP_CV,
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

    # Pre-filter: only sessions with AC-like cycling can be central AC candidates
    ac_candidates = [s for s in unclassified_sessions if _is_central_ac_candidate(s)]

    if ac_candidates:
        logger.info(
            f"  Central AC: {len(ac_candidates)}/{len(unclassified_sessions)} "
            f"sessions are AC candidates"
        )

    multi_phase = detect_phase_synchronized_groups(ac_candidates)

    for mp in multi_phase:
        conf, breakdown = _central_ac_confidence(mp)
        result['central_ac'].append(ClassifiedSession(
            session=mp,
            device_type='central_ac',
            reason=(
                f"Synchronised across {len(mp.phases)} phases, "
                f"each showing AC-like cycling"
            ),
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
        # Passed boiler classification → not_boiler is low (close to boiler)
        demoted_breakdown = {
            'not_boiler': round(1.0 - cs.confidence, 2),
            'not_ac': 1.0,  # not checked, it was a valid boiler
        }
        demoted_breakdown.update({f'boiler_{k}': v for k, v in cs.confidence_breakdown.items()})
        result['unknown'].append(ClassifiedSession(
            session=cs.session,
            device_type='unknown',
            reason=(
                f"not_boiler: {1.0 - cs.confidence:.0%} (non-dominant phase {cs.session.phase}) "
                f"| not_ac: 100%"
            ),
            confidence=round(1.0 - cs.confidence, 2),
            confidence_breakdown=demoted_breakdown,
        ))
        classified_session_ids.add(cs.session.session_id)

    for s in sessions:
        if s.session_id not in classified_session_ids:
            conf, breakdown = _unknown_confidence(s, all_matches)
            reason = _unknown_reason(s, breakdown)
            result['unknown'].append(ClassifiedSession(
                session=s,
                device_type='unknown',
                reason=reason,
                confidence=conf,
                confidence_breakdown=breakdown,
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
    # (excluding the session's own events to avoid self-detection)
    phase_matches = all_matches[all_matches['phase'] == session.phase]
    if not phase_matches.empty:
        session_event_starts = {pd.Timestamp(e['on_start']) for e in session.events}
        medium = phase_matches[
            (phase_matches['duration'] >= AC_MIN_CYCLE_DURATION) &
            (phase_matches['duration'] <= AC_MAX_CYCLE_DURATION) &
            (~phase_matches['on_start'].isin(session_event_starts))
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

    # NOTE: Multi-phase check is NOT done here.  Boilers are single-phase,
    # but checking raw matches for simultaneous events on other phases causes
    # too many false negatives (unrelated appliances on other phases block
    # legitimate boilers).  Instead, multi-phase overlap is handled as a
    # post-processing step in _enforce_boiler_phase_exclusivity(), which
    # keeps only the dominant phase and demotes time-overlapping "boilers"
    # from other phases.

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
        (phase_matches['on_start'] <= window_end) &
        (phase_matches['on_start'] != event_start)  # exclude self
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

    # Exclude session's own events
    session_event_starts = {pd.Timestamp(e['on_start']) for e in session.events}
    medium = phase_matches[
        (phase_matches['duration'] >= AC_MIN_CYCLE_DURATION) &
        (phase_matches['duration'] <= AC_MAX_CYCLE_DURATION) &
        (~phase_matches['on_start'].isin(session_event_starts))
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


def _magnitude_monotonicity(session: Session) -> float:
    """Score how monotonic the magnitude trend is across events (0.0–1.0).

    AC compressors show a consistent or gradually-trending magnitude:
    people set a temperature and the compressor stabilises.  If they adjust
    the thermostat, magnitude shifts once — but never zig-zags.

    Returns the fraction of consecutive magnitude changes that follow the
    dominant direction.  A perfectly monotonic sequence returns 1.0, a
    random sequence returns ~0.5.

    Requires ≥3 events (≥2 consecutive differences).  Returns 1.0 for
    fewer events (too little data to penalise).
    """
    if not session.events or len(session.events) < 3:
        return 1.0

    sorted_events = sorted(session.events, key=lambda e: e.get('on_start', ''))
    magnitudes = [abs(float(e.get('on_magnitude', 0))) for e in sorted_events]

    diffs = [magnitudes[i + 1] - magnitudes[i] for i in range(len(magnitudes) - 1)]
    if not diffs:
        return 1.0

    n_up = sum(1 for d in diffs if d > 0)
    n_down = sum(1 for d in diffs if d < 0)
    n_flat = sum(1 for d in diffs if d == 0)
    dominant = max(n_up, n_down) + n_flat

    return dominant / len(diffs)


def _cycling_regularity(session: Session) -> tuple:
    """Compute duration CV and gap CV for a session's events.

    Real compressor cycling has consistent cycle durations and regular gaps
    between cycles.  Random device activity has irregular durations and gaps.

    Returns:
        (duration_cv, gap_cv) — both 0.0–∞.  Lower = more regular.
        Returns (0, 0) if too few events to compute.
    """
    if not session.events or len(session.events) < 3:
        return 0.0, 0.0

    sorted_events = sorted(session.events, key=lambda e: e.get('on_start', ''))

    # Duration CV (excluding first event — the initial long run)
    durations = [
        e.get('duration', 0) or 0
        for e in sorted_events[1:]  # skip initial activation
        if (e.get('duration') or 0) > 0
    ]
    if len(durations) >= 2 and np.mean(durations) > 0:
        duration_cv = float(np.std(durations) / np.mean(durations))
    else:
        duration_cv = 0.0

    # Gap CV — time between consecutive events
    gaps = []
    for i in range(len(sorted_events) - 1):
        curr_end = sorted_events[i].get('off_end') or sorted_events[i].get('off_start') or sorted_events[i].get('on_start')
        next_start = sorted_events[i + 1].get('on_start')
        if curr_end and next_start:
            gap = (pd.Timestamp(next_start) - pd.Timestamp(curr_end)).total_seconds() / 60
            if gap > 0:
                gaps.append(gap)

    if len(gaps) >= 2 and np.mean(gaps) > 0:
        gap_cv = float(np.std(gaps) / np.mean(gaps))
    else:
        gap_cv = 0.0

    return duration_cv, gap_cv


def _session_ac_likeness(session: Session) -> float:
    """Score how AC-like a single-phase session is (0.0–1.0).

    Used as a component of central AC confidence.
    Considers: cycle count, magnitude, cycle duration range, magnitude
    consistency, and monotonicity of magnitude trend.
    """
    scores = []

    # Cycle count: 4 → 0.5, ≥10 → 1.0
    scores.append(_linear_score(session.cycle_count, CENTRAL_AC_MIN_CYCLES, 10))

    # Magnitude: 800W → 0.5, ≥1500W → 1.0
    scores.append(_linear_score(session.avg_magnitude, CENTRAL_AC_MIN_MAGNITUDE, 1500))

    # Magnitude CV: lower is better. CV=0.30 → 0.5, CV=0 → 1.0
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


def _central_ac_confidence(mp: MultiPhaseSession) -> tuple:
    """Compute confidence score for a central AC session.

    Factors:
    - phase_count: 2 phases → 0.7, 3 phases → 1.0
    - sync_quality: how tightly the phases overlap in time
    - magnitude_balance: how similar the magnitudes are across phases
    - ac_likeness: how AC-like the constituent sessions are
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

    # AC likeness — how AC-like each constituent session is
    likeness_scores = [_session_ac_likeness(ps) for ps in mp.phase_sessions.values()]
    breakdown['ac_likeness'] = round(float(np.mean(likeness_scores)), 2) if likeness_scores else 0.5

    confidence = round(float(np.mean(list(breakdown.values()))), 2)
    return confidence, breakdown


def _regular_ac_confidence(session: Session) -> tuple:
    """Compute confidence score for a regular AC session.

    Factors:
    - cycle_count: 4 cycles → 0.5, ≥10 cycles → 1.0
    - magnitude_cv: CV=0.20 → 0.5, CV≤0.05 → 1.0
    - initial_duration: anchor event length (≥15min → 0.5, ≥30min → 1.0)
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

    # Initial duration — anchor event (first long activation, not necessarily first event)
    anchor_dur = _find_anchor_duration(session)
    breakdown['initial_duration'] = round(_linear_score(
        anchor_dur, AC_MIN_INITIAL_DURATION, AC_MIN_INITIAL_DURATION * 2
    ), 2)

    # Magnitude — higher magnitude = clearer signal
    breakdown['magnitude'] = round(_linear_score(
        session.avg_magnitude, AC_MIN_MAGNITUDE, AC_MIN_MAGNITUDE * 2
    ), 2)

    confidence = round(float(np.mean(list(breakdown.values()))), 2)
    return confidence, breakdown


def _find_anchor_duration(session: Session) -> float:
    """Return the duration of the first event with duration ≥ AC_MIN_INITIAL_DURATION.

    Falls back to the longest event duration if no event meets the threshold.
    """
    if not session.events:
        return 0
    sorted_events = sorted(session.events, key=lambda e: e['on_start'])
    for e in sorted_events:
        dur = e.get('duration', 0) or 0
        if dur >= AC_MIN_INITIAL_DURATION:
            return dur
    # No anchor found — return longest event
    return max((e.get('duration', 0) or 0) for e in sorted_events)


def _unknown_confidence(
    session: Session,
    all_matches: pd.DataFrame,
) -> tuple:
    """Compute two exclusion confidences for an unknown session.

    For each known device type, scores how confident we are that the session
    is **not** that type.  Low ``not_X`` → the session looks like X, might
    be misclassified.  High ``not_X`` → definitely not X.

    Returns:
        (confidence, breakdown) where:
        - confidence = min(not_boiler, not_ac) — how confidently *unknown*
          (low = borderline, close to some pattern; high = truly unknown).
        - breakdown contains ``not_boiler``, ``not_ac`` and per-criterion
          detail scores.
    """
    breakdown = {}

    # --- Boiler proximity (how boiler-like is it, 0–1) ---
    b_dur = round(_linear_score(
        session.avg_cycle_duration, 0, BOILER_MIN_DURATION
    ), 2)
    b_mag = round(_linear_score(
        session.avg_magnitude, 0, BOILER_MIN_MAGNITUDE
    ), 2)
    if session.cycle_count <= 2:
        b_cycles = 1.0
    elif session.cycle_count <= 4:
        b_cycles = 0.5
    else:
        b_cycles = max(0.0, 1.0 - session.cycle_count * 0.1)

    min_dist = _min_distance_to_medium_events(session, all_matches)
    if min_dist is None:
        b_isolation = 1.0
    else:
        b_isolation = round(_linear_score(
            min_dist, 0, BOILER_ISOLATION_WINDOW
        ), 2)

    boiler_proximity = float(np.mean([b_dur, b_mag, b_cycles, b_isolation]))
    not_boiler = round(1.0 - boiler_proximity, 2)

    breakdown['not_boiler'] = not_boiler
    breakdown['boiler_dur'] = b_dur
    breakdown['boiler_mag'] = b_mag
    breakdown['boiler_cycles'] = round(b_cycles, 2)
    breakdown['boiler_isolation'] = b_isolation

    # --- Regular AC proximity (how AC-like is it, 0–1) ---
    a_cycles = round(_linear_score(
        session.cycle_count, 1, 1 + AC_MIN_FOLLOWING_CYCLES
    ), 2)
    a_mag = round(_linear_score(
        session.avg_magnitude, 0, AC_MIN_MAGNITUDE
    ), 2)
    anchor_dur = _find_anchor_duration(session)
    a_initial = round(_linear_score(
        anchor_dur, 0, AC_MIN_INITIAL_DURATION
    ), 2)
    if session.magnitude_cv is not None and session.magnitude_cv >= 0:
        a_cv = round(_inverse_linear_score(
            session.magnitude_cv, 0, AC_MAX_MAGNITUDE_CV * 2
        ), 2)
    else:
        a_cv = 0.5

    ac_proximity = float(np.mean([a_cycles, a_mag, a_initial, a_cv]))
    not_ac = round(1.0 - ac_proximity, 2)

    breakdown['not_ac'] = not_ac
    breakdown['ac_cycles'] = a_cycles
    breakdown['ac_mag'] = a_mag
    breakdown['ac_initial_dur'] = a_initial
    breakdown['ac_cv'] = a_cv

    # Overall: how confidently "truly unknown" (low = borderline)
    confidence = round(min(not_boiler, not_ac), 2)
    return confidence, breakdown


def _unknown_reason(session: Session, breakdown: dict) -> str:
    """Generate descriptive reason for unknown classification.

    Shows both exclusion confidences and which criteria failed for the
    nearest device type.
    """
    not_boiler = breakdown.get('not_boiler', 1.0)
    not_ac = breakdown.get('not_ac', 1.0)

    parts = []

    # Boiler side
    boiler_fails = []
    if session.avg_cycle_duration < BOILER_MIN_DURATION:
        boiler_fails.append(f"dur={session.avg_cycle_duration:.0f}/{BOILER_MIN_DURATION}min")
    if session.avg_magnitude < BOILER_MIN_MAGNITUDE:
        boiler_fails.append(f"mag={session.avg_magnitude:.0f}/{BOILER_MIN_MAGNITUDE}W")
    if session.cycle_count > 2:
        boiler_fails.append(f"cycles={session.cycle_count} (max 2)")
    boiler_str = f"not_boiler: {not_boiler:.0%}"
    if boiler_fails:
        boiler_str += f" ({', '.join(boiler_fails)})"
    parts.append(boiler_str)

    # AC side
    ac_fails = []
    if session.cycle_count < (1 + AC_MIN_FOLLOWING_CYCLES):
        ac_fails.append(f"cycles={session.cycle_count} (min {1 + AC_MIN_FOLLOWING_CYCLES})")
    if session.avg_magnitude < AC_MIN_MAGNITUDE:
        ac_fails.append(f"mag={session.avg_magnitude:.0f}/{AC_MIN_MAGNITUDE}W")
    anchor_dur = _find_anchor_duration(session)
    if anchor_dur < AC_MIN_INITIAL_DURATION:
        ac_fails.append(f"init_dur={anchor_dur:.0f}/{AC_MIN_INITIAL_DURATION}min")
    if session.magnitude_cv > AC_MAX_MAGNITUDE_CV:
        ac_fails.append(f"CV={session.magnitude_cv:.2f} (max {AC_MAX_MAGNITUDE_CV})")
    ac_str = f"not_ac: {not_ac:.0%}"
    if ac_fails:
        ac_str += f" ({', '.join(ac_fails)})"
    parts.append(ac_str)

    return ' | '.join(parts)


# ============================================================================
# Regular AC detection
# ============================================================================

def _is_regular_ac_session(session: Session) -> bool:
    """Check if a session qualifies as a regular AC compressor pattern.

    Criteria:
    - Average magnitude ≥ 800 W
    - 4+ cycles (1 initial ≥ 15 min + 3 following)
    - Magnitude CV ≤ 20% (overall or dominant iteration)

    Because the pipeline detects events iteratively at different thresholds
    (2000, 1500, 1100, 800W), the same compressor appears with different
    magnitudes across iterations, inflating the overall CV.  When the overall
    CV exceeds the limit, we fall back to checking the dominant iteration
    (most events) — if it alone shows AC-like behaviour, that's sufficient.
    """
    if session.avg_magnitude < AC_MIN_MAGNITUDE:
        return False

    if session.cycle_count < (1 + AC_MIN_FOLLOWING_CYCLES):
        return False

    # Find the first long activation (≥ AC_MIN_INITIAL_DURATION) — the AC
    # startup.  Earlier short events may belong to a different device that
    # happened to be grouped into the same session.
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
        # Anchor found — evaluate from anchor onward
        following_events = sorted_events[anchor_idx:]
        if len(following_events) < (1 + AC_MIN_FOLLOWING_CYCLES):
            return False

        mags = [abs(float(e.get('on_magnitude', 0))) for e in following_events]
        if not mags or np.mean(mags) <= 0:
            return False
        anchor_cv = float(np.std(mags) / np.mean(mags))

        if anchor_cv <= AC_MAX_MAGNITUDE_CV:
            return True

    # No anchor event, or anchor CV too high — fall back to checking
    # individual iterations.  AC sessions sometimes lack a single long
    # initial run (e.g. the compressor was already running when detection
    # started, or the startup wasn't captured).  A single iteration with
    # consistent cycling is sufficient evidence.
    return _has_ac_pattern_in_dominant_iteration(session)


def _has_ac_pattern_in_dominant_iteration(session: Session) -> bool:
    """Check if any single iteration shows AC-like cycling.

    Groups events by iteration and checks each one with enough events.
    Accepts if ANY iteration meets regular AC criteria independently.

    Because different devices can contribute events to the same iteration
    (e.g., an outlier from a different appliance), checking only the
    dominant iteration may fail.  Checking all iterations catches cases
    where a smaller but cleaner iteration proves the AC pattern.
    """
    if not session.events:
        return False

    # Group events by iteration
    by_iter: Dict[int, list] = {}
    for e in session.events:
        it = e.get('iteration')
        if it is not None:
            by_iter.setdefault(int(it), []).append(e)

    if not by_iter:
        return False

    # Check each iteration — accept if any one passes
    min_events = 1 + AC_MIN_FOLLOWING_CYCLES
    for it_events in by_iter.values():
        if len(it_events) < min_events:
            continue

        magnitudes = [abs(float(e.get('on_magnitude', 0))) for e in it_events]
        if not magnitudes or np.mean(magnitudes) <= 0:
            continue

        cv = float(np.std(magnitudes) / np.mean(magnitudes))
        if cv > AC_MAX_MAGNITUDE_CV:
            continue

        if np.mean(magnitudes) < AC_MIN_MAGNITUDE:
            continue

        return True

    return False


def _is_central_ac_candidate(session: Session) -> bool:
    """Check if a session shows AC-like compressor cycling.

    Criteria (all must pass):
    - Magnitude ≥ 800W
    - ≥ 4 cycles
    - Majority of events in AC duration range (3-30 min)
    - Magnitude CV ≤ 30%
    - Duration CV ≤ 40% (cycle durations should be consistent)
    - Gap CV ≤ 50% (gaps between cycles should be regular)
    - Magnitude trend is roughly monotonic (not zig-zagging)

    A session that passes this check exhibits compressor-like cycling, which is
    a prerequisite for being part of a multi-phase central AC group.
    """
    if session.avg_magnitude < CENTRAL_AC_MIN_MAGNITUDE:
        return False

    if session.cycle_count < CENTRAL_AC_MIN_CYCLES:
        return False

    # Majority of events must have duration in AC range (3-30 min)
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
            if in_range < len(durations) * 0.5:
                return False

    if session.magnitude_cv > CENTRAL_AC_MAX_MAGNITUDE_CV:
        return False

    # Cycling regularity — durations and gaps must be consistent
    dur_cv, gap_cv = _cycling_regularity(session)
    if dur_cv > CENTRAL_AC_MAX_DURATION_CV:
        return False
    if gap_cv > CENTRAL_AC_MAX_GAP_CV:
        return False

    # Magnitude trend must be roughly monotonic (not random zig-zag)
    mono = _magnitude_monotonicity(session)
    if mono < 0.6:
        return False

    return True
