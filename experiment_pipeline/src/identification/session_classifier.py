"""
Event-level device classifier (classify-first, group-second).

Priority-based classification that identifies devices BEFORE grouping:
  1. **Boiler** — single long high-power isolated event (event-level)
     Multi-phase check: similar events on other phases → three_phase_device
  2. **AC** — cycling patterns detected from remaining events
     Cross-phase check: overlapping AC patterns → central_ac vs regular_ac
  3. **Unknown** — everything else (grouped by time proximity)
"""
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

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
    DEFAULT_SESSION_GAP_MINUTES,
    MULTI_PHASE_WINDOW,
    PHASES,
    THREE_PHASE_OVERLAP_TOLERANCE,
    THREE_PHASE_MAX_DURATION_RATIO,
    THREE_PHASE_MIN_PHASES,
)
from .session_grouper import Session, MultiPhaseSession, build_single_event_session

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

def classify_events(
    filtered_matches: pd.DataFrame,
) -> Dict[str, List[ClassifiedSession]]:
    """Classify events using priority-based approach (classify-first, group-second).

    Pipeline:
        Step 1: Boiler identification (event-level, before any grouping)
                Multi-phase check → three_phase_device
                Phase exclusivity for remaining boilers
        Step 2: AC identification (cycling pattern detection from remaining events)
                Cross-phase overlap → central_ac vs regular_ac
        Step 3: Unknown (group remaining by time proximity)

    Args:
        filtered_matches: DataFrame from filter_transient_events() with columns:
            on_start, off_end, on_magnitude, duration, phase, tag, iteration, threshold

    Returns:
        Dict mapping device_type → list of ClassifiedSession.
        Keys: 'boiler', 'three_phase_device', 'central_ac', 'regular_ac', 'unknown'
    """
    from .session_grouper import group_into_sessions

    result: Dict[str, List[ClassifiedSession]] = {
        'boiler': [],
        'three_phase_device': [],
        'central_ac': [],
        'regular_ac': [],
        'unknown': [],
    }

    if filtered_matches.empty:
        return result

    # Ensure timestamps
    for col in ['on_start', 'off_end', 'off_start']:
        if col in filtered_matches.columns and not pd.api.types.is_datetime64_any_dtype(filtered_matches[col]):
            filtered_matches[col] = pd.to_datetime(filtered_matches[col])

    # --- Step 1: Boiler (event-level) -------------------------------------
    boiler_list, three_phase_list, remaining = _identify_boiler_events(filtered_matches)
    result['boiler'] = boiler_list
    result['three_phase_device'] = three_phase_list

    n_boiler = len(boiler_list)
    n_3phase = len(three_phase_list)
    if n_boiler or n_3phase:
        logger.info(
            f"  Step 1: {n_boiler} boiler + {n_3phase} three_phase_device "
            f"({len(filtered_matches) - len(remaining)}/{len(filtered_matches)} events consumed)"
        )

    # --- Step 2: AC (pattern-based grouping) ------------------------------
    central_list, regular_list, remaining = _identify_ac_sessions(remaining)
    result['central_ac'] = central_list
    result['regular_ac'] = regular_list

    n_central = len(central_list)
    n_regular = len(regular_list)
    if n_central or n_regular:
        logger.info(f"  Step 2: {n_central} central_ac + {n_regular} regular_ac")

    # --- Step 3: Unknown (group remaining by time proximity) --------------
    if not remaining.empty:
        unknown_sessions = group_into_sessions(remaining)
        for s in unknown_sessions:
            conf, breakdown = _unknown_confidence(s, filtered_matches)
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
        f"Classification: {n_boiler} boiler, {n_3phase} three_phase, "
        f"{n_central} central_ac, {n_regular} regular_ac, {n_unknown} unknown"
    )

    return result


def classify_sessions(
    sessions: List[Session],
    all_matches: pd.DataFrame,
) -> Dict[str, List[ClassifiedSession]]:
    """Backward-compatible wrapper — delegates to classify_events().

    .. deprecated:: Use :func:`classify_events` directly.
    """
    logger.info("classify_sessions() delegating to classify_events()")
    return classify_events(all_matches)


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
# Step 1: Boiler identification (event-level)
# ============================================================================

def _identify_boiler_events(
    all_matches: pd.DataFrame,
) -> Tuple[List[ClassifiedSession], List[ClassifiedSession], pd.DataFrame]:
    """Identify boiler events at the individual event level (no pre-grouping).

    Algorithm:
        1. Find all events with duration ≥ 15min AND magnitude ≥ 1500W AND isolated
        2. For each candidate, check other phases for simultaneous long events:
           - If 2+ other phases have overlapping long events with similar duration
             → three_phase_device
        3. Apply phase exclusivity to remaining boiler candidates
        4. Remove all classified events from the pool

    Returns:
        (boiler_classified, three_phase_classified, remaining_matches)
    """
    boiler_candidates: List[Tuple[int, pd.Series]] = []  # (index, row)
    consumed_indices: Set[int] = set()

    for idx, row in all_matches.iterrows():
        if not _is_boiler_candidate_event(row, all_matches):
            continue
        boiler_candidates.append((idx, row))

    # --- Three-phase check: similar long events on other phases? ---
    three_phase_classified: List[ClassifiedSession] = []
    non_three_phase: List[Tuple[int, pd.Series]] = []

    # Track which candidates were consumed by three-phase grouping
    three_phase_consumed: Set[int] = set()

    for idx, row in boiler_candidates:
        if idx in three_phase_consumed:
            continue

        other_phase_matches = _find_simultaneous_long_events(row, all_matches)

        if len(other_phase_matches) >= THREE_PHASE_MIN_PHASES:
            # This is a three-phase device (charger?)
            # Collect all events involved (reference + other phases)
            all_phase_events = {row['phase']: row.to_dict()}
            for other_idx, other_row in other_phase_matches:
                all_phase_events[other_row['phase']] = other_row.to_dict()

            # Build MultiPhaseSession
            phase_sessions = {}
            consumed_in_this_group = {idx}
            for phase, ev_dict in all_phase_events.items():
                session = build_single_event_session(ev_dict, phase)
                phase_sessions[phase] = session

            # Find and consume other boiler candidates that are part of this group
            for other_idx, other_row in other_phase_matches:
                consumed_in_this_group.add(other_idx)
                # Also check if this other event was a boiler candidate
                for bc_idx, bc_row in boiler_candidates:
                    if bc_idx == other_idx:
                        three_phase_consumed.add(bc_idx)

            mp = MultiPhaseSession(
                session_id=f"mp_{uuid.uuid4().hex[:8]}",
                phases=sorted(all_phase_events.keys()),
                phase_sessions=phase_sessions,
                start=min(ps.start for ps in phase_sessions.values()),
                end=max(ps.end for ps in phase_sessions.values()),
                total_magnitude=sum(ps.avg_magnitude for ps in phase_sessions.values()),
            )

            conf, breakdown = _three_phase_confidence(mp)
            three_phase_classified.append(ClassifiedSession(
                session=mp,
                device_type='three_phase_device',
                reason=f"Simultaneous long events on {len(mp.phases)} phases",
                confidence=conf,
                confidence_breakdown=breakdown,
            ))

            consumed_indices.update(consumed_in_this_group)
            three_phase_consumed.add(idx)
        else:
            non_three_phase.append((idx, row))

    # --- Phase exclusivity for remaining boiler candidates ---
    boiler_classified_pre: List[ClassifiedSession] = []
    for idx, row in non_three_phase:
        if idx in three_phase_consumed:
            continue
        session = build_single_event_session(row.to_dict(), row['phase'])
        conf, breakdown = _boiler_confidence(session, all_matches)
        boiler_classified_pre.append(ClassifiedSession(
            session=session,
            device_type='boiler',
            reason=f"Single event ≥{BOILER_MIN_DURATION}min, ≥{BOILER_MIN_MAGNITUDE}W, isolated",
            confidence=conf,
            confidence_breakdown=breakdown,
        ))

    boiler_classified, demoted = _enforce_boiler_phase_exclusivity(boiler_classified_pre)

    # Demoted boiler sessions go back to the pool (will become unknown in Step 3)
    boiler_indices = set()
    for cs in boiler_classified:
        for ev in cs.session.events:
            ts = pd.Timestamp(ev['on_start'])
            mask = all_matches['on_start'] == ts
            matching = all_matches.index[mask]
            boiler_indices.update(matching.tolist())

    consumed_indices.update(boiler_indices)

    if demoted:
        logger.info(
            f"  Boiler phase exclusivity: kept {len(boiler_classified)}, "
            f"demoted {len(demoted)} back to pool"
        )

    remaining = all_matches.loc[~all_matches.index.isin(consumed_indices)].reset_index(drop=True)
    return boiler_classified, three_phase_classified, remaining


def _is_boiler_candidate_event(row: pd.Series, all_matches: pd.DataFrame) -> bool:
    """Check if a single event qualifies as a boiler candidate.

    Criteria:
        - duration ≥ BOILER_MIN_DURATION (15 min)
        - abs(on_magnitude) ≥ BOILER_MIN_MAGNITUDE (1500W)
        - No nearby compressor cycles (isolation check)
    """
    duration = row.get('duration', 0) or 0
    if duration < BOILER_MIN_DURATION:
        return False

    magnitude = abs(row.get('on_magnitude', 0) or 0)
    if magnitude < BOILER_MIN_MAGNITUDE:
        return False

    # Isolation: no medium-duration (AC-like) events nearby on same phase
    phase = row['phase']
    phase_matches = all_matches[all_matches['phase'] == phase]
    event_start = pd.Timestamp(row['on_start'])

    medium = phase_matches[
        (phase_matches['duration'] >= AC_MIN_CYCLE_DURATION) &
        (phase_matches['duration'] <= AC_MAX_CYCLE_DURATION) &
        (phase_matches['on_start'] != event_start)
    ]
    if not medium.empty:
        time_diffs = (medium['on_start'] - event_start).abs()
        if hasattr(time_diffs.iloc[0], 'total_seconds'):
            nearby_mask = (time_diffs.dt.total_seconds() / 60) <= BOILER_ISOLATION_WINDOW
        else:
            nearby_mask = time_diffs <= BOILER_ISOLATION_WINDOW
        if nearby_mask.any():
            return False

    # AC filter: check for compressor cycling nearby
    event_dict = row.to_dict()
    if _has_nearby_compressor_cycles(event_dict, all_matches, phase, magnitude):
        return False

    return True


def _find_simultaneous_long_events(
    reference: pd.Series,
    all_matches: pd.DataFrame,
) -> List[Tuple[int, pd.Series]]:
    """Find long events on OTHER phases that overlap with the reference event.

    Overlap criteria (per user requirements):
        - Time overlap with ±10% of event duration tolerance
        - Similar duration (max ratio 2:1)
        - Duration ≥ BOILER_MIN_DURATION
        - Magnitude doesn't matter — only pattern/character matters
    """
    ref_phase = reference['phase']
    ref_start = pd.Timestamp(reference['on_start'])
    ref_end = pd.Timestamp(reference.get('off_end') or reference.get('off_start') or reference['on_start'])
    ref_duration = reference.get('duration', 0) or 0

    if ref_duration <= 0:
        return []

    # Tolerance: ±10% of event duration
    tolerance = pd.Timedelta(minutes=ref_duration * THREE_PHASE_OVERLAP_TOLERANCE)

    results = []
    for other_phase in PHASES:
        if other_phase == ref_phase:
            continue

        phase_df = all_matches[all_matches['phase'] == other_phase]
        if phase_df.empty:
            continue

        for other_idx, other_row in phase_df.iterrows():
            other_duration = other_row.get('duration', 0) or 0
            if other_duration < BOILER_MIN_DURATION:
                continue

            # Duration similarity: max ratio
            if ref_duration > 0 and other_duration > 0:
                ratio = max(ref_duration, other_duration) / min(ref_duration, other_duration)
                if ratio > THREE_PHASE_MAX_DURATION_RATIO:
                    continue

            # Time overlap check with tolerance
            other_start = pd.Timestamp(other_row['on_start'])
            other_end = pd.Timestamp(other_row.get('off_end') or other_row.get('off_start') or other_row['on_start'])

            if (ref_start - tolerance <= other_end) and (other_start - tolerance <= ref_end):
                results.append((other_idx, other_row))
                break  # one match per phase is enough

    return results


def _three_phase_confidence(mp: MultiPhaseSession) -> Tuple[float, dict]:
    """Compute confidence for three_phase_device classification.

    Factors:
        - phase_count: 2 phases → 0.7, 3 phases → 1.0
        - timing_sync: how close start times are
        - duration_similarity: how similar durations are across phases
    """
    breakdown = {}

    n_phases = len(mp.phases)
    breakdown['phase_count'] = 1.0 if n_phases >= 3 else 0.7

    # Timing sync
    starts = [ps.start for ps in mp.phase_sessions.values()]
    if len(starts) >= 2:
        spread_min = (max(starts) - min(starts)).total_seconds() / 60
        breakdown['timing_sync'] = round(_inverse_linear_score(spread_min, 0, 15), 2)
    else:
        breakdown['timing_sync'] = 0.5

    # Duration similarity (CV of durations)
    durations = [ps.total_duration_minutes for ps in mp.phase_sessions.values()]
    if len(durations) >= 2 and np.mean(durations) > 0:
        dur_cv = float(np.std(durations) / np.mean(durations))
        breakdown['duration_similarity'] = round(_inverse_linear_score(dur_cv, 0, 0.5), 2)
    else:
        breakdown['duration_similarity'] = 0.5

    confidence = round(float(np.mean(list(breakdown.values()))), 2)
    return confidence, breakdown


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
# Step 2: AC identification (pattern-based grouping)
# ============================================================================

def _identify_ac_sessions(
    remaining: pd.DataFrame,
) -> Tuple[List[ClassifiedSession], List[ClassifiedSession], pd.DataFrame]:
    """Find AC cycling patterns in remaining events, then check cross-phase overlap.

    Algorithm:
        1. For each phase, find cycling sequences (anchor + following cycles)
        2. Validate each sequence passes AC criteria
        3. Cross-phase check: independently-identified AC sessions overlapping?
           → central_ac vs regular_ac
        4. Remove classified events from pool

    Returns:
        (central_ac_classified, regular_ac_classified, remaining_matches)
    """
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
        - Find anchor events (duration ≥ AC_MIN_INITIAL_DURATION, magnitude ≥ AC_MIN_MAGNITUDE)
        - For each anchor, collect following events:
          * duration 3-30 min
          * gap from previous event < DEFAULT_SESSION_GAP_MINUTES
          * magnitude within reasonable range
        - Require at least AC_MIN_FOLLOWING_CYCLES following events
        - Build Session objects from qualifying sequences

    Returns:
        (sessions, consumed_indices)
    """
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
    from .session_grouper import _build_session_from_dicts
    return _build_session_from_dicts(events, phase)


def _check_central_ac_overlap(
    ac_by_phase: Dict[str, List[Session]],
    all_matches: pd.DataFrame,
) -> Tuple[List[ClassifiedSession], List[ClassifiedSession]]:
    """Check for independently-identified AC patterns that overlap across phases.

    Key difference from old approach: BOTH sides must be independently
    identified as AC patterns — not just any overlapping sessions.

    Returns:
        (central_ac_classified, regular_ac_classified)
    """
    central_list: List[ClassifiedSession] = []
    regular_list: List[ClassifiedSession] = []
    used_session_ids: Set[str] = set()

    phases_with_ac = [p for p in PHASES if p in ac_by_phase]

    if len(phases_with_ac) >= 2:
        tolerance = pd.Timedelta(minutes=CENTRAL_AC_SYNC_TOLERANCE)

        # Check each pair of phases for overlapping AC sessions
        for i, ref_phase in enumerate(phases_with_ac):
            for ref_session in ac_by_phase[ref_phase]:
                if ref_session.session_id in used_session_ids:
                    continue

                synced: Dict[str, Session] = {ref_phase: ref_session}

                for other_phase in phases_with_ac[i + 1:]:
                    for other_session in ac_by_phase[other_phase]:
                        if other_session.session_id in used_session_ids:
                            continue

                        # Check time overlap with tolerance
                        if (ref_session.start - tolerance <= other_session.end and
                                other_session.start - tolerance <= ref_session.end):
                            synced[other_phase] = other_session
                            break

                if len(synced) >= 2:
                    for s in synced.values():
                        used_session_ids.add(s.session_id)

                    mp = MultiPhaseSession(
                        session_id=f"mp_{uuid.uuid4().hex[:8]}",
                        phases=sorted(synced.keys()),
                        phase_sessions=synced,
                        start=min(s.start for s in synced.values()),
                        end=max(s.end for s in synced.values()),
                        total_magnitude=sum(s.avg_magnitude for s in synced.values()),
                    )

                    conf, breakdown = _central_ac_confidence(mp)
                    central_list.append(ClassifiedSession(
                        session=mp,
                        device_type='central_ac',
                        reason=(
                            f"Independent AC cycling on {len(mp.phases)} phases, "
                            f"overlapping in time"
                        ),
                        confidence=conf,
                        confidence_breakdown=breakdown,
                    ))

    # Remaining AC sessions (not part of central AC) → regular AC
    for phase, sessions in ac_by_phase.items():
        for s in sessions:
            if s.session_id not in used_session_ids:
                conf, breakdown = _regular_ac_confidence(s)
                regular_list.append(ClassifiedSession(
                    session=s,
                    device_type='regular_ac',
                    reason=f"{s.cycle_count} cycles, CV={s.magnitude_cv:.2f}",
                    confidence=conf,
                    confidence_breakdown=breakdown,
                ))

    return central_list, regular_list


# ============================================================================
# AC detection helpers (validation)
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
