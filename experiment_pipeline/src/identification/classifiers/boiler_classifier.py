"""
Boiler device classifier.

Identifies boiler events at the individual event level (no pre-grouping).
Includes three-phase device detection and phase exclusivity enforcement.
"""
import logging
import uuid
from collections import Counter
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ..config import (
    BOILER_MIN_DURATION,
    BOILER_MIN_MAGNITUDE,
    BOILER_ISOLATION_WINDOW,
    AC_MIN_CYCLE_DURATION,
    AC_MAX_CYCLE_DURATION,
    AC_MIN_MAGNITUDE,
    AC_FILTER_WINDOW,
    AC_FILTER_MIN_CYCLES,
    AC_FILTER_MIN_CYCLE_MAG,
    AC_FILTER_MAG_RATIO,
    PHASES,
    THREE_PHASE_OVERLAP_TOLERANCE,
    THREE_PHASE_MAX_DURATION_RATIO,
    THREE_PHASE_MIN_PHASES,
    THREE_PHASE_MIN_OVERLAP_RATIO,
)
from ..session_grouper import Session, MultiPhaseSession, build_single_event_session
from .scoring_utils import (
    _linear_score,
    _inverse_linear_score,
    _min_distance_to_medium_events,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Boiler phase exclusivity
# ============================================================================

def _enforce_boiler_phase_exclusivity(
    boiler_sessions: list,
) -> tuple:
    """A household has ONE boiler connected to ONE phase.

    If boiler sessions appear on multiple phases, keep only those on the
    dominant phase (most sessions).  Sessions on other phases are returned
    as ``demoted`` so the caller can reclassify them as unknown.

    Returns:
        (kept, demoted) -- two lists of ClassifiedSession.
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
) -> tuple:
    """Identify boiler events at the individual event level (no pre-grouping).

    Algorithm:
        1. Find all events with duration >= 15min AND magnitude >= 1500W AND isolated
        2. For each candidate, check other phases for simultaneous long events:
           - If 2+ other phases have overlapping long events with similar duration
             -> three_phase_device
        3. Apply phase exclusivity to remaining boiler candidates
        4. Remove all classified events from the pool

    Returns:
        (boiler_classified, three_phase_classified, remaining_matches)
    """
    # Import here to avoid circular import
    from ..session_classifier import ClassifiedSession

    consumed_indices: Set[int] = set()

    # --- Pass 1: find all boiler SIZE candidates (duration + magnitude only) ---
    size_candidate_indices: Set[int] = set()
    for idx, row in all_matches.iterrows():
        duration = row.get('duration', 0) or 0
        magnitude = abs(row.get('on_magnitude', 0) or 0)
        if duration >= BOILER_MIN_DURATION and magnitude >= BOILER_MIN_MAGNITUDE:
            size_candidate_indices.add(idx)

    # --- Pass 2: check isolation, excluding other size candidates from pool ---
    boiler_candidates: List[Tuple[int, pd.Series]] = []
    for idx in size_candidate_indices:
        row = all_matches.loc[idx]
        peer_indices = size_candidate_indices - {idx}
        pool = all_matches.loc[~all_matches.index.isin(peer_indices)]
        if not _is_boiler_candidate_event(row, pool):
            continue
        boiler_candidates.append((idx, row))

    # --- Three-phase check: similar long events on other phases? ---
    three_phase_classified: list = []
    non_three_phase: List[Tuple[int, pd.Series]] = []

    # Track which candidates were consumed by three-phase grouping
    three_phase_consumed: Set[int] = set()

    for idx, row in boiler_candidates:
        if idx in three_phase_consumed:
            continue

        other_phase_matches = _find_simultaneous_long_events(row, all_matches)

        if len(other_phase_matches) >= THREE_PHASE_MIN_PHASES:
            # This is a three-phase device (charger?)
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
    boiler_classified_pre: list = []
    for idx, row in non_three_phase:
        if idx in three_phase_consumed:
            continue
        session = build_single_event_session(row.to_dict(), row['phase'])
        conf, breakdown = _boiler_confidence(session, all_matches)
        boiler_classified_pre.append(ClassifiedSession(
            session=session,
            device_type='boiler',
            reason=f"Single event >={BOILER_MIN_DURATION}min, >={BOILER_MIN_MAGNITUDE}W, isolated",
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
        - duration >= BOILER_MIN_DURATION (15 min)
        - abs(on_magnitude) >= BOILER_MIN_MAGNITUDE (1500W)
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
        - Time overlap with +/-10% of event duration tolerance
        - Similar duration (max ratio 2:1)
        - Duration >= BOILER_MIN_DURATION
        - Magnitude doesn't matter -- only pattern/character matters
    """
    ref_phase = reference['phase']
    ref_start = pd.Timestamp(reference['on_start'])
    ref_end = pd.Timestamp(reference.get('off_end') or reference.get('off_start') or reference['on_start'])
    ref_duration = reference.get('duration', 0) or 0

    if ref_duration <= 0:
        return []

    # Tolerance: +/-10% of event duration
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
                # Compute actual overlap ratio (without tolerance)
                overlap_start = max(ref_start, other_start)
                overlap_end = min(ref_end, other_end)
                actual_overlap = max(0, (overlap_end - overlap_start).total_seconds() / 60)
                shorter_duration = min(ref_duration, other_duration)
                overlap_ratio = actual_overlap / shorter_duration if shorter_duration > 0 else 0

                if overlap_ratio < THREE_PHASE_MIN_OVERLAP_RATIO:
                    continue

                results.append((other_idx, other_row))
                break  # one match per phase is enough

    return results


def _three_phase_confidence(mp: MultiPhaseSession) -> Tuple[float, dict]:
    """Compute confidence for three_phase_device classification.

    Factors:
        - phase_count: 2 phases -> lower confidence, 3 phases -> can reach 100%
        - magnitude_similarity: CV of magnitudes across phases (<=10% -> perfect)
        - timing_sync: how close start times are
        - duration_similarity: how similar durations are across phases

    100% confidence requires: 3 phases + magnitudes within 10% CV.
    """
    breakdown = {}

    n_phases = len(mp.phases)

    # Magnitude similarity across phases (CV)
    magnitudes = [ps.avg_magnitude for ps in mp.phase_sessions.values()]
    if len(magnitudes) >= 2 and np.mean(magnitudes) > 0:
        mag_cv = float(np.std(magnitudes) / np.mean(magnitudes))
        breakdown['magnitude_similarity'] = round(_inverse_linear_score(mag_cv, 0, 0.3), 2)
    else:
        breakdown['magnitude_similarity'] = 0.5

    # Phase count: 2 phases caps at 0.7, 3 phases at 1.0
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

    # Perfect score: 3 phases + magnitude CV <= 10%
    if n_phases >= 3 and len(magnitudes) >= 3:
        mag_cv = float(np.std(magnitudes) / np.mean(magnitudes))
        if mag_cv <= 0.10:
            confidence = 1.0

    return confidence, breakdown


def _has_nearby_compressor_cycles(
    event: dict,
    all_matches: pd.DataFrame,
    phase: str,
    boiler_magnitude: float,
) -> bool:
    """Check if compressor cycles exist near a boiler candidate event.

    Two filters prevent false positives:
    1. Events that ended well before the boiler started are from separate
       devices and should not count.
    2. Events temporally contained within the boiler's time range are likely
       iterative-detection artifacts and should not count.
    """
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

    if nearby.empty:
        return False

    # --- Filter 1: exclude events that ended before the boiler started ---
    pre_boiler_margin = pd.Timedelta(minutes=5)
    nearby_off_end = nearby['off_end'].combine_first(
        nearby['off_start'].combine_first(nearby['on_start'])
    )
    nearby = nearby[nearby_off_end >= (event_start - pre_boiler_margin)]

    if nearby.empty:
        return False

    # --- Filter 2: exclude events contained within the boiler's time range ---
    containment_margin = pd.Timedelta(minutes=2)
    contained = (
        (nearby['on_start'] >= event_start) &
        (nearby_off_end.loc[nearby.index] <= event_end + containment_margin)
    )
    nearby = nearby[~contained]

    if nearby.empty:
        return False

    cycles = nearby[
        (nearby['duration'] >= AC_MIN_CYCLE_DURATION) &
        (nearby['duration'] <= AC_MAX_CYCLE_DURATION) &
        (nearby['on_magnitude'].abs() >= AC_FILTER_MIN_CYCLE_MAG) &
        (nearby['on_magnitude'].abs() >= boiler_magnitude * AC_FILTER_MAG_RATIO)
    ]

    return len(cycles) >= AC_FILTER_MIN_CYCLES


def _boiler_confidence(
    session: Session,
    all_matches: pd.DataFrame,
) -> tuple:
    """Compute confidence score for a boiler session.

    Factors:
    - duration: 25min -> 0.5, >=50min -> 1.0
    - magnitude: 1500W -> 0.5, >=3000W -> 1.0
    - isolation: distance to nearest medium event (farther = higher)
    - cycle_count: 1 event -> 1.0, 2 events -> 0.6
    """
    breakdown = {}

    # Duration -- how far above minimum
    breakdown['duration'] = round(_linear_score(
        session.avg_cycle_duration, BOILER_MIN_DURATION, BOILER_MIN_DURATION * 2
    ), 2)

    # Magnitude -- how far above minimum
    breakdown['magnitude'] = round(_linear_score(
        session.avg_magnitude, BOILER_MIN_MAGNITUDE, BOILER_MIN_MAGNITUDE * 2
    ), 2)

    # Cycle count -- single event is most boiler-like
    breakdown['cycle_count'] = 1.0 if session.cycle_count == 1 else 0.6

    # Isolation -- minimum distance to nearest medium-duration event
    min_dist = _min_distance_to_medium_events(session, all_matches)
    if min_dist is None:
        breakdown['isolation'] = 1.0  # no medium events at all -- perfect isolation
    else:
        # 0 min distance -> 0.5, >=BOILER_ISOLATION_WINDOW -> 1.0
        breakdown['isolation'] = round(_linear_score(
            min_dist, 0, BOILER_ISOLATION_WINDOW
        ), 2)

    confidence = round(float(np.mean(list(breakdown.values()))), 2)
    return confidence, breakdown
