"""
Central AC classifier.

Detects multi-phase synchronized AC patterns by checking for independently
identified AC sessions that overlap across phases. Also provides iteration-level
AC pattern validation.
"""
import logging
import uuid
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ..config import (
    AC_MIN_MAGNITUDE,
    AC_MIN_FOLLOWING_CYCLES,
    AC_MAX_MAGNITUDE_CV,
    CENTRAL_AC_SYNC_TOLERANCE,
    CENTRAL_AC_MIN_MAGNITUDE,
    CENTRAL_AC_MIN_CYCLES,
    CENTRAL_AC_MIN_CYCLE_DURATION,
    CENTRAL_AC_MAX_CYCLE_DURATION,
    CENTRAL_AC_MAX_MAGNITUDE_CV,
    CENTRAL_AC_MAX_DURATION_CV,
    CENTRAL_AC_MAX_GAP_CV,
    PHASES,
)
from ..session_grouper import Session, MultiPhaseSession
from .scoring_utils import (
    _linear_score,
    _inverse_linear_score,
    _magnitude_monotonicity,
    _cycling_regularity,
)

logger = logging.getLogger(__name__)


def _check_central_ac_overlap(
    ac_by_phase: Dict[str, List[Session]],
    all_matches: pd.DataFrame,
) -> tuple:
    """Check for independently-identified AC patterns that overlap across phases.

    Key difference from old approach: BOTH sides must be independently
    identified as AC patterns -- not just any overlapping sessions.

    Returns:
        (central_ac_classified, regular_ac_classified)
    """
    from ..session_classifier import ClassifiedSession
    from .ac_classifier import _regular_ac_confidence

    central_list: list = []
    regular_list: list = []
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

    # Remaining AC sessions (not part of central AC) -> regular AC
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


def _central_ac_confidence(mp: MultiPhaseSession) -> tuple:
    """Compute confidence score for a central AC session.

    Factors:
    - phase_count: 2 phases -> 0.7, 3 phases -> 1.0
    - sync_quality: how tightly the phases overlap in time
    - magnitude_balance: how similar the magnitudes are across phases
    - ac_likeness: how AC-like the constituent sessions are
    """
    from .ac_classifier import _session_ac_likeness

    breakdown = {}

    # Phase count -- 3 phases is ideal for Israeli 3-phase systems
    n_phases = len(mp.phases)
    breakdown['phase_count'] = 1.0 if n_phases >= 3 else 0.7

    # Sync quality -- how close the start times are across phases
    starts = [ps.start for ps in mp.phase_sessions.values()]
    if len(starts) >= 2:
        start_spread = (max(starts) - min(starts)).total_seconds() / 60
        # 0 min spread -> 1.0, >=CENTRAL_AC_SYNC_TOLERANCE -> 0.5
        breakdown['sync_quality'] = round(_inverse_linear_score(
            start_spread, 0, CENTRAL_AC_SYNC_TOLERANCE
        ), 2)
    else:
        breakdown['sync_quality'] = 0.5

    # Magnitude balance -- CV of magnitudes across phases (lower = more balanced)
    mags = [ps.avg_magnitude for ps in mp.phase_sessions.values()]
    if len(mags) >= 2 and np.mean(mags) > 0:
        mag_cv = float(np.std(mags) / np.mean(mags))
        # CV=0 -> 1.0, CV>=0.5 -> 0.5
        breakdown['magnitude_balance'] = round(_inverse_linear_score(mag_cv, 0, 0.5), 2)
    else:
        breakdown['magnitude_balance'] = 0.5

    # AC likeness -- how AC-like each constituent session is
    likeness_scores = [_session_ac_likeness(ps) for ps in mp.phase_sessions.values()]
    breakdown['ac_likeness'] = round(float(np.mean(likeness_scores)), 2) if likeness_scores else 0.5

    confidence = round(float(np.mean(list(breakdown.values()))), 2)
    return confidence, breakdown


def _has_ac_pattern_in_dominant_iteration(session: Session) -> bool:
    """Check if any single iteration shows AC-like cycling.

    Groups events by iteration and checks each one with enough events.
    Accepts if ANY iteration meets regular AC criteria independently.
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

    # Check each iteration -- accept if any one passes
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
    - Magnitude >= 800W
    - >= 4 cycles
    - Majority of events in AC duration range (3-30 min)
    - Magnitude CV <= 30%
    - Duration CV <= 40% (cycle durations should be consistent)
    - Gap CV <= 50% (gaps between cycles should be regular)
    - Magnitude trend is roughly monotonic (not zig-zagging)
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

    # Cycling regularity -- durations and gaps must be consistent
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
