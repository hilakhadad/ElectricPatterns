"""
Session builder helpers.

Functions for constructing Session objects from event rows or dicts,
and for splitting sessions at initial-run boundaries.
"""
import logging
import uuid
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import PHASES
from .session_grouper import Session

logger = logging.getLogger(__name__)


def _build_session(rows, phase: str) -> Session:
    """Create a Session from a list of named-tuple rows."""
    events = [row._asdict() for row in rows]

    starts = [r.on_start for r in rows]
    ends = [getattr(r, 'off_end', None) or getattr(r, 'off_start', None) or r.on_start for r in rows]
    magnitudes = [abs(r.on_magnitude) for r in rows]
    durations = [r.duration for r in rows if hasattr(r, 'duration') and r.duration is not None]

    session_start = min(starts)
    session_end = max(ends)
    total_dur = (session_end - session_start).total_seconds() / 60

    mean_mag = float(np.mean(magnitudes)) if magnitudes else 0
    std_mag = float(np.std(magnitudes)) if len(magnitudes) > 1 else 0
    cv = std_mag / mean_mag if mean_mag > 0 else 0

    thresholds = sorted(set(
        int(r.threshold) for r in rows if hasattr(r, 'threshold')
    ))

    return Session(
        session_id=f"s_{uuid.uuid4().hex[:8]}",
        phase=phase,
        events=events,
        start=session_start,
        end=session_end,
        cycle_count=len(rows),
        total_duration_minutes=round(total_dur, 1),
        avg_magnitude=round(mean_mag, 1),
        max_magnitude=round(float(max(magnitudes)), 1) if magnitudes else 0,
        magnitude_cv=round(cv, 3),
        avg_cycle_duration=round(float(np.mean(durations)), 1) if durations else 0,
        thresholds=thresholds,
    )


def _build_session_from_dicts(event_dicts: list, phase: str) -> Session:
    """Create a Session from a list of event dicts (used after splitting)."""
    starts = [pd.Timestamp(e['on_start']) for e in event_dicts]
    ends = [
        pd.Timestamp(e.get('off_end') or e.get('off_start') or e['on_start'])
        for e in event_dicts
    ]
    magnitudes = [abs(float(e.get('on_magnitude', 0))) for e in event_dicts]
    durations = [
        e.get('duration', 0) or 0
        for e in event_dicts
        if e.get('duration') is not None
    ]

    session_start = min(starts)
    session_end = max(ends)
    total_dur = (session_end - session_start).total_seconds() / 60

    mean_mag = float(np.mean(magnitudes)) if magnitudes else 0
    std_mag = float(np.std(magnitudes)) if len(magnitudes) > 1 else 0
    cv = std_mag / mean_mag if mean_mag > 0 else 0

    thresholds = sorted(set(
        int(e['threshold']) for e in event_dicts if 'threshold' in e
    ))

    return Session(
        session_id=f"s_{uuid.uuid4().hex[:8]}",
        phase=phase,
        events=event_dicts,
        start=session_start,
        end=session_end,
        cycle_count=len(event_dicts),
        total_duration_minutes=round(total_dur, 1),
        avg_magnitude=round(mean_mag, 1),
        max_magnitude=round(float(max(magnitudes)), 1) if magnitudes else 0,
        magnitude_cv=round(cv, 3),
        avg_cycle_duration=round(float(np.mean(durations)), 1) if durations else 0,
        thresholds=thresholds,
    )


def build_single_event_session(event_dict: dict, phase: str) -> Session:
    """Create a Session from a single event (used for boiler / three_phase_device).

    Boilers are always single-event sessions -- no grouping needed.
    """
    return _build_session_from_dicts([event_dict], phase)


def _split_sessions_at_initial_run(
    sessions: List[Session],
    long_ratio: float = 3.0,
    min_long_duration: float = 10.0,
) -> List[Session]:
    """Split sessions where a long event appears after short prefix events.

    In AC sessions, the initial long compressor run should be the first event.
    If short events (from a different device) precede it within the session gap,
    they get grouped into the same session by accident.

    This detects the pattern [short, short, ..., LONG, cycling...] and splits
    the session so the LONG event becomes the first event of a new session.

    Args:
        sessions: Sessions to process.
        long_ratio: An event is "long" if duration >= this x median of preceding events.
        min_long_duration: Minimum duration (minutes) for an event to trigger a split.
    """
    result: List[Session] = []

    for session in sessions:
        if len(session.events) < 3:
            result.append(session)
            continue

        sorted_events = sorted(session.events, key=lambda e: e.get('on_start', ''))
        durations = [e.get('duration', 0) or 0 for e in sorted_events]

        # Find the first event that's significantly longer than all preceding events
        split_idx = None
        for i in range(1, min(len(sorted_events), 4)):  # only check first few events
            if durations[i] < min_long_duration:
                continue
            preceding = durations[:i]
            if not preceding:
                continue
            median_preceding = float(np.median(preceding))
            if median_preceding > 0 and durations[i] >= median_preceding * long_ratio:
                split_idx = i
                break

        if split_idx is None:
            result.append(session)
            continue

        # Split: prefix events become one session, long event + rest become another
        prefix_rows = sorted_events[:split_idx]
        main_rows = sorted_events[split_idx:]

        prefix_session = _build_session_from_dicts(prefix_rows, session.phase)
        main_session = _build_session_from_dicts(main_rows, session.phase)

        result.append(prefix_session)
        result.append(main_session)

        logger.debug(
            f"Split session on {session.phase}: "
            f"{len(prefix_rows)} prefix events + "
            f"{len(main_rows)} main events (initial run at idx {split_idx})"
        )

    return result
