"""
Scoring utility functions for device classification confidence.

Provides linear and inverse-linear mapping functions used by all
classifier confidence computations.
"""
import numpy as np

from ..config import (
    AC_MIN_CYCLE_DURATION,
    AC_MAX_CYCLE_DURATION,
    BOILER_ISOLATION_WINDOW,
)
from ..session_grouper import Session

import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


def _linear_score(value: float, low: float, high: float) -> float:
    """Map value from [low, high] -> [0.5, 1.0]. Below low -> 0.5, above high -> 1.0."""
    if value >= high:
        return 1.0
    if value <= low:
        return 0.5
    return 0.5 + 0.5 * (value - low) / (high - low)


def _inverse_linear_score(value: float, low: float, high: float) -> float:
    """Map value from [low, high] -> [1.0, 0.5]. Lower is better."""
    if value <= low:
        return 1.0
    if value >= high:
        return 0.5
    return 1.0 - 0.5 * (value - low) / (high - low)


def _magnitude_monotonicity(session: Session) -> float:
    """Score how monotonic the magnitude trend is across events (0.0-1.0).

    AC compressors show a consistent or gradually-trending magnitude:
    people set a temperature and the compressor stabilises.  If they adjust
    the thermostat, magnitude shifts once -- but never zig-zags.

    Returns the fraction of consecutive magnitude changes that follow the
    dominant direction.  A perfectly monotonic sequence returns 1.0, a
    random sequence returns ~0.5.

    Requires >=3 events (>=2 consecutive differences).  Returns 1.0 for
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
        (duration_cv, gap_cv) -- both 0.0-inf.  Lower = more regular.
        Returns (0, 0) if too few events to compute.
    """
    if not session.events or len(session.events) < 3:
        return 0.0, 0.0

    sorted_events = sorted(session.events, key=lambda e: e.get('on_start', ''))

    # Duration CV (excluding first event -- the initial long run)
    durations = [
        e.get('duration', 0) or 0
        for e in sorted_events[1:]  # skip initial activation
        if (e.get('duration') or 0) > 0
    ]
    if len(durations) >= 2 and np.mean(durations) > 0:
        duration_cv = float(np.std(durations) / np.mean(durations))
    else:
        duration_cv = 0.0

    # Gap CV -- time between consecutive events
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
