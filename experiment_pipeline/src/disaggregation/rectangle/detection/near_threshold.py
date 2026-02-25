"""
Near-threshold event detection.

Detects events where a single-minute diff is close to but below the threshold,
and extending by 1-3 adjacent minutes pushes the total magnitude over the threshold.

Example: threshold=1300W, diff at 19:57 = -1297W (misses by 3W).
Power at 19:55 = 1580W, power at 19:57 = 280W.
Extending to include 19:56: magnitude = 280 - 1580 = -1300W >= threshold.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


def _calc_magnitude_from_phase(data_indexed: pd.DataFrame, phase: str,
                                start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Calculate magnitude as power(end) - power(start - 1min)."""
    before_start = start - pd.Timedelta(minutes=1)
    try:
        value_end = data_indexed.loc[end, phase]
    except KeyError:
        return 0.0
    try:
        value_before = data_indexed.loc[before_start, phase]
    except KeyError:
        return 0.0
    return float(value_end - value_before)


def _is_covered(timestamp: pd.Timestamp, events_df: pd.DataFrame) -> bool:
    """Check if timestamp falls within any existing event's [start, end] range."""
    if len(events_df) == 0:
        return False
    return ((events_df['start'] <= timestamp) & (events_df['end'] >= timestamp)).any()


def _try_extend(data_indexed: pd.DataFrame, phase: str,
                near_miss_ts: pd.Timestamp, threshold: float,
                max_extend: int, event_type: str) -> Optional[Dict]:
    """
    Try extending a near-miss diff to reach the threshold.

    Tries progressively larger windows (shortest first):
    total=1: (1,0), (0,1)
    total=2: (1,1), (2,0), (0,2)
    total=3: (1,2), (2,1), (3,0), (0,3)
    ...

    For each window, computes magnitude from actual phase power values.
    No direction filtering on individual diffs — only total magnitude matters.

    Args:
        data_indexed: DataFrame with timestamp index and phase column
        phase: Phase column name
        near_miss_ts: Timestamp of the near-miss diff
        threshold: Detection threshold (positive value)
        max_extend: Max minutes to extend in each direction
        event_type: 'on' or 'off'

    Returns:
        dict with {start, end, magnitude} if threshold reached, else None
    """
    for total in range(1, 2 * max_extend + 1):
        for before in range(max(0, total - max_extend), min(total, max_extend) + 1):
            after = total - before

            start = near_miss_ts - pd.Timedelta(minutes=before)
            end = near_miss_ts + pd.Timedelta(minutes=after)

            magnitude = _calc_magnitude_from_phase(data_indexed, phase, start, end)

            if event_type == 'on' and magnitude >= threshold:
                return {'start': start, 'end': end, 'magnitude': magnitude}
            if event_type == 'off' and magnitude <= -threshold:
                return {'start': start, 'end': end, 'magnitude': magnitude}

    return None


def detect_near_threshold_events(
    data: pd.DataFrame,
    data_indexed: pd.DataFrame,
    diff_col: str,
    threshold: int,
    off_threshold: int,
    existing_on: pd.DataFrame,
    existing_off: pd.DataFrame,
    phase: str,
    min_factor: float = 0.85,
    max_extend_minutes: int = 3,
    logger=None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect near-threshold events by extending near-miss diffs with adjacent minutes.

    A near-miss diff is one where |diff| is close to but below the threshold.
    By including 1-3 adjacent minutes, the total magnitude may reach the threshold.

    Args:
        data: DataFrame with timestamp column and diff column
        data_indexed: Same data with timestamp as index (for fast lookups)
        diff_col: Name of the diff column (e.g., 'w1_diff')
        threshold: ON event threshold (e.g., 1300)
        off_threshold: OFF event threshold (e.g., 1300)
        existing_on: Already-detected ON events DataFrame
        existing_off: Already-detected OFF events DataFrame
        phase: Phase column name (e.g., 'w1')
        min_factor: Minimum diff as fraction of threshold to consider (default 0.85)
        max_extend_minutes: Max minutes to extend in each direction (default 3)

    Returns:
        Tuple of (new_on_events_df, new_off_events_df)
    """
    empty = pd.DataFrame(columns=['start', 'end', 'magnitude'])

    if len(data) < 2:
        return empty.copy(), empty.copy()

    diffs = data[diff_col].values
    timestamps = data['timestamp'].values

    # Near-miss ON: threshold * min_factor <= diff < threshold
    on_min = threshold * min_factor
    near_on_mask = (diffs >= on_min) & (diffs < threshold)

    # Near-miss OFF: -off_threshold < diff <= -off_threshold * min_factor
    off_min = off_threshold * min_factor
    near_off_mask = (diffs <= -off_min) & (diffs > -off_threshold)

    new_on_events = []
    new_off_events = []

    # Process near-miss ON events
    for idx in np.where(near_on_mask)[0]:
        ts = pd.Timestamp(timestamps[idx])

        if _is_covered(ts, existing_on):
            continue

        result = _try_extend(data_indexed, phase, ts, threshold,
                             max_extend_minutes, 'on')
        if result is not None:
            new_on_events.append(result)

    # Process near-miss OFF events
    for idx in np.where(near_off_mask)[0]:
        ts = pd.Timestamp(timestamps[idx])

        if _is_covered(ts, existing_off):
            continue

        result = _try_extend(data_indexed, phase, ts, off_threshold,
                             max_extend_minutes, 'off')
        if result is not None:
            new_off_events.append(result)

    on_df = pd.DataFrame(new_on_events) if new_on_events else empty.copy()
    off_df = pd.DataFrame(new_off_events) if new_off_events else empty.copy()

    if logger and (len(new_on_events) > 0 or len(new_off_events) > 0):
        logger.debug(f"Near-threshold {phase}: {len(new_on_events)} ON, {len(new_off_events)} OFF confirmed")
    return on_df, off_df
