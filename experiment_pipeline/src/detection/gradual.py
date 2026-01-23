"""
Gradual event detection.

Detects multi-minute power ramps that together exceed the threshold.
"""
import pandas as pd
import numpy as np
from typing import List


def _calc_magnitude_from_phase(df: pd.DataFrame, phase: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Calculate event magnitude as the difference between power at end and power before start.
    """
    before_start = start - pd.Timedelta(minutes=1)

    # Use index-based lookup if timestamp is the index
    if df.index.name == 'timestamp':
        try:
            value_end = df.loc[end, phase]
        except KeyError:
            value_end = 0
        try:
            value_before = df.loc[before_start, phase]
        except KeyError:
            value_before = 0
    else:
        # Fallback to boolean indexing
        end_mask = df['timestamp'] == end
        value_end = df.loc[end_mask, phase].values[0] if end_mask.any() else 0

        before_mask = df['timestamp'] == before_start
        value_before = df.loc[before_mask, phase].values[0] if before_mask.any() else 0

    return value_end - value_before


def detect_gradual_events(
    data: pd.DataFrame,
    diff_col: str,
    threshold: int,
    event_type: str = 'on',
    window_minutes: int = 3,
    partial_factor: float = 0.8,
    max_factor: float = 1.3,
    max_duration_minutes: int = 3,
    progressive_search: bool = False,
    phase: str = None
) -> pd.DataFrame:
    """
    Detect gradual ON/OFF events (multi-minute ramps).

    A gradual event is when power changes slowly over multiple minutes,
    with the total change reaching the threshold even though individual
    minute changes are below threshold.

    When progressive_search=True:
    - First try ±1 minute window
    - If no valid event found, try ±2 minutes
    - If still no valid event found, try ±3 minutes
    - Stop as soon as a valid event is found (prefer shorter windows)

    Args:
        data: DataFrame with timestamp and diff columns
        diff_col: Column name with power differences
        threshold: Full threshold (e.g., 1500W)
        event_type: 'on' or 'off'
        window_minutes: Maximum window to search (±N minutes)
        partial_factor: Minimum factor to be "interesting" (0.8 = 80%)
        max_factor: Maximum factor to avoid merging separate devices (1.3 = 130%)
        max_duration_minutes: Maximum duration for gradual event
        progressive_search: If True, try smaller windows first (1, 2, 3...)
        phase: Phase column name for accurate magnitude calculation

    Returns:
        DataFrame with gradual events: start, end, magnitude
    """
    # Infer phase from diff_col if not provided (e.g., 'w1_diff' -> 'w1')
    if phase is None:
        phase = diff_col.replace('_diff', '')

    if len(data) < 2:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    # Keep phase column for magnitude calculation
    cols_to_keep = ['timestamp', diff_col]
    if phase in data.columns:
        cols_to_keep.append(phase)
    df = data[cols_to_keep].copy().sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=[diff_col])

    if len(df) < 2:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    partial_threshold = threshold * partial_factor  # e.g., 1200W for 1500W
    max_threshold = threshold * max_factor  # e.g., 1950W for 1500W

    timestamps = df['timestamp'].values
    diffs = df[diff_col].values

    # Pre-filter to "significant" changes (>= 50% of threshold)
    min_significant = threshold * 0.5

    if event_type == 'on':
        candidate_mask = (diffs >= min_significant)
    else:
        candidate_mask = (diffs <= -min_significant)

    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    events = []
    used_indices = set()

    # Determine which windows to try
    if progressive_search:
        windows_to_try = list(range(1, window_minutes + 1))
    else:
        windows_to_try = [window_minutes]

    for i in candidate_indices:
        if i in used_indices:
            continue

        event_found = False
        for current_window in windows_to_try:
            if event_found:
                break

            # Try symmetric window first, then asymmetric
            window_configs = [
                (current_window, current_window),  # Symmetric: ±N minutes
                (current_window, 0),               # Only before
                (0, current_window),               # Only after
            ]

            for before_mins, after_mins in window_configs:
                if event_found:
                    break

                event = _try_window(
                    i, timestamps, diffs, used_indices,
                    before_mins, after_mins,
                    event_type, threshold,
                    partial_threshold, max_threshold,
                    max_duration_minutes
                )

                if event is not None:
                    events.append(event['event'])
                    used_indices.update(event['used_indices'])
                    event_found = True

    if not events:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    result_df = pd.DataFrame(events).sort_values('start').reset_index(drop=True)
    result_df = result_df.drop_duplicates(subset=['start', 'end'], keep='first').reset_index(drop=True)

    # Recalculate magnitude using actual phase values (not sum of diffs)
    if phase in df.columns and len(result_df) > 0:
        result_df['magnitude'] = result_df.apply(
            lambda row: _calc_magnitude_from_phase(df, phase, row['start'], row['end']),
            axis=1
        )

    return result_df


def _try_window(
    i: int,
    timestamps: np.ndarray,
    diffs: np.ndarray,
    used_indices: set,
    before_mins: int,
    after_mins: int,
    event_type: str,
    threshold: int,
    partial_threshold: float,
    max_threshold: float,
    max_duration_minutes: int
) -> dict:
    """
    Try to find a valid gradual event in a specific time window.

    Returns:
        dict with 'event' and 'used_indices' if found, None otherwise
    """
    start_time = timestamps[i] - np.timedelta64(before_mins, 'm')
    end_time = timestamps[i] + np.timedelta64(after_mins, 'm')

    window_mask = (timestamps >= start_time) & (timestamps <= end_time)
    window_indices = np.where(window_mask)[0]

    if len(window_indices) < 1:
        return None

    # Filter to adjacent indices that are NOT already used
    adjacent_indices = _get_adjacent_unused_indices(
        window_indices, timestamps, used_indices
    )

    if len(adjacent_indices) < 1:
        return None

    adjacent_indices = np.array(adjacent_indices)
    window_diffs = diffs[adjacent_indices]
    window_sum = window_diffs.sum()

    # Check direction consistency
    if not _check_direction_consistency(window_diffs, event_type):
        return None

    # Check duration constraint
    min_significant_for_duration = threshold * 0.05
    significant_mask = np.abs(diffs[adjacent_indices]) >= min_significant_for_duration
    significant_indices = adjacent_indices[significant_mask]

    if len(significant_indices) == 0:
        return None

    duration_minutes = len(significant_indices)
    if duration_minutes > max_duration_minutes:
        return None

    # Check if magnitude is in valid range (80%-130% of threshold)
    abs_sum = abs(window_sum)
    if not (partial_threshold <= abs_sum <= max_threshold):
        return None

    # Valid event found!
    return {
        'event': {
            'start': timestamps[significant_indices[0]],
            'end': timestamps[significant_indices[-1]],
            'magnitude': window_sum
        },
        'used_indices': set(adjacent_indices)
    }


def _get_adjacent_unused_indices(
    window_indices: np.ndarray,
    timestamps: np.ndarray,
    used_indices: set
) -> List[int]:
    """Get adjacent indices that haven't been used."""
    adjacent = []
    for idx in sorted(window_indices):
        if idx in used_indices:
            continue
        if not adjacent:
            adjacent.append(idx)
        else:
            # Check if within 2 minutes of last added
            time_diff = (timestamps[idx] - timestamps[adjacent[-1]]) / np.timedelta64(1, 'm')
            if time_diff <= 2:
                adjacent.append(idx)
            else:
                break
    return adjacent


def _check_direction_consistency(window_diffs: np.ndarray, event_type: str) -> bool:
    """
    Check that opposite-direction changes are small enough (noise).

    Allow max(50W, 3% of primary direction) in opposite direction.
    """
    if event_type == 'on':
        primary_sum = window_diffs[window_diffs > 0].sum()
        opposite_sum = abs(window_diffs[window_diffs < 0].sum())
    else:
        primary_sum = abs(window_diffs[window_diffs < 0].sum())
        opposite_sum = window_diffs[window_diffs > 0].sum()

    noise_threshold = max(50, abs(primary_sum) * 0.03)
    return opposite_sum <= noise_threshold
