"""
Event removal validation.

Validates that removing matched events won't create negative power values.

Performance optimized: Uses numpy arrays with searchsorted for O(log n) lookups
instead of O(n) boolean masking.
"""
import pandas as pd
import numpy as np


def is_valid_event_removal(data: pd.DataFrame, on_event: dict, off_event: dict, logger) -> bool:
    """
    Validate that removing a matched ON-OFF pair won't create negative power values.

    Checks:
    1. Magnitude matching - ON and OFF magnitudes must be similar (within 350W)
    2. Remaining power (what's left after removing event) must not be negative
    3. Event power itself must not be negative

    Args:
        data: DataFrame with power data and diff columns
        on_event: ON event dict with start, end, magnitude, phase, event_id
        off_event: OFF event dict with start, end, magnitude, event_id
        logger: Logger instance

    Returns:
        True if removal is valid, False otherwise
    """
    phase = on_event['phase']
    on_id = on_event['event_id']
    off_id = off_event['event_id']

    power_col = phase
    diff_col = f"{phase}_diff"

    on_magnitude = abs(on_event['magnitude'])
    # Handle both dict and pandas Series for off_event
    try:
        off_magnitude = abs(off_event['magnitude'])
    except (KeyError, TypeError):
        off_magnitude = on_magnitude

    # Check magnitude matching - reject if difference is too large
    magnitude_diff = abs(on_magnitude - off_magnitude)
    if magnitude_diff > 350:
        logger.debug(f"Validator rejected {on_id}-{off_id}: magnitude mismatch ({on_magnitude}W vs {off_magnitude}W, diff={magnitude_diff}W)")
        return False

    # Get timestamps as numpy datetime64 array for searchsorted
    timestamps = data['timestamp'].values.astype('datetime64[ns]')

    # Convert event timestamps to numpy datetime64 for comparison
    on_start = np.datetime64(on_event['start'])
    on_end = np.datetime64(on_event['end'])
    off_start = np.datetime64(off_event['start'])
    off_end_time = np.datetime64(off_event['end'] - pd.Timedelta(minutes=1))

    # Use numpy searchsorted for O(log n) range lookups instead of O(n) boolean masking
    on_start_idx = np.searchsorted(timestamps, on_start)
    on_end_idx = np.searchsorted(timestamps, on_end, side='right')
    off_start_idx = np.searchsorted(timestamps, off_start)
    off_end_idx = np.searchsorted(timestamps, off_end_time, side='right')

    # Get numpy arrays for the columns
    power_arr = data[power_col].values
    diff_arr = data[diff_col].values

    magnitude = on_magnitude

    try:
        # ON segment: indices [on_start_idx, on_end_idx)
        on_diff = diff_arr[on_start_idx:on_end_idx]
        on_power = power_arr[on_start_idx:on_end_idx]
        if len(on_diff) > 0:
            on_seg = np.cumsum(on_diff)
            on_remain = on_power - on_seg
        else:
            on_seg = np.array([])
            on_remain = np.array([])

        # Event segment: indices (on_end_idx, off_start_idx)
        event_diff = diff_arr[on_end_idx:off_start_idx]
        event_power = power_arr[on_end_idx:off_start_idx]
        if len(event_diff) > 0:
            event_seg = magnitude + np.cumsum(event_diff)
            event_remain = event_power - event_seg
        else:
            event_seg = np.array([])
            event_remain = np.array([])

        # OFF segment: indices [off_start_idx, off_end_idx)
        off_diff = diff_arr[off_start_idx:off_end_idx]
        off_power = power_arr[off_start_idx:off_end_idx]
        if len(off_diff) > 0:
            off_seg = magnitude + np.cumsum(off_diff)
            off_remain = off_power - off_seg
        else:
            off_seg = np.array([])
            off_remain = np.array([])
    except Exception as e:
        logger.error(f"Error adjusting data ranges of {on_id} and {off_id}: {e}")
        return False

    # Check remaining power - must not be negative
    if (len(on_remain) > 0 and np.any(on_remain < 0)) or \
       (len(event_remain) > 0 and np.any(event_remain < 0)) or \
       (len(off_remain) > 0 and np.any(off_remain < 0)):
        logger.debug(f"Validator rejected {on_id}-{off_id}: negative remaining power")
        return False

    # Check event power itself - must not be negative
    if (len(on_seg) > 0 and np.any(on_seg < 0)) or \
       (len(event_seg) > 0 and np.any(event_seg < 0)) or \
       (len(off_seg) > 0 and np.any(off_seg < 0)):
        logger.debug(f"Validator rejected {on_id}-{off_id}: negative event power")
        return False

    return True


def is_valid_partial_removal(data: pd.DataFrame, on_event: dict, off_event, match_magnitude: float, logger) -> bool:
    """
    Validate that removing a partial match won't create negative power values.

    Like is_valid_event_removal but uses match_magnitude instead of the ON event magnitude.
    Used for Stage 3 partial matching where ON and OFF have different magnitudes.

    Args:
        data: DataFrame with power data and diff columns
        on_event: ON event dict with start, end, magnitude, phase, event_id
        off_event: OFF event (dict or pandas Series) with start, end, magnitude, event_id
        match_magnitude: The magnitude to use for removal (min of ON and OFF)
        logger: Logger instance

    Returns:
        True if removal is valid, False otherwise
    """
    phase = on_event['phase']
    on_id = on_event['event_id']

    # Handle both dict and pandas Series for off_event
    try:
        off_id = off_event['event_id']
        off_start = off_event['start']
        off_end = off_event['end']
    except (KeyError, TypeError):
        off_id = "unknown"
        off_start = off_event.get('start', on_event['end'])
        off_end = off_event.get('end', off_start)

    power_col = phase
    diff_col = f"{phase}_diff"

    # Get timestamps as numpy datetime64 array for searchsorted
    timestamps = data['timestamp'].values.astype('datetime64[ns]')

    # Convert event timestamps to numpy datetime64 for comparison
    on_start_np = np.datetime64(on_event['start'])
    on_end_np = np.datetime64(on_event['end'])
    off_start_np = np.datetime64(off_start)
    off_end_time_np = np.datetime64(off_end - pd.Timedelta(minutes=1))

    # Use numpy searchsorted for O(log n) range lookups
    on_start_idx = np.searchsorted(timestamps, on_start_np)
    on_end_idx = np.searchsorted(timestamps, on_end_np, side='right')
    off_start_idx = np.searchsorted(timestamps, off_start_np)
    off_end_idx = np.searchsorted(timestamps, off_end_time_np, side='right')

    # Get numpy arrays for the columns
    power_arr = data[power_col].values
    diff_arr = data[diff_col].values

    try:
        # ON segment
        on_diff = diff_arr[on_start_idx:on_end_idx]
        on_power = power_arr[on_start_idx:on_end_idx]
        if len(on_diff) > 0:
            on_seg = np.cumsum(on_diff)
            on_remain = on_power - on_seg
        else:
            on_seg = np.array([])
            on_remain = np.array([])

        # Event segment
        event_diff = diff_arr[on_end_idx:off_start_idx]
        event_power = power_arr[on_end_idx:off_start_idx]
        if len(event_diff) > 0:
            event_seg = match_magnitude + np.cumsum(event_diff)
            event_remain = event_power - event_seg
        else:
            event_seg = np.array([])
            event_remain = np.array([])

        # OFF segment
        off_diff = diff_arr[off_start_idx:off_end_idx]
        off_power = power_arr[off_start_idx:off_end_idx]
        if len(off_diff) > 0:
            off_seg = match_magnitude + np.cumsum(off_diff)
            off_remain = off_power - off_seg
        else:
            off_seg = np.array([])
            off_remain = np.array([])
    except Exception as e:
        logger.error(f"Error adjusting data ranges of {on_id} and {off_id}: {e}")
        return False

    # Check remaining power - must not be negative
    if (len(on_remain) > 0 and np.any(on_remain < 0)) or \
       (len(event_remain) > 0 and np.any(event_remain < 0)) or \
       (len(off_remain) > 0 and np.any(off_remain < 0)):
        logger.debug(f"Partial validator rejected {on_id}-{off_id}: negative remaining power")
        return False

    # Check event power itself - must not be negative
    if (len(on_seg) > 0 and np.any(on_seg < 0)) or \
       (len(event_seg) > 0 and np.any(event_seg < 0)) or \
       (len(off_seg) > 0 and np.any(off_seg < 0)):
        logger.debug(f"Partial validator rejected {on_id}-{off_id}: negative event power")
        return False

    return True
