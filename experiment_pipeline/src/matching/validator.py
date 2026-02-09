"""
Event removal validation.

Validates that removing matched events won't create negative power values.

Performance optimized: Uses numpy arrays with searchsorted for O(log n) lookups
instead of O(n) boolean masking.

Supports correction mode: when negative values are small (within tolerance),
returns a correction amount to reduce match magnitude instead of rejecting.
"""
import pandas as pd
import numpy as np

# Default tolerance for negative values - if within this range, we correct instead of reject
DEFAULT_NEGATIVE_TOLERANCE = -10  # watts


def get_magnitude_quality_tag(on_magnitude: float, off_magnitude: float) -> str:
    """
    Get magnitude quality tag based on ON/OFF magnitude difference.

    Tags:
    - EXACT: < 50W difference
    - CLOSE: 50-100W difference
    - APPROX: 100-200W difference
    - LOOSE: 200-350W difference
    """
    diff = abs(abs(on_magnitude) - abs(off_magnitude))
    if diff < 50:
        return "EXACT"
    elif diff < 100:
        return "CLOSE"
    elif diff < 200:
        return "APPROX"
    else:
        return "LOOSE"


def get_duration_tag(duration_minutes: float) -> str:
    """
    Get duration tag based on event duration in minutes.

    Tags:
    - SPIKE: ≤ 2 minutes
    - QUICK: < 5 minutes (microwave, kettle)
    - MEDIUM: 5-30 minutes (washing machine, dishwasher)
    - EXTENDED: > 30 minutes (water heater, AC)
    """
    if duration_minutes <= 2:
        return "SPIKE"
    elif duration_minutes < 5:
        return "QUICK"
    elif duration_minutes <= 30:
        return "MEDIUM"
    else:
        return "EXTENDED"


def build_match_tag(on_magnitude: float, off_magnitude: float, duration_minutes: float,
                    is_noisy: bool = False, is_partial: bool = False, is_corrected: bool = False) -> str:
    """
    Build a complete match tag combining magnitude quality and duration.

    Format: [NOISY-|PARTIAL-]{magnitude_quality}-{duration}[-CORRECTED]

    Examples:
    - EXACT-SPIKE
    - CLOSE-QUICK
    - NOISY-APPROX-MEDIUM
    - PARTIAL-EXTENDED
    - EXACT-QUICK-CORRECTED
    """
    parts = []

    # Prefix for special match types
    if is_noisy:
        parts.append("NOISY")
    elif is_partial:
        parts.append("PARTIAL")

    # Magnitude quality (skip for partial since magnitudes differ significantly)
    if not is_partial:
        parts.append(get_magnitude_quality_tag(on_magnitude, off_magnitude))

    # Duration
    parts.append(get_duration_tag(duration_minutes))

    # Suffix for corrected
    if is_corrected:
        parts.append("CORRECTED")

    return "-".join(parts)


def is_valid_event_removal(data: pd.DataFrame, on_event: dict, off_event: dict, logger,
                           negative_tolerance: float = DEFAULT_NEGATIVE_TOLERANCE) -> tuple:
    """
    Validate that removing a matched ON-OFF pair won't create negative power values.

    Checks:
    1. Magnitude matching - ON and OFF magnitudes must be similar (within 350W)
    2. Remaining power (what's left after removing event) must not be negative
    3. Event power itself must not be negative

    When negative values are small (within negative_tolerance), returns a correction
    amount instead of rejecting. The caller should reduce the match magnitude by this
    amount to avoid negative values.

    Args:
        data: DataFrame with power data and diff columns
        on_event: ON event dict with start, end, magnitude, phase, event_id
        off_event: OFF event dict with start, end, magnitude, event_id
        logger: Logger instance
        negative_tolerance: Minimum negative value to allow with correction (default -10W)

    Returns:
        Tuple of (is_valid, correction):
        - is_valid: True if removal is valid (possibly with correction)
        - correction: 0 if no correction needed, positive value if magnitude should be reduced
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
    threshold = 350
    if magnitude_diff > threshold:
        logger.info(f"REJECTED {on_id}-{off_id}: magnitude_mismatch (on={on_magnitude:.0f}W, off={off_magnitude:.0f}W, diff={magnitude_diff:.0f}W, threshold={threshold}W)")
        return False, 0

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
        return False, 0

    # Check remaining power - must not be negative
    min_remain = min(
        np.min(on_remain) if len(on_remain) > 0 else 0,
        np.min(event_remain) if len(event_remain) > 0 else 0,
        np.min(off_remain) if len(off_remain) > 0 else 0
    )

    # Check event power itself - must not be negative
    min_seg = min(
        np.min(on_seg) if len(on_seg) > 0 else 0,
        np.min(event_seg) if len(event_seg) > 0 else 0,
        np.min(off_seg) if len(off_seg) > 0 else 0
    )

    # Calculate required correction (if any)
    min_value = min(min_remain, min_seg)

    if min_value >= 0:
        # No correction needed
        return True, 0

    if min_value >= negative_tolerance:
        # Small negative value - can be corrected by reducing magnitude
        correction = abs(min_value)
        logger.info(f"CORRECTABLE {on_id}-{off_id}: small_negative (min={min_value:.0f}W, correction={correction:.0f}W)")
        return True, correction

    # Negative value too large - reject
    if min_remain < min_seg:
        logger.info(f"REJECTED {on_id}-{off_id}: negative_remaining_power (min={min_remain:.0f}W, tolerance={negative_tolerance}W)")
    else:
        logger.info(f"REJECTED {on_id}-{off_id}: negative_event_power (min={min_seg:.0f}W, tolerance={negative_tolerance}W)")
    return False, 0


def is_valid_partial_removal(data: pd.DataFrame, on_event: dict, off_event, match_magnitude: float, logger,
                              negative_tolerance: float = DEFAULT_NEGATIVE_TOLERANCE) -> tuple:
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
        negative_tolerance: Minimum negative value to allow with correction (default -10W)

    Returns:
        Tuple of (is_valid, correction):
        - is_valid: True if removal is valid (possibly with correction)
        - correction: 0 if no correction needed, positive value if magnitude should be reduced
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
        return False, 0

    # Check remaining power - must not be negative
    min_remain = min(
        np.min(on_remain) if len(on_remain) > 0 else 0,
        np.min(event_remain) if len(event_remain) > 0 else 0,
        np.min(off_remain) if len(off_remain) > 0 else 0
    )

    # Check event power itself - must not be negative
    min_seg = min(
        np.min(on_seg) if len(on_seg) > 0 else 0,
        np.min(event_seg) if len(event_seg) > 0 else 0,
        np.min(off_seg) if len(off_seg) > 0 else 0
    )

    # Calculate required correction (if any)
    min_value = min(min_remain, min_seg)

    if min_value >= 0:
        # No correction needed
        return True, 0

    if min_value >= negative_tolerance:
        # Small negative value - can be corrected by reducing magnitude
        correction = abs(min_value)
        logger.info(f"CORRECTABLE {on_id}-{off_id}: partial_small_negative (min={min_value:.0f}W, correction={correction:.0f}W)")
        return True, correction

    # Negative value too large - reject
    if min_remain < min_seg:
        logger.info(f"REJECTED {on_id}-{off_id}: partial_negative_remaining (min={min_remain:.0f}W, tolerance={negative_tolerance}W)")
    else:
        logger.info(f"REJECTED {on_id}-{off_id}: partial_negative_event (min={min_seg:.0f}W, tolerance={negative_tolerance}W)")
    return False, 0
