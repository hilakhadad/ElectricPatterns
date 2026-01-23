"""
Event removal validation.

Validates that removing matched events won't create negative power values.
"""
import pandas as pd


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

    on_range = (data['timestamp'] >= on_event['start']) & (data['timestamp'] <= on_event['end'])
    event_range = (data['timestamp'] > on_event['end']) & (data['timestamp'] < off_event['start'])
    off_range = (data['timestamp'] >= off_event['start']) & (data['timestamp'] <= off_event['end'] - pd.Timedelta(minutes=1))

    magnitude = on_magnitude

    try:
        on_seg = data.loc[on_range, diff_col].cumsum()
        on_remain = data.loc[on_range, power_col] - on_seg
        event_seg = magnitude + data.loc[event_range, diff_col].cumsum()
        event_remain = data.loc[event_range, power_col] - event_seg
        off_seg = magnitude + data.loc[off_range, diff_col].cumsum()
        off_remain = data.loc[off_range, power_col] - off_seg
    except Exception as e:
        logger.error(f"Error adjusting data ranges of {on_id} and {off_id}: {e}")
        return False

    # Check remaining power - must not be negative
    if (on_remain < 0).any() or (event_remain < 0).any() or (off_remain < 0).any():
        logger.debug(f"Validator rejected {on_id}-{off_id}: negative remaining power")
        return False

    # Check event power itself - must not be negative
    if (on_seg < 0).any() or (event_seg < 0).any() or (off_seg < 0).any():
        logger.debug(f"Validator rejected {on_id}-{off_id}: negative event power")
        return False

    return True
