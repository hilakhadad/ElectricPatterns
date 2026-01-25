"""
Core segmentation logic.

Extracts device power consumption from total power based on matched events.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List


def process_phase_segmentation(
    data: pd.DataFrame,
    events: pd.DataFrame,
    phase: str,
    logger
) -> Tuple[pd.DataFrame, dict, List]:
    """
    Process segmentation for a single phase.

    Args:
        data: DataFrame with timestamp and phase power columns
        events: DataFrame with matched events (on_start, on_end, off_start, off_end, etc.)
        phase: Phase name (w1, w2, w3)
        logger: Logger instance

    Returns:
        Tuple of (updated_data, new_columns_dict, errors_list)
    """
    unique_durations = np.sort(events[events['phase'] == phase]['duration'].unique())

    diff_col = f'diff_{phase}'
    remaining_col = f'remaining_power_{phase}'

    data[diff_col] = data[phase].diff()
    data[remaining_col] = data[phase].copy()

    new_columns = {}
    errors_log = []

    for duration in unique_durations:
        duration = int(duration)
        event_power_col = f'event_power_{duration}_m_{phase}'
        event_power_values = np.zeros(len(data))

        phase_events = events[
            (events['duration'] == duration) & (events['phase'] == phase)
        ].sort_values(by='on_start', ascending=True)

        for _, event in phase_events.iterrows():
            errors = _process_single_event(
                data, event, phase, duration,
                diff_col, remaining_col, event_power_values, logger
            )
            errors_log.extend(errors)

        logger.info(f"Finished recording events with {duration} min duration for phase {phase}.")
        new_columns[event_power_col] = event_power_values

    data.drop(columns=diff_col, inplace=True)

    return data, new_columns, errors_log


def _process_single_event(
    data: pd.DataFrame,
    event: pd.Series,
    phase: str,
    duration: int,
    diff_col: str,
    remaining_col: str,
    event_power_values: np.ndarray,
    logger
) -> List:
    """
    Process a single event and update power columns.

    Returns list of error timestamps if negative values detected.
    """
    errors = []

    # Define time ranges
    on_range = (data['timestamp'] >= event['on_start']) & (data['timestamp'] <= event['on_end'])
    event_range = (data['timestamp'] > event['on_end']) & (data['timestamp'] < event['off_start'])
    off_range = (data['timestamp'] >= event['off_start']) & (data['timestamp'] <= event['off_end'] - pd.Timedelta(minutes=1))

    magnitude = event['on_magnitude']
    tag = event.get('tag', '')
    is_noisy = tag == 'NOISY'
    is_partial = tag == 'PARTIAL'

    # Process ON segment
    # For PARTIAL matches: the ON event magnitude may be larger than match_magnitude
    # We only extract match_magnitude, not the full ON ramp
    if is_partial:
        # For partial: extract only match_magnitude worth of the ON ramp
        on_cumsum = data.loc[on_range, diff_col].cumsum()
        on_seg = on_cumsum.clip(upper=magnitude)  # Don't extract more than match_magnitude
    else:
        on_seg = data.loc[on_range, diff_col].cumsum()
    on_remain = data.loc[on_range, remaining_col] - on_seg

    if (on_remain < 0).any():
        error_timestamps = _get_negative_timestamps(data, on_remain)
        errors.extend(error_timestamps)
        logger.error(f"Negative values in ON phase {phase}, duration {duration}: {error_timestamps}")

    # Process EVENT segment
    if is_noisy:
        current_remaining = data.loc[event_range, remaining_col]
        event_seg = current_remaining.clip(upper=magnitude)
    elif is_partial:
        # For partial: constant magnitude extraction (device consumes match_magnitude)
        # The remainder stays in remaining_power
        event_seg = pd.Series(magnitude, index=data.loc[event_range].index)
    else:
        event_seg = magnitude + data.loc[event_range, diff_col].cumsum()

    event_remain = data.loc[event_range, remaining_col] - event_seg

    if (event_remain < 0).any():
        error_timestamps = _get_negative_timestamps(data, event_remain)
        errors.extend(error_timestamps)
        logger.error(f"Negative values in EVENT phase {phase}, duration {duration}: {error_timestamps}")

    # Process OFF segment
    if is_noisy:
        off_seg, off_remain = _process_noisy_off_segment(data, event, remaining_col)
    elif is_partial:
        # For partial: extract only match_magnitude worth of the OFF ramp
        # The remainder (difference between original OFF and match) stays
        off_cumsum = data.loc[off_range, diff_col].cumsum()
        # OFF cumsum is negative, we want to extract up to -magnitude
        off_seg = (magnitude + off_cumsum).clip(lower=0)
        off_remain = data.loc[off_range, remaining_col] - off_seg
    else:
        off_seg = magnitude + data.loc[off_range, diff_col].cumsum()
        off_remain = data.loc[off_range, remaining_col] - off_seg

    if (off_remain < 0).any():
        error_timestamps = _get_negative_timestamps(data, off_remain)
        errors.extend(error_timestamps)
        logger.error(f"Negative values in OFF phase {phase}, duration {duration}: {error_timestamps}")

    # Update columns
    event_power_values[on_range] = on_seg
    data.loc[on_range, remaining_col] = on_remain
    event_power_values[event_range] = event_seg
    data.loc[event_range, remaining_col] = event_remain
    event_power_values[off_range] = off_seg
    data.loc[off_range, remaining_col] = off_remain

    return errors


def _process_noisy_off_segment(
    data: pd.DataFrame,
    event: pd.Series,
    remaining_col: str
) -> Tuple[pd.Series, pd.Series]:
    """
    Process OFF segment for NOISY events.

    Aligns remaining to the value at off_end (when device is fully off).
    """
    off_range = (data['timestamp'] >= event['off_start']) & (data['timestamp'] <= event['off_end'] - pd.Timedelta(minutes=1))
    current_remaining = data.loc[off_range, remaining_col]

    if len(current_remaining) == 0:
        return current_remaining, current_remaining

    # Find target remaining value at off_end
    off_end_mask = data['timestamp'] == event['off_end']
    if off_end_mask.any():
        target_remaining = data.loc[off_end_mask, remaining_col].iloc[0]
    else:
        target_remaining = current_remaining.iloc[-1]

    off_seg = (current_remaining - target_remaining).clip(lower=0)
    off_remain = current_remaining - off_seg

    return off_seg, off_remain


def _get_negative_timestamps(data: pd.DataFrame, series: pd.Series) -> List:
    """Get timestamps where values are negative."""
    error_indices = series[series < 0].index
    return data.loc[error_indices.intersection(data.index), 'timestamp'].tolist()
