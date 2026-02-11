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
) -> Tuple[pd.DataFrame, dict, List, List]:
    """
    Process segmentation for a single phase.

    Args:
        data: DataFrame with timestamp and phase power columns
        events: DataFrame with matched events (on_start, on_end, off_start, off_end, etc.)
        phase: Phase name (w1, w2, w3)
        logger: Logger instance

    Returns:
        Tuple of (updated_data, new_columns_dict, errors_list, skipped_event_ids)
    """
    unique_durations = np.sort(events[events['phase'] == phase]['duration'].unique())

    diff_col = f'{phase}_diff'  # Must match validator.py format
    remaining_col = f'remaining_power_{phase}'

    # Use existing diff column if present, otherwise create it
    if diff_col not in data.columns:
        data[diff_col] = data[phase].diff()
    data[remaining_col] = data[phase].values  # .values avoids SettingWithCopyWarning

    new_columns = {}
    errors_log = []
    skipped_event_ids = []

    for duration in unique_durations:
        duration = int(duration)
        event_power_col = f'event_power_{duration}_m_{phase}'
        event_power_values = np.zeros(len(data))

        phase_events = events[
            (events['duration'] == duration) & (events['phase'] == phase)
        ].sort_values(by='on_start', ascending=True)

        # Use to_dict('records') instead of iterrows - 10-100x faster
        for event in phase_events.to_dict('records'):
            errors, was_skipped = _process_single_event(
                data, event, phase, duration,
                diff_col, remaining_col, event_power_values, logger
            )
            errors_log.extend(errors)
            if was_skipped:
                skipped_event_ids.append(event.get('on_event_id'))

        logger.info(f"Finished recording events with {duration} min duration for phase {phase}.")
        new_columns[event_power_col] = event_power_values

    # Don't drop diff column - it might be needed for other phases or validation
    # data.drop(columns=diff_col, inplace=True)

    return data, new_columns, errors_log, skipped_event_ids


def _process_single_event(
    data: pd.DataFrame,
    event: dict,
    phase: str,
    duration: int,
    diff_col: str,
    remaining_col: str,
    event_power_values: np.ndarray,
    logger
) -> Tuple[List, bool]:
    """
    Process a single event and update power columns.

    Returns:
        Tuple of (errors_list, was_skipped):
        - errors_list: list of error timestamps if negative values detected
        - was_skipped: True if the event was skipped due to negative values
    """
    errors = []

    # Define time ranges
    on_range = (data['timestamp'] >= event['on_start']) & (data['timestamp'] <= event['on_end'])
    event_range = (data['timestamp'] > event['on_end']) & (data['timestamp'] < event['off_start'])
    off_range = (data['timestamp'] >= event['off_start']) & (data['timestamp'] <= event['off_end'] - pd.Timedelta(minutes=1))

    magnitude = event['on_magnitude']
    tag = event.get('tag', '')
    is_noisy = 'NOISY' in tag
    is_partial = 'PARTIAL' in tag

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

    # Track device power continuously via diffs (for standard events)
    device_power = on_seg.iloc[-1] if not on_seg.empty else magnitude

    # Process EVENT segment (stable period between ON end and OFF start)
    if is_noisy:
        current_remaining = data.loc[event_range, remaining_col]
        event_seg = current_remaining.clip(upper=magnitude)
    elif is_partial:
        # For partial: constant magnitude extraction, clipped to remaining
        event_seg = data.loc[event_range, remaining_col].clip(upper=magnitude).clip(lower=0)
    else:
        # Constant device power, clipped to remaining to avoid negative remaining
        event_seg = data.loc[event_range, remaining_col].clip(upper=device_power).clip(lower=0)

    event_remain = data.loc[event_range, remaining_col] - event_seg

    # Update device power for OFF segment (for standard events)
    if not is_noisy and not is_partial and not event_seg.empty:
        device_power = event_seg.iloc[-1]

    # Process OFF segment (device ramp-down)
    if is_noisy:
        off_seg, off_remain = _process_noisy_off_segment(data, event, remaining_col)
    elif is_partial:
        off_cumsum = data.loc[off_range, diff_col].cumsum()
        off_seg = (magnitude + off_cumsum).clip(lower=0)
        off_seg = off_seg.clip(upper=data.loc[off_range, remaining_col])
        off_remain = data.loc[off_range, remaining_col] - off_seg
    else:
        # OFF ramp: track device power from end of event, clipped to remaining
        off_seg = (device_power + data.loc[off_range, diff_col].cumsum()).clip(lower=0)
        off_seg = off_seg.clip(upper=data.loc[off_range, remaining_col])
        off_remain = data.loc[off_range, remaining_col] - off_seg

    # Check for negative values BEFORE applying - skip event if would create negatives
    # Check both remaining power AND extracted event power
    would_create_negatives = (
        (on_remain < 0).any() or
        (event_remain < 0).any() or
        (off_remain < 0).any() or
        (on_seg < 0).any() or
        (event_seg < 0).any() or
        (off_seg < 0).any()
    )

    if would_create_negatives:
        event_id = event.get('on_event_id', f'dur_{duration}')
        logger.warning(f"Skipping event {event_id} in phase {phase}: would create negative remaining power")
        return errors, True  # Return without updating - skip this event

    # Only update columns if no negative values would be created
    event_power_values[on_range] = on_seg
    data.loc[on_range, remaining_col] = on_remain
    event_power_values[event_range] = event_seg
    data.loc[event_range, remaining_col] = event_remain
    event_power_values[off_range] = off_seg
    data.loc[off_range, remaining_col] = off_remain

    return errors, False


def _process_noisy_off_segment(
    data: pd.DataFrame,
    event: dict,
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
