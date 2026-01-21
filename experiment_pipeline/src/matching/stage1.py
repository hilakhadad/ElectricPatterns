"""
Stage 1 matching: clean matching for events without noise between ON and OFF.

Finds matches where:
- Same phase
- OFF event occurs after ON event
- Power between ON and OFF is stable (no significant deviations)
"""
import pandas as pd
from .validator import is_valid_event_removal


def find_match(data: pd.DataFrame, on_event: dict, off_events: pd.DataFrame,
               max_time_diff: int, max_magnitude_diff: int, logger):
    """
    Find a matching OFF event for an ON event (Stage 1 - clean matching).

    Matches are classified as:
    - SPIKE: ON and OFF are within 2 minutes of each other
    - NON-M: Power is stable between ON and OFF (deviation < max_magnitude_diff)

    Args:
        data: DataFrame with power data
        on_event: ON event dict
        off_events: DataFrame of potential OFF events
        max_time_diff: Maximum hours between ON and OFF
        max_magnitude_diff: Maximum watts deviation allowed between events
        logger: Logger instance

    Returns:
        Tuple of (matched_off_event, tag) or (None, None) if no match found
    """
    phase = on_event['phase']
    on_id = on_event['event_id']
    on_start = on_event['start']
    on_end = on_event['end']

    logger.info(f"Processing ON event: ID={on_id}, Phase={phase}, End Time={on_end}")

    candidates = off_events[
        (off_events['phase'] == phase) &
        (off_events['start'] > on_end) &
        (off_events['start'] - on_end <= pd.Timedelta(hours=max_time_diff))
    ]

    if candidates.empty:
        logger.info(f"No matching OFF events found for {on_id} within the allowed time window.")
        return None, None

    candidates = candidates.copy()
    candidates['time_diff'] = (candidates['start'] - on_end).abs()
    candidates = candidates.sort_values(by='time_diff')

    logger.info(f"Found {len(candidates)} potential OFF events for {on_id}. Sorting by closest time difference.")

    for _, off_event in candidates.iterrows():
        off_end = off_event['end']
        off_start = off_event['start']
        off_id = off_event['event_id']
        logger.info(f"Checking for {on_id} OFF event {off_id} at {off_start} for match.")

        # Check for SPIKE match (very short duration)
        if (off_end - on_start).total_seconds() / 60 <= 2:
            if not is_valid_event_removal(data, on_event, off_event, logger):
                logger.warning(
                    f"Skipping incorrect match for ON event {on_id} and OFF event {off_id} due to negative residuals.")
                continue
            else:
                logger.info(f"On {on_id} and off {off_id} - matched as SPIKE: ON at {on_end}, OFF at {off_start}")
                return off_event, "SPIKE"

        # Check for NON-M match (stable power between events)
        event_range = (data['timestamp'] > on_end) & (data['timestamp'] < off_start)
        phase_data = data.loc[event_range, phase]

        if phase_data.empty:
            logger.info(f"No data found in the time range between On {on_id} and off {off_id} events.")
            continue

        baseline_power = phase_data.iloc[0]
        max_deviation = (phase_data - baseline_power).max()
        min_deviation = (phase_data - baseline_power).min()

        logger.info(f"Between On {on_id} and off {off_id} events - Max deviation: {max_deviation}, Min deviation: {min_deviation}, Magnitude: {max_magnitude_diff}")

        if abs(max_deviation) <= max_magnitude_diff and abs(min_deviation) <= max_magnitude_diff:
            if not is_valid_event_removal(data, on_event, off_event, logger):
                logger.warning(
                    f"Skipping incorrect match for ON event {on_id} and OFF event {off_id} due to negative residuals.")
                continue
            else:
                logger.info(
                    f"On {on_id} and off {off_id} events - matched as NON-M: ON at {on_end}, OFF at {off_start}")
                return off_event, "NON-M"

    logger.info(f"No suitable OFF event found matching the criteria of {on_id}.")
    return None, None
