"""
Stage 2 matching: noisy matching for events with other devices active between ON and OFF.

Finds matches where:
- Same phase
- Similar magnitudes between ON and OFF
- Power never drops significantly below baseline (device stays on)
"""
import pandas as pd
from .validator import is_valid_event_removal


def find_noisy_match(data: pd.DataFrame, on_event: dict, off_events: pd.DataFrame,
                     max_time_diff: int, max_magnitude_diff: int, logger):
    """
    Stage 2 matcher: finds matches when there's noise (other devices) between ON and OFF.

    Criteria:
    1. Same phase
    2. OFF within time window (6 hours)
    3. ON and OFF magnitudes are similar (±max_magnitude_diff)
    4. Power never drops below baseline - 200W (device stays on)

    These matches get tag "NOISY" and will use clipped cumsum in segregation.

    Args:
        data: DataFrame with power data
        on_event: Unmatched ON event dict
        off_events: DataFrame of remaining unmatched OFF events
        max_time_diff: Maximum hours between ON and OFF
        max_magnitude_diff: Maximum watts difference for magnitude similarity
        logger: Logger instance

    Returns:
        Tuple of (matched_off_event, "NOISY") or (None, None) if no match found
    """
    phase = on_event['phase']
    on_id = on_event['event_id']
    on_end = on_event['end']
    on_magnitude = abs(on_event['magnitude'])

    logger.info(f"[Stage 2] Processing unmatched ON event: ID={on_id}, Phase={phase}, Magnitude={on_magnitude}")

    # Find OFF events in same phase within time window
    candidates = off_events[
        (off_events['phase'] == phase) &
        (off_events['start'] > on_end) &
        (off_events['start'] - on_end <= pd.Timedelta(hours=max_time_diff))
    ]

    if candidates.empty:
        logger.info(f"[Stage 2] No OFF events found for {on_id} within time window.")
        return None, None

    # Filter by magnitude similarity
    candidates = candidates.copy()
    candidates['magnitude_diff'] = abs(abs(candidates['magnitude']) - on_magnitude)
    candidates = candidates[candidates['magnitude_diff'] <= max_magnitude_diff]

    if candidates.empty:
        logger.info(f"[Stage 2] No OFF events with similar magnitude found for {on_id}.")
        return None, None

    # Sort by magnitude similarity first, then by time
    candidates['time_diff'] = (candidates['start'] - on_end).abs()
    candidates = candidates.sort_values(by=['magnitude_diff', 'time_diff'])

    logger.info(f"[Stage 2] Found {len(candidates)} magnitude-compatible OFF events for {on_id}.")

    for _, off_event in candidates.iterrows():
        off_start = off_event['start']
        off_id = off_event['event_id']
        off_magnitude = abs(off_event['magnitude'])

        logger.info(f"[Stage 2] Checking {on_id} with {off_id}: ON={on_magnitude}W, OFF={off_magnitude}W")

        # Get data between ON and OFF
        event_range = (data['timestamp'] > on_end) & (data['timestamp'] < off_start)
        phase_data = data.loc[event_range, phase]

        if phase_data.empty:
            continue

        # Check that power never drops significantly below baseline
        # (device must stay on the whole time)
        baseline_power = phase_data.iloc[0]
        min_power = phase_data.min()
        min_allowed = baseline_power - 200  # Allow 200W tolerance

        if min_power < min_allowed:
            logger.info(f"[Stage 2] Rejected {on_id}-{off_id}: power dropped to {min_power}W (baseline={baseline_power}W)")
            continue

        # Validate that removal won't create negative values
        off_event_dict = off_event.to_dict()
        off_event_dict['phase'] = phase
        on_event_dict = {
            'start': on_event['start'],
            'end': on_end,
            'magnitude': on_magnitude,
            'phase': phase,
            'event_id': on_id
        }

        if not is_valid_event_removal(data, on_event_dict, off_event_dict, logger):
            logger.info(f"[Stage 2] Rejected {on_id}-{off_id}: would create negative values")
            continue

        logger.info(f"[Stage 2] Matched {on_id} with {off_id} as NOISY: ON={on_magnitude}W, OFF={off_magnitude}W")
        return off_event, "NOISY"

    logger.info(f"[Stage 2] No suitable noisy match found for {on_id}.")
    return None, None
