"""
Stage 2 matching: noisy matching for events with other devices active between ON and OFF.

Finds matches where:
- Same phase
- Similar magnitudes between ON and OFF
- Power never drops significantly below baseline (device stays on)

Performance optimizations:
- Pre-filter candidates by magnitude similarity in initial query
- Sort by magnitude similarity first to find best matches faster
"""
import pandas as pd
from .validator import is_valid_event_removal


def find_noisy_match(data: pd.DataFrame, on_event: dict, off_events: pd.DataFrame,
                     max_time_diff: int, max_magnitude_diff: int, logger):
    """
    Stage 2 matcher: finds matches when there's noise (other devices) between ON and OFF.

    Uses progressive window search: starts with small time window and expands
    gradually to find the closest match first.

    Criteria:
    1. Same phase
    2. OFF within time window (progressive: 15min -> 30min -> 1hr -> 2hr -> 4hr -> max)
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

    logger.debug(f"[Stage 2] Processing {on_id}, Phase={phase}, Magnitude={on_magnitude}W")

    # Progressive window search: start small and expand
    # Windows in minutes: 15min, 30min, 1hr, 2hr, 4hr, then max_time_diff
    window_steps_minutes = [15, 30, 60, 120, 240, max_time_diff * 60]

    for window_minutes in window_steps_minutes:
        if window_minutes > max_time_diff * 60:
            break

        # Pre-filter by phase, time, AND magnitude similarity in one query
        candidates = off_events[
            (off_events['phase'] == phase) &
            (off_events['start'] > on_end) &
            (off_events['start'] - on_end <= pd.Timedelta(minutes=window_minutes)) &
            (abs(abs(off_events['magnitude']) - on_magnitude) <= max_magnitude_diff)
        ]

        if candidates.empty:
            continue

        candidates = candidates.copy()
        candidates['magnitude_diff'] = abs(abs(candidates['magnitude']) - on_magnitude)
        candidates['time_diff'] = (candidates['start'] - on_end).abs()
        # Sort by magnitude similarity first, then by time
        candidates = candidates.sort_values(by=['magnitude_diff', 'time_diff'])

        for _, off_event in candidates.iterrows():
            off_start = off_event['start']
            off_id = off_event['event_id']

            # Get data between ON and OFF
            event_range = (data['timestamp'] > on_end) & (data['timestamp'] < off_start)
            phase_data = data.loc[event_range, phase]

            if phase_data.empty:
                continue

            # Check that power never drops significantly below baseline
            baseline_power = phase_data.iloc[0]
            min_power = phase_data.min()
            min_allowed = baseline_power - 200

            if min_power < min_allowed:
                logger.debug(f"[Stage 2] Rejected {on_id}-{off_id}: power dropped below baseline")
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
                logger.debug(f"[Stage 2] Rejected {on_id}-{off_id}: negative residuals")
                continue

            logger.info(f"Matched NOISY: {on_id} <-> {off_id} (window={window_minutes}m)")
            return off_event, "NOISY"

    return None, None
