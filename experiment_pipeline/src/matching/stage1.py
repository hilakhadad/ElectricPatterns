"""
Stage 1 matching: clean matching for events without noise between ON and OFF.

Finds matches where:
- Same phase
- OFF event occurs after ON event
- Power between ON and OFF is stable (no significant deviations)

Performance optimizations:
- Pre-filter candidates by magnitude similarity (skip mismatches early)
- Sort by magnitude similarity first to find best matches faster
"""
import pandas as pd
from .validator import is_valid_event_removal

# Maximum magnitude difference to even consider a match
MAX_MAGNITUDE_DIFF_FILTER = 350  # watts

# If magnitude diff is below this threshold, skip stability check and match directly
# This handles cases where an appliance has matching ON/OFF but other devices caused
# power fluctuations during the event's "on" period
SMALL_MAGNITUDE_DIFF_THRESHOLD = 100  # watts


def find_match(data: pd.DataFrame, on_event: dict, off_events: pd.DataFrame,
               max_time_diff: int, max_magnitude_diff: int, logger):
    """
    Find a matching OFF event for an ON event (Stage 1 - clean matching).

    Uses progressive window search: starts with small time window and expands
    gradually to find the closest match first.

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

    logger.debug(f"Processing ON event: ID={on_id}, Phase={phase}, End Time={on_end}")

    # Progressive window search: start small and expand
    # Windows in minutes: 15min, 30min, 1hr, 2hr, 4hr, then max_time_diff
    window_steps_minutes = [15, 30, 60, 120, 240, max_time_diff * 60]
    on_magnitude = abs(on_event['magnitude'])

    for window_minutes in window_steps_minutes:
        if window_minutes > max_time_diff * 60:
            break

        # Pre-filter by phase, time, AND magnitude similarity
        candidates = off_events[
            (off_events['phase'] == phase) &
            (off_events['start'] > on_end) &
            (off_events['start'] - on_end <= pd.Timedelta(minutes=window_minutes)) &
            (abs(abs(off_events['magnitude']) - on_magnitude) <= MAX_MAGNITUDE_DIFF_FILTER)
        ]

        if candidates.empty:
            continue

        # Use .assign() to add columns without explicit copy
        candidates = candidates.assign(
            magnitude_diff=abs(abs(candidates['magnitude']) - on_magnitude),
            time_diff=(candidates['start'] - on_end).abs()
        ).sort_values(by=['magnitude_diff', 'time_diff'])

        # Log candidates summary on first window that has candidates
        if len(candidates) > 0 and window_minutes == window_steps_minutes[0]:
            summary = ", ".join([
                f"{row['event_id']}({abs(row['magnitude']):.0f}W, +{row['time_diff'].total_seconds()/60:.0f}m)"
                for _, row in candidates.head(5).iterrows()
            ])
            if len(candidates) > 5:
                summary += f", ... +{len(candidates)-5} more"
            logger.info(f"{on_id}({on_magnitude:.0f}W) candidates[{window_minutes}m]: {summary}")

        for _, off_event in candidates.iterrows():
            off_end = off_event['end']
            off_start = off_event['start']
            off_id = off_event['event_id']

            # Check for SPIKE match (very short duration)
            if (off_end - on_start).total_seconds() / 60 <= 2:
                if is_valid_event_removal(data, on_event, off_event, logger):
                    logger.info(f"Matched SPIKE: {on_id} <-> {off_id}")
                    return off_event, "SPIKE"
                else:
                    logger.debug(f"Skipping {on_id} <-> {off_id}: negative residuals")
                    continue

            # Get magnitude difference for this candidate
            off_magnitude = abs(off_event['magnitude'])
            mag_diff = abs(on_magnitude - off_magnitude)

            # Small magnitude difference fallback: if magnitudes match very closely,
            # skip stability check and rely only on validator
            # This handles cases where other devices cause fluctuations during the event
            if mag_diff <= SMALL_MAGNITUDE_DIFF_THRESHOLD:
                if is_valid_event_removal(data, on_event, off_event, logger):
                    logger.info(f"Matched CLOSE-MAG: {on_id} <-> {off_id} (diff={mag_diff:.0f}W)")
                    return off_event, "NON-M"  # Tag as NON-M for compatibility
                else:
                    logger.debug(f"Skipping {on_id} <-> {off_id}: negative residuals (close-mag fallback)")
                    continue

            # Check for NON-M match (stable power between events)
            event_range = (data['timestamp'] > on_end) & (data['timestamp'] < off_start)
            phase_data = data.loc[event_range, phase]

            if phase_data.empty:
                continue

            baseline_power = phase_data.iloc[0]
            max_deviation = (phase_data - baseline_power).max()
            min_deviation = (phase_data - baseline_power).min()

            if abs(max_deviation) <= max_magnitude_diff and abs(min_deviation) <= max_magnitude_diff:
                if is_valid_event_removal(data, on_event, off_event, logger):
                    logger.info(f"Matched NON-M: {on_id} <-> {off_id} (window={window_minutes}m)")
                    return off_event, "NON-M"
                else:
                    logger.debug(f"Skipping {on_id} <-> {off_id}: negative residuals")
                    continue

    logger.debug(f"No match found for {on_id}")
    return None, None
