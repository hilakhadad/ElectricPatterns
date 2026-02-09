"""
Stage 3 matching: partial matching for events with significant magnitude difference.

When ON and OFF magnitudes differ by more than threshold (350W), instead of rejecting:
- Match using the smaller magnitude
- Create a remainder event for the larger one

This handles cases where two devices turn on at the same time but only one turns off,
or vice versa.

Match tags combine PARTIAL prefix with duration:
- Format: PARTIAL-{duration}[-CORRECTED]
- Example: PARTIAL-MEDIUM, PARTIAL-EXTENDED-CORRECTED
- Note: No magnitude quality tag since magnitudes differ significantly by design
"""
import pandas as pd
from .validator import is_valid_partial_removal, build_match_tag


def find_partial_match(data: pd.DataFrame, on_event: dict, off_events: pd.DataFrame,
                       max_time_diff: int, max_magnitude_diff: int, logger):
    """
    Stage 3 matcher: finds partial matches when ON and OFF magnitudes differ significantly.

    Uses progressive window search: starts with small time window and expands
    gradually to find the closest match first.

    Criteria:
    1. Same phase
    2. OFF within time window (progressive: 15min -> 30min -> 1hr -> 2hr -> 4hr -> max)
    3. Magnitude difference > max_magnitude_diff (350W) - THIS IS THE TRIGGER
    4. Stable power between ON and OFF (like STABLE)
    5. Removing smaller magnitude doesn't create negative values

    These matches get tag "PARTIAL" and create a remainder event.

    Args:
        data: DataFrame with power data
        on_event: Unmatched ON event dict
        off_events: DataFrame of remaining unmatched OFF events
        max_time_diff: Maximum hours between ON and OFF
        max_magnitude_diff: Minimum watts difference to trigger partial matching
        logger: Logger instance

    Returns:
        Tuple of (matched_off_event, tag, remainder_event, correction) or (None, None, None, 0)
        - correction: Amount to reduce match magnitude by (0 if no correction needed)
    """
    phase = on_event['phase']
    on_id = on_event['event_id']
    on_start = on_event['start']
    on_end = on_event['end']
    on_magnitude = abs(on_event['magnitude'])
    on_duration = on_event.get('duration', 0)

    logger.debug(f"[Stage 3] Processing {on_id}, Phase={phase}, Magnitude={on_magnitude}W")

    # Progressive window search: start small and expand
    window_steps_minutes = [15, 30, 60, 120, 240, max_time_diff * 60]
    candidates_logged = False

    for window_minutes in window_steps_minutes:
        if window_minutes > max_time_diff * 60:
            break

        # Pre-filter by phase, time, AND magnitude difference > threshold
        # This is the OPPOSITE of Stage 1/2 - we WANT large magnitude differences
        candidates = off_events[
            (off_events['phase'] == phase) &
            (off_events['start'] > on_end) &
            (off_events['start'] - on_end <= pd.Timedelta(minutes=window_minutes)) &
            (abs(abs(off_events['magnitude']) - on_magnitude) > max_magnitude_diff)
        ]

        if candidates.empty:
            continue

        # Use .assign() to add columns without explicit copy
        candidates = candidates.assign(
            magnitude_diff=abs(abs(candidates['magnitude']) - on_magnitude),
            time_diff=(candidates['start'] - on_end).abs()
        ).sort_values(by=['time_diff', 'magnitude_diff'])

        # Log candidates on first window that has them
        if not candidates_logged:
            candidates_logged = True
            summary = ", ".join([
                f"{row['event_id']}({abs(row['magnitude']):.0f}W, +{row['time_diff'].total_seconds()/60:.0f}m)"
                for _, row in candidates.head(5).iterrows()
            ])
            if len(candidates) > 5:
                summary += f", ... +{len(candidates)-5} more"
            logger.info(f"[Stage 3] {on_id}({on_magnitude:.0f}W) candidates[{window_minutes}m]: {summary}")

        for _, off_event in candidates.iterrows():
            off_start = off_event['start']
            off_end = off_event['end']
            off_id = off_event['event_id']
            off_magnitude = abs(off_event['magnitude'])
            off_duration = off_event.get('duration', 0)

            # Get data between ON and OFF
            event_range = (data['timestamp'] > on_end) & (data['timestamp'] < off_start)
            phase_data = data.loc[event_range, phase]

            # If events are adjacent (no data between them), that's OK for partial matching
            # We skip the stability check in this case
            if not phase_data.empty:
                # Check that power is stable between events (like NON-M)
                baseline_power = phase_data.iloc[0]
                max_deviation = (phase_data - baseline_power).abs().max()

                if max_deviation > max_magnitude_diff:
                    logger.info(f"REJECTED {on_id}-{off_id}: partial_unstable_power (dev={max_deviation:.0f}W, threshold={max_magnitude_diff}W, baseline={baseline_power:.0f}W)")
                    continue

            # Calculate match magnitude and remainder
            match_magnitude = min(on_magnitude, off_magnitude)
            remainder_magnitude = abs(on_magnitude - off_magnitude)

            # CONSTRAINT: ON and OFF must differ by MORE than 150% of each other
            # max(ON, OFF) / min(ON, OFF) > 1.5
            # This ensures Stage 3 only handles cases with significant magnitude difference
            # Example: ON=2400W, OFF=1600W → 2400/1600=1.5 → rejected (too similar, Stage 1/2 should handle)
            # Example: ON=3000W, OFF=1600W → 3000/1600=1.875 → OK (significant difference)
            min_ratio = 1.50  # 150%
            actual_ratio = max(on_magnitude, off_magnitude) / match_magnitude if match_magnitude > 0 else float('inf')
            if actual_ratio <= min_ratio:
                logger.info(f"REJECTED {on_id}-{off_id}: ratio_too_small ({actual_ratio:.2f} <= {min_ratio:.2f}, ON={on_magnitude:.0f}W OFF={off_magnitude:.0f}W)")
                continue

            # Validate that removal with match_magnitude won't create negative values
            is_valid, correction = is_valid_partial_removal(data, on_event, off_event, match_magnitude, logger)
            if not is_valid:
                # Rejection reason already logged by validator
                continue

            # Calculate duration for tagging
            duration_minutes = (off_end - on_start).total_seconds() / 60
            tag = build_match_tag(on_magnitude, off_magnitude, duration_minutes, is_partial=True, is_corrected=correction > 0)

            # Create remainder event
            if on_magnitude > off_magnitude:
                # ON was bigger - remainder is new ON event
                remainder = {
                    'start': on_start,
                    'end': on_end,
                    'magnitude': remainder_magnitude,
                    'phase': phase,
                    'event_id': f"{on_id}_remainder",
                    'event': 'on',
                    'duration': on_duration
                }
                logger.info(f"Matched {tag}: {on_id}({on_magnitude}W) <-> {off_id}({off_magnitude}W), "
                           f"match={match_magnitude}W, remainder ON={remainder_magnitude}W" + (f" (correction={correction:.0f}W)" if correction > 0 else ""))
            else:
                # OFF was bigger - remainder is new OFF event
                remainder = {
                    'start': off_start,
                    'end': off_end,
                    'magnitude': -remainder_magnitude,  # OFF events are negative
                    'phase': phase,
                    'event_id': f"{off_id}_remainder",
                    'event': 'off',
                    'duration': off_duration
                }
                logger.info(f"Matched {tag}: {on_id}({on_magnitude}W) <-> {off_id}({off_magnitude}W), "
                           f"match={match_magnitude}W, remainder OFF={remainder_magnitude}W" + (f" (correction={correction:.0f}W)" if correction > 0 else ""))

            return off_event, tag, remainder, correction

    return None, None, None, 0
