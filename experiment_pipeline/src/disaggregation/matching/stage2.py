"""
Stage 2 matching: noisy matching for events with other devices active between ON and OFF.

Finds matches where:
- Same phase
- Similar magnitudes between ON and OFF
- Power never drops significantly below baseline (device stays on)

Performance optimizations:
- Pre-filter candidates by magnitude similarity in initial query
- Sort by magnitude similarity first to find best matches faster

Match tags combine NOISY prefix with magnitude quality and duration:
- Format: NOISY-{magnitude_quality}-{duration}[-CORRECTED]
- Example: NOISY-EXACT-MEDIUM, NOISY-CLOSE-EXTENDED-CORRECTED
"""
import pandas as pd
from .validator import is_valid_event_removal, build_match_tag


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
        Tuple of (matched_off_event, tag, correction) or (None, None, 0) if no match found
        - correction: Amount to reduce match magnitude by (0 if no correction needed)
    """
    phase = on_event['phase']
    on_id = on_event['event_id']
    on_end = on_event['end']
    on_magnitude = abs(on_event['magnitude'])

    logger.debug(f"[Stage 2] Processing {on_id}, Phase={phase}, Magnitude={on_magnitude}W")

    # Progressive window search: start small and expand
    # Windows in minutes: 15min, 30min, 1hr, 2hr, 4hr, then max_time_diff
    window_steps_minutes = [15, 30, 60, 120, 240, max_time_diff * 60]
    candidates_logged = False

    # Pre-filter by invariant conditions (phase, chronological order, magnitude range)
    # These don't change across window sizes, so compute once instead of 6x per ON event
    base_candidates = off_events[
        (off_events['phase'] == phase) &
        (off_events['start'] > on_end) &
        (abs(abs(off_events['magnitude']) - on_magnitude) <= max_magnitude_diff)
    ]

    if base_candidates.empty:
        return None, None, 0

    # Pre-compute derived columns once on the base set
    base_candidates = base_candidates.assign(
        magnitude_diff=abs(abs(base_candidates['magnitude']) - on_magnitude),
        time_diff=(base_candidates['start'] - on_end)
    ).sort_values(by=['magnitude_diff', 'time_diff'])

    for window_minutes in window_steps_minutes:
        if window_minutes > max_time_diff * 60:
            break

        # Only filter by time window — phase/magnitude/chronological already applied
        candidates = base_candidates[
            base_candidates['time_diff'] <= pd.Timedelta(minutes=window_minutes)
        ]

        if candidates.empty:
            continue

        # Log candidates on first window that has them
        if not candidates_logged:
            candidates_logged = True
            summary = ", ".join([
                f"{row['event_id']}({abs(row['magnitude']):.0f}W, +{row['time_diff'].total_seconds()/60:.0f}m)"
                for _, row in candidates.head(5).iterrows()
            ])
            if len(candidates) > 5:
                summary += f", ... +{len(candidates)-5} more"
            logger.info(f"[Stage 2] {on_id}({on_magnitude:.0f}W) candidates[{window_minutes}m]: {summary}")

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
                drop_amount = min_allowed - min_power
                logger.info(f"REJECTED {on_id}-{off_id}: power_drop (min={min_power:.0f}W, allowed={min_allowed:.0f}W, drop={drop_amount:.0f}W, baseline={baseline_power:.0f}W)")
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

            is_valid, correction = is_valid_event_removal(data, on_event_dict, off_event_dict, logger)
            if not is_valid:
                # Rejection reason already logged by validator
                continue

            # Calculate duration for tagging
            off_magnitude = abs(off_event['magnitude'])
            duration_minutes = (off_event['end'] - on_event['start']).total_seconds() / 60
            tail_ext = on_event.get('tail_extended', False) or off_event.get('tail_extended', False)
            tag = build_match_tag(on_magnitude, off_magnitude, duration_minutes, is_noisy=True, is_corrected=correction > 0, is_tail_extended=tail_ext)
            logger.info(f"Matched {tag}: {on_id} <-> {off_id} (window={window_minutes}m)" + (f" (correction={correction:.0f}W)" if correction > 0 else ""))
            return off_event, tag, correction

    return None, None, 0
