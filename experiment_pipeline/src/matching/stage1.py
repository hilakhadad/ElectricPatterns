"""
Stage 1 matching: clean matching for events without noise between ON and OFF.

Finds matches where:
- Same phase
- OFF event occurs after ON event
- Power between ON and OFF is stable (no significant deviations)

Performance optimizations:
- Pre-filter candidates by magnitude similarity (skip mismatches early)
- Sort by magnitude similarity first to find best matches faster

Match tags combine magnitude quality and duration:
- Magnitude quality: EXACT (<50W), CLOSE (50-100W), APPROX (100-200W), LOOSE (200-350W)
- Duration: SPIKE (≤2min), QUICK (<5min), MEDIUM (5-30min), EXTENDED (>30min)
- Example: EXACT-SPIKE, CLOSE-MEDIUM, APPROX-EXTENDED-CORRECTED
"""
import pandas as pd
from .validator import is_valid_event_removal, build_match_tag

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
    - STABLE: Power is stable between ON and OFF
    - CLOSE-MAG: ON and OFF magnitudes match closely
    - CORRECTED: Match required magnitude correction

    Args:
        data: DataFrame with power data
        on_event: ON event dict
        off_events: DataFrame of potential OFF events
        max_time_diff: Maximum hours between ON and OFF
        max_magnitude_diff: Maximum watts deviation allowed between events
        logger: Logger instance

    Returns:
        Tuple of (matched_off_event, tag, correction) or (None, None, 0) if no match found
        - correction: Amount to reduce match magnitude by (0 if no correction needed)
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
    candidates_logged = False

    # Pre-filter by invariant conditions (phase, chronological order, magnitude range)
    # These don't change across window sizes, so compute once instead of 6x per ON event
    base_candidates = off_events[
        (off_events['phase'] == phase) &
        (off_events['start'] > on_end) &
        (abs(abs(off_events['magnitude']) - on_magnitude) <= MAX_MAGNITUDE_DIFF_FILTER)
    ]

    if base_candidates.empty:
        logger.info(f"[Stage 1] No match for {on_id}({on_magnitude:.0f}W)")
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

        # Log candidates summary on first window that has candidates
        if not candidates_logged:
            candidates_logged = True
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
            off_magnitude = abs(off_event['magnitude'])

            # Calculate duration for tagging
            duration_minutes = (off_end - on_start).total_seconds() / 60

            # Check for very short duration (spike) - skip stability check
            if duration_minutes <= 2:
                is_valid, correction = is_valid_event_removal(data, on_event, off_event, logger)
                if is_valid:
                    tail_ext = on_event.get('tail_extended', False) or off_event.get('tail_extended', False)
                    tag = build_match_tag(on_magnitude, off_magnitude, duration_minutes, is_corrected=correction > 0, is_tail_extended=tail_ext)
                    logger.info(f"Matched {tag}: {on_id} <-> {off_id}" + (f" (correction={correction:.0f}W)" if correction > 0 else ""))
                    return off_event, tag, correction
                else:
                    # Rejection reason already logged by validator
                    continue

            # Get magnitude difference for this candidate
            mag_diff = abs(on_magnitude - off_magnitude)

            # Small magnitude difference fallback: if magnitudes match very closely,
            # skip stability check and rely only on validator
            # This handles cases where other devices cause fluctuations during the event
            if mag_diff <= SMALL_MAGNITUDE_DIFF_THRESHOLD:
                is_valid, correction = is_valid_event_removal(data, on_event, off_event, logger)
                if is_valid:
                    tail_ext = on_event.get('tail_extended', False) or off_event.get('tail_extended', False)
                    tag = build_match_tag(on_magnitude, off_magnitude, duration_minutes, is_corrected=correction > 0, is_tail_extended=tail_ext)
                    logger.info(f"Matched {tag}: {on_id} <-> {off_id} (diff={mag_diff:.0f}W)" + (f" (correction={correction:.0f}W)" if correction > 0 else ""))
                    return off_event, tag, correction
                else:
                    # Rejection reason already logged by validator
                    continue

            # Check for stable power between events
            event_range = (data['timestamp'] > on_end) & (data['timestamp'] < off_start)
            phase_data = data.loc[event_range, phase]

            if phase_data.empty:
                continue

            baseline_power = phase_data.iloc[0]
            max_deviation = (phase_data - baseline_power).max()
            min_deviation = (phase_data - baseline_power).min()

            if abs(max_deviation) <= max_magnitude_diff and abs(min_deviation) <= max_magnitude_diff:
                is_valid, correction = is_valid_event_removal(data, on_event, off_event, logger)
                if is_valid:
                    tail_ext = on_event.get('tail_extended', False) or off_event.get('tail_extended', False)
                    tag = build_match_tag(on_magnitude, off_magnitude, duration_minutes, is_corrected=correction > 0, is_tail_extended=tail_ext)
                    logger.info(f"Matched {tag}: {on_id} <-> {off_id} (window={window_minutes}m)" + (f" (correction={correction:.0f}W)" if correction > 0 else ""))
                    return off_event, tag, correction
                else:
                    # Rejection reason already logged by validator
                    continue
            else:
                # Log stability rejection (power fluctuations between ON and OFF)
                # Include baseline and threshold for context
                logger.info(f"REJECTED {on_id}-{off_id}: unstable_power (max_dev={max_deviation:.0f}W, min_dev={min_deviation:.0f}W, threshold={max_magnitude_diff}W, baseline={baseline_power:.0f}W)")

    logger.info(f"[Stage 1] No match for {on_id}({on_magnitude:.0f}W)")
    return None, None, 0
