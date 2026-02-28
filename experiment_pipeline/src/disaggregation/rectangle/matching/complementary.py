"""
Stage 4 matching: complementary OFF matching.

When an ON event can't be matched by Stages 1-3 because the device shuts down
in two steps (e.g., AC compressor: 1500W ON, then -800W OFF followed by -700W OFF),
this stage finds two consecutive OFF events that together match the ON magnitude.

Key safety constraint: OFF events are merged ONLY when a matching ON exists.
This prevents false merges in unrelated event sequences.

Match tags: COMP-OFF-{MAGNITUDE_QUALITY}-{DURATION}[-CORRECTED]
Example: COMP-OFF-EXACT-EXTENDED, COMP-OFF-CLOSE-MEDIUM
"""
import pandas as pd
from .validator import is_valid_event_removal, build_match_tag
from .match_tags import get_magnitude_quality_tag, get_duration_tag

# Maximum gap between two OFF events to consider them complementary
DEFAULT_MAX_GAP_MINUTES = 10

# Minimum gap — skip pairs already handled by split-OFF merger in detection
DEFAULT_MIN_GAP_MINUTES = 2

# Maximum difference between ON magnitude and combined OFF magnitude
DEFAULT_MAX_COMBINED_DIFF = 350  # same as MAX_MAGNITUDE_DIFF_FILTER


def find_complementary_off_matches(data: pd.DataFrame, unmatched_on: list,
                                   unmatched_off: list,
                                   max_gap_minutes: int = DEFAULT_MAX_GAP_MINUTES,
                                   min_gap_minutes: int = DEFAULT_MIN_GAP_MINUTES,
                                   max_combined_diff: int = DEFAULT_MAX_COMBINED_DIFF,
                                   logger=None):
    """
    Find ON events that match two consecutive OFF events (complementary OFF).

    For each unmatched ON event, searches for two consecutive OFF events on the
    same phase where:
    1. Both OFFs occur after the ON
    2. The gap between OFF₁.end and OFF₂.start <= max_gap_minutes
    3. No ON event occurs between them
    4. Combined OFF magnitude matches ON magnitude within max_combined_diff
    5. Validator passes (CV, min_power, no negative remaining)

    Args:
        data: DataFrame with power data
        unmatched_on: List of unmatched ON event dicts
        unmatched_off: List of unmatched OFF event dicts
        max_gap_minutes: Maximum gap between two OFF events
        max_combined_diff: Maximum |ON_mag - (OFF₁_mag + OFF₂_mag)|
        logger: Logger instance

    Returns:
        Tuple of (new_matches, remaining_unmatched_on, remaining_unmatched_off)
    """
    if not unmatched_on or len(unmatched_off) < 2:
        return [], unmatched_on, unmatched_off

    new_matches = []
    used_on_ids = set()
    used_off_ids = set()

    # Build ON starts set for intervening-ON check
    on_starts = set()
    for ev in unmatched_on:
        on_starts.add(ev['start'])

    # Group OFF events by phase and sort by start time
    off_by_phase = {}
    for off_ev in unmatched_off:
        phase = off_ev['phase']
        if phase not in off_by_phase:
            off_by_phase[phase] = []
        off_by_phase[phase].append(off_ev)

    for phase in off_by_phase:
        off_by_phase[phase].sort(key=lambda x: x['start'])

    # Process each unmatched ON event
    for on_event in unmatched_on:
        if on_event['event_id'] in used_on_ids:
            continue

        phase = on_event['phase']
        on_magnitude = abs(on_event['magnitude'])
        on_end = on_event['end']

        if phase not in off_by_phase:
            continue

        # Get OFF candidates: same phase, after ON
        off_candidates = [
            off_ev for off_ev in off_by_phase[phase]
            if off_ev['start'] > on_end
            and off_ev['event_id'] not in used_off_ids
        ]

        if len(off_candidates) < 2:
            continue

        # Try consecutive pairs
        best_match = None
        best_diff = float('inf')

        for i in range(len(off_candidates) - 1):
            off1 = off_candidates[i]
            off2 = off_candidates[i + 1]

            # Check gap between the two OFFs
            gap_minutes = (off2['start'] - off1['end']).total_seconds() / 60
            if gap_minutes > max_gap_minutes:
                continue
            if gap_minutes <= min_gap_minutes:
                # Skip small gaps — already handled by split-OFF merger in detection
                continue

            # Check no ON event between them
            has_on_between = False
            for os in on_starts:
                if off1['end'] < os < off2['start']:
                    has_on_between = True
                    break
            if has_on_between:
                continue

            # Check combined magnitude
            combined_mag = abs(off1['magnitude']) + abs(off2['magnitude'])
            mag_diff = abs(on_magnitude - combined_mag)

            if mag_diff > max_combined_diff:
                continue

            # Keep best (closest magnitude match)
            if mag_diff < best_diff:
                best_diff = mag_diff
                best_match = (off1, off2, combined_mag)

        if best_match is None:
            continue

        off1, off2, combined_mag = best_match

        # Build a synthetic combined OFF event for validation
        combined_off = {
            'event_id': f"{off1['event_id']}+{off2['event_id']}",
            'start': off1['start'],
            'end': off2['end'],
            'magnitude': -(combined_mag),  # OFF events are negative
            'phase': phase,
            'event': 'off',
        }

        # Validate using the standard validator
        is_valid, correction = is_valid_event_removal(
            data, on_event, combined_off, logger,
            negative_tolerance=-10
        )

        if not is_valid:
            if logger:
                logger.debug(
                    f"[Comp-OFF] Rejected {on_event['event_id']}: "
                    f"ON={on_magnitude:.0f}W, OFF₁={abs(off1['magnitude']):.0f}W + "
                    f"OFF₂={abs(off2['magnitude']):.0f}W = {combined_mag:.0f}W (validator failed)"
                )
            continue

        # Build match record
        duration = (off2['end'] - on_event['start']).total_seconds() / 60

        # Apply correction
        match_on_mag = on_event['magnitude']
        match_off_mag = combined_off['magnitude']
        if correction > 0:
            match_on_mag = abs(match_on_mag) - correction
            match_off_mag = -(combined_mag - correction)

        # Build tag
        mag_tag = get_magnitude_quality_tag(on_magnitude, combined_mag)
        dur_tag = get_duration_tag(duration)
        tag_parts = ["COMP-OFF", mag_tag, dur_tag]
        if correction > 0:
            tag_parts.append("CORRECTED")
        tag = "-".join(tag_parts)

        match_record = {
            'on_event_id': on_event['event_id'],
            'off_event_id': combined_off['event_id'],
            'on_start': on_event['start'],
            'on_end': on_event['end'],
            'off_start': off1['start'],
            'off_end': off2['end'],
            'duration': duration,
            'on_magnitude': match_on_mag,
            'off_magnitude': match_off_mag,
            'correction': correction,
            'tag': tag,
            'phase': phase,
        }

        new_matches.append(match_record)
        used_on_ids.add(on_event['event_id'])
        used_off_ids.add(off1['event_id'])
        used_off_ids.add(off2['event_id'])

        if logger:
            logger.info(
                f"[Comp-OFF] {on_event['event_id']}: "
                f"ON={on_magnitude:.0f}W matched OFF₁={abs(off1['magnitude']):.0f}W + "
                f"OFF₂={abs(off2['magnitude']):.0f}W = {combined_mag:.0f}W "
                f"(gap={((off2['start'] - off1['end']).total_seconds() / 60):.1f}min, tag={tag})"
            )

    # Build remaining unmatched lists
    remaining_on = [ev for ev in unmatched_on if ev['event_id'] not in used_on_ids]
    remaining_off = [ev for ev in unmatched_off if ev['event_id'] not in used_off_ids]

    if logger and new_matches:
        logger.info(f"[Comp-OFF] Found {len(new_matches)} complementary OFF matches")

    return new_matches, remaining_on, remaining_off
