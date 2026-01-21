"""
Stack-based matching: LIFO approach for handling overlapping events.

When multiple devices turn on/off in sequence, this uses a stack to match
the most recent ON event with the current OFF event.
"""
import pandas as pd
from typing import List, Tuple
from .validator import is_valid_event_removal


def find_matches_stack_based(
    data: pd.DataFrame,
    on_events_list: List[dict],
    off_events_list: List[dict],
    phase: str,
    max_time_diff: int,
    max_magnitude_diff: int,
    logger
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Stack-based matching: process events chronologically using LIFO approach.

    When an OFF event arrives, try to match it with the most recent ON event (LIFO)
    that has similar magnitude. This handles overlapping events better.

    Args:
        data: DataFrame with power data
        on_events_list: List of ON event dicts
        off_events_list: List of OFF event dicts
        phase: Phase name (w1, w2, w3)
        max_time_diff: Max hours between ON and OFF
        max_magnitude_diff: Max watts difference for deviation check
        logger: Logger instance

    Returns:
        Tuple of (matches, unmatched_on, unmatched_off)
    """
    # Combine and sort all events chronologically
    all_events = []
    for e in on_events_list:
        all_events.append({**e, 'type': 'on', 'sort_time': e['end']})
    for e in off_events_list:
        all_events.append({**e, 'type': 'off', 'sort_time': e['start']})

    all_events.sort(key=lambda x: x['sort_time'])

    # Stack of active ON events (LIFO)
    on_stack = []
    matches = []
    unmatched_off = []

    for event in all_events:
        if event['type'] == 'on':
            # Push ON event to stack
            on_stack.append(event)
            logger.info(f"[Stack] Pushed {event['event_id']} to stack (magnitude={event['magnitude']}W). Stack size: {len(on_stack)}")

        else:  # OFF event
            off_event = event
            off_id = off_event['event_id']
            off_magnitude = abs(off_event['magnitude'])
            off_start = off_event['start']

            logger.info(f"[Stack] Processing OFF {off_id} (magnitude={off_magnitude}W). Stack size: {len(on_stack)}")

            # Try to find matching ON from stack (LIFO order - check from end)
            matched = False
            for i in range(len(on_stack) - 1, -1, -1):
                on_event = on_stack[i]
                on_id = on_event['event_id']
                on_magnitude = abs(on_event['magnitude'])
                on_end = on_event['end']

                # Check time constraint
                time_diff_hours = (off_start - on_end).total_seconds() / 3600
                if time_diff_hours > max_time_diff or time_diff_hours < 0:
                    continue

                # Check magnitude similarity (±500W for stack matching)
                mag_diff = abs(on_magnitude - off_magnitude)
                if mag_diff > 500:
                    logger.info(f"[Stack] {on_id} magnitude mismatch with {off_id}: {on_magnitude}W vs {off_magnitude}W (diff={mag_diff}W)")
                    continue

                # Validate that removal won't create negative values
                if not is_valid_event_removal(data, on_event, off_event, logger):
                    logger.info(f"[Stack] {on_id}-{off_id} failed validation, trying next in stack")
                    continue

                # Determine tag (SPIKE or NON-M)
                duration_minutes = (off_event['end'] - on_event['start']).total_seconds() / 60
                if duration_minutes <= 2:
                    tag = "SPIKE"
                else:
                    tag = "NON-M"

                # Match found!
                matches.append({
                    'on_event_id': on_id,
                    'off_event_id': off_id,
                    'on_start': on_event['start'],
                    'on_end': on_event['end'],
                    'off_start': off_event['start'],
                    'off_end': off_event['end'],
                    'duration': duration_minutes,
                    'on_magnitude': on_event['magnitude'],
                    'off_magnitude': off_event['magnitude'],
                    'tag': tag,
                    'phase': phase
                })

                logger.info(f"[Stack] Matched {on_id} with {off_id} as {tag} (mag diff={mag_diff}W)")

                # Remove matched ON from stack
                on_stack.pop(i)
                matched = True
                break

            if not matched:
                logger.info(f"[Stack] No match found for OFF {off_id}, adding to unmatched")
                unmatched_off.append(off_event)

    # Remaining ON events in stack are unmatched
    unmatched_on = on_stack
    for on_event in unmatched_on:
        logger.info(f"[Stack] Unmatched ON: {on_event['event_id']}")

    logger.info(f"[Stack] Phase {phase} complete: {len(matches)} matches, {len(unmatched_on)} unmatched ON, {len(unmatched_off)} unmatched OFF")

    return matches, unmatched_on, unmatched_off
