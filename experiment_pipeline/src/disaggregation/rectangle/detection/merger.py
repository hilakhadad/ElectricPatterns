"""
Event merging logic.

Merges overlapping or touching events into single events.
"""
import pandas as pd


def merge_overlapping_events(events_df: pd.DataFrame, max_gap_minutes: int = 0,
                             data: pd.DataFrame = None, phase: str = None,
                             logger=None) -> pd.DataFrame:
    """
    Merge events that overlap or touch each other.

    Only merges if:
    - Events overlap (gap < 0)
    - Events touch (next starts exactly when current ends, gap = 0)

    Does NOT merge events with a gap between them
    (e.g., one ends at 22:05, next starts at 22:06).

    Magnitude is accumulated from the detected events (summed), preserving
    the detection results without recalculating from raw data.

    Args:
        events_df: DataFrame with 'start', 'end', 'magnitude' columns
        max_gap_minutes: Maximum gap between events to still merge them
                        (default 0 = only overlapping/touching)
        data: Unused, kept for backward compatibility
        phase: Unused, kept for backward compatibility

    Returns:
        DataFrame with merged events
    """
    if len(events_df) <= 1:
        return events_df

    original_count = len(events_df)

    # Sort by start time
    df = events_df.sort_values('start').reset_index(drop=True)

    merged = []
    current = df.iloc[0].copy()

    for i in range(1, len(df)):
        next_event = df.iloc[i]

        # Check if events overlap or touch
        # gap < 0: overlap, gap = 0: touch, gap > 0: separate events
        gap = (next_event['start'] - current['end']).total_seconds() / 60

        if gap <= max_gap_minutes:
            # Merge: extend end time, sum magnitudes
            current['end'] = max(current['end'], next_event['end'])
            current['magnitude'] = current['magnitude'] + next_event['magnitude']
        else:
            merged.append(current)
            current = next_event.copy()

    merged.append(current)

    result = pd.DataFrame(merged).reset_index(drop=True)
    if logger and original_count != len(result):
        logger.debug(f"Merge overlapping: {original_count} -> {len(result)} events")
    return result


def merge_split_off_events(off_events: pd.DataFrame, on_events: pd.DataFrame,
                           max_gap_minutes: int = 2,
                           data: pd.DataFrame = None, phase: str = None,
                           logger=None) -> pd.DataFrame:
    """
    Merge OFF events that represent a split device shutdown.

    More relaxed than merge_consecutive_off_events — does NOT require both events
    to be instantaneous. Handles cases where a device shuts down in two steps
    due to measurement error (e.g., boiler 2500W → 1200W → 0W).

    Criteria:
    1. Same phase (implicit — called per-phase)
    2. Gap between first OFF end and second OFF start <= max_gap_minutes
    3. No ON event between them
    4. Power between them is elevated (device still running):
       power_between >= |OFF_2.magnitude| * 0.5

    Args:
        off_events: DataFrame with OFF events ('start', 'end', 'magnitude')
        on_events: DataFrame with ON events (to check for intervening ONs)
        max_gap_minutes: Maximum gap between split OFF events (default 2)
        data: DataFrame with timestamp as index (for power lookup and magnitude recalc)
        phase: Phase column name (for magnitude recalculation from actual values)

    Returns:
        DataFrame with merged OFF events
    """
    if len(off_events) <= 1:
        return off_events

    original_count = len(off_events)

    # Sort by start time
    df = off_events.sort_values('start').reset_index(drop=True)

    # Get ON event times for intervening check
    on_starts = set()
    if on_events is not None and len(on_events) > 0:
        on_starts = set(on_events['start'])

    merged = []
    current = df.iloc[0].copy()

    for i in range(1, len(df)):
        next_event = df.iloc[i]

        gap_minutes = (next_event['start'] - current['end']).total_seconds() / 60

        # Check if any ON event occurs between current end and next start
        has_on_between = False
        if on_starts:
            for on_start in on_starts:
                if current['end'] < on_start < next_event['start']:
                    has_on_between = True
                    break

        # Check power between the two OFFs (device should still be running)
        power_ok = False
        if data is not None and phase is not None and gap_minutes >= 0:
            try:
                between_start = current['end']
                between_end = next_event['start']
                between_data = data.loc[between_start:between_end, phase]
                if len(between_data) > 0:
                    power_between = between_data.mean()
                    # Device should still be running: power between drops should be
                    # at least 50% of the second drop's magnitude
                    power_ok = power_between >= abs(next_event['magnitude']) * 0.5
                else:
                    # Adjacent events (no data between) — OK to merge
                    power_ok = True
            except (KeyError, TypeError):
                power_ok = False

        # Merge if: close gap, no ON between, power still elevated
        if gap_minutes <= max_gap_minutes and not has_on_between and power_ok:
            # Merge: extend boundaries, recalculate magnitude from actual values
            new_end = max(current['end'], next_event['end'])

            if data is not None and phase is not None:
                # Recalculate magnitude from phase values
                before_start = current['start'] - pd.Timedelta(minutes=1)
                try:
                    value_before = data.loc[before_start, phase]
                    value_end = data.loc[new_end, phase]
                    new_magnitude = value_end - value_before
                except (KeyError, TypeError):
                    new_magnitude = current['magnitude'] + next_event['magnitude']
            else:
                new_magnitude = current['magnitude'] + next_event['magnitude']

            current['end'] = new_end
            current['magnitude'] = new_magnitude
        else:
            merged.append(current)
            current = next_event.copy()

    merged.append(current)

    result = pd.DataFrame(merged).reset_index(drop=True)
    if logger and original_count != len(result):
        logger.debug(f"Merge split-OFF: {original_count} -> {len(result)} events")
    return result


def merge_consecutive_on_events(on_events: pd.DataFrame, off_events: pd.DataFrame,
                                 max_gap_minutes: int = 2,
                                 data: pd.DataFrame = None, phase: str = None,
                                 logger=None) -> pd.DataFrame:
    """
    Merge consecutive ON events that likely represent the same appliance activation.

    Two ON events are merged if:
    1. Both are instantaneous (duration = 0)
    2. Gap between them is <= 1 minute
    3. No OFF event occurs between them

    This handles cases where an appliance turns on in stages (e.g., AC compressor)
    creating multiple threshold crossings over a few minutes. Magnitudes are summed
    from the detected events, preserving the original detection results.

    Args:
        on_events: DataFrame with ON events ('start', 'end', 'magnitude')
        off_events: DataFrame with OFF events (to check for intervening OFFs)
        max_gap_minutes: Maximum gap between ON events to consider merging
        data: Unused, kept for backward compatibility
        phase: Unused, kept for backward compatibility
        logger: Optional logger for debug messages

    Returns:
        DataFrame with merged ON events
    """
    return _merge_consecutive_events(
        on_events, off_events, max_gap_minutes, data, phase, logger=logger
    )


def merge_consecutive_off_events(off_events: pd.DataFrame, on_events: pd.DataFrame,
                                  max_gap_minutes: int = 2,
                                  data: pd.DataFrame = None, phase: str = None,
                                  logger=None) -> pd.DataFrame:
    """
    Merge consecutive OFF events that likely represent the same appliance deactivation.

    Two OFF events are merged if:
    1. Both are instantaneous (duration = 0)
    2. Gap between them is <= 1 minute
    3. No ON event occurs between them

    This handles cases where an appliance turns off in stages. Magnitudes are summed
    from the detected events, preserving the original detection results.

    Args:
        off_events: DataFrame with OFF events ('start', 'end', 'magnitude')
        on_events: DataFrame with ON events (to check for intervening ONs)
        max_gap_minutes: Maximum gap between OFF events to consider merging
        data: Unused, kept for backward compatibility
        phase: Unused, kept for backward compatibility
        logger: Optional logger for debug messages

    Returns:
        DataFrame with merged OFF events
    """
    return _merge_consecutive_events(
        off_events, on_events, max_gap_minutes, data, phase, logger=logger
    )


def _merge_consecutive_events(events: pd.DataFrame, opposite_events: pd.DataFrame,
                               max_gap_minutes: int = 2,
                               data: pd.DataFrame = None, phase: str = None,
                               logger=None) -> pd.DataFrame:
    """
    Internal function to merge consecutive events of the same type.

    Only merges events if:
    1. Both events are instantaneous (duration = 0, i.e., start == end)
    2. Events are truly consecutive (gap <= 1 minute, one ends when next starts)
    3. No opposite event occurs between them

    This is very restrictive to avoid merging events from different appliances.
    Only merges when it's clearly the same appliance turning on/off across
    consecutive minutes (e.g., threshold crossed at 23:55 and again at 23:56).

    Magnitude is accumulated from the detected events (summed), preserving
    the detection results without recalculating from raw data.

    Args:
        events: DataFrame with events to merge ('start', 'end', 'magnitude')
        opposite_events: DataFrame with opposite events (to check for intervening events)
        max_gap_minutes: Maximum gap between events to consider merging (default 2, but
                        actual merge only happens if both events have duration 0)
        data: Unused, kept for backward compatibility
        phase: Unused, kept for backward compatibility

    Returns:
        DataFrame with merged events
    """
    if len(events) <= 1:
        return events

    original_count = len(events)

    # Sort by start time
    df = events.sort_values('start').reset_index(drop=True)

    # Get opposite event starts for quick lookup
    opposite_starts = set()
    if opposite_events is not None and len(opposite_events) > 0:
        opposite_starts = set(opposite_events['start'])

    merged = []
    current = df.iloc[0].copy()

    for i in range(1, len(df)):
        next_event = df.iloc[i]

        # Calculate gap between current end and next start
        gap_minutes = (next_event['start'] - current['end']).total_seconds() / 60

        # Check if any opposite event occurs between current end and next start
        has_opposite_between = False
        if opposite_starts:
            for opp_start in opposite_starts:
                if current['end'] < opp_start < next_event['start']:
                    has_opposite_between = True
                    break

        # Check if both events are instantaneous (duration = 0)
        # This means they're single-minute threshold crossings, not extended events
        current_duration = (current['end'] - current['start']).total_seconds() / 60
        next_duration = (next_event['end'] - next_event['start']).total_seconds() / 60
        both_instantaneous = (current_duration == 0) and (next_duration == 0)

        # Merge ONLY if:
        # 1. Both events are instantaneous (duration 0)
        # 2. Gap is at most 1 minute (truly consecutive)
        # 3. No opposite event between them
        if both_instantaneous and gap_minutes <= 1 and not has_opposite_between:
            # Merge: extend end time, sum magnitudes
            current['end'] = next_event['end']
            current['magnitude'] = current['magnitude'] + next_event['magnitude']
        else:
            merged.append(current)
            current = next_event.copy()

    merged.append(current)

    result = pd.DataFrame(merged).reset_index(drop=True)
    if logger and original_count != len(result):
        logger.debug(f"Merge consecutive: {original_count} -> {len(result)} events")
    return result
