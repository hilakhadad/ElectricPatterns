"""
Event merging logic.

Merges overlapping or touching events into single events.
"""
import pandas as pd


def merge_overlapping_events(events_df: pd.DataFrame, max_gap_minutes: int = 0,
                             data: pd.DataFrame = None, phase: str = None) -> pd.DataFrame:
    """
    Merge events that overlap or touch each other.

    Only merges if:
    - Events overlap (gap < 0)
    - Events touch (next starts exactly when current ends, gap = 0)

    Does NOT merge events with a gap between them
    (e.g., one ends at 22:05, next starts at 22:06).

    Args:
        events_df: DataFrame with 'start', 'end', 'magnitude' columns
        max_gap_minutes: Maximum gap between events to still merge them
                        (default 0 = only overlapping/touching)
        data: Optional DataFrame with power data for recalculating magnitude
        phase: Optional phase column name for magnitude calculation

    Returns:
        DataFrame with merged events
    """
    if len(events_df) <= 1:
        return events_df

    # Sort by start time
    df = events_df.sort_values('start').reset_index(drop=True)

    merged = []
    current = df.iloc[0].copy()
    was_merged = False

    for i in range(1, len(df)):
        next_event = df.iloc[i]

        # Check if events overlap or touch
        # gap < 0: overlap, gap = 0: touch, gap > 0: separate events
        gap = (next_event['start'] - current['end']).total_seconds() / 60

        if gap <= max_gap_minutes:
            # Merge: extend end time
            current['end'] = max(current['end'], next_event['end'])
            was_merged = True
        else:
            # No overlap - recalculate magnitude if merged, then save
            if was_merged and data is not None and phase is not None:
                current['magnitude'] = _calc_magnitude(data, phase, current['start'], current['end'])
            merged.append(current)
            current = next_event.copy()
            was_merged = False

    # Don't forget the last event
    if was_merged and data is not None and phase is not None:
        current['magnitude'] = _calc_magnitude(data, phase, current['start'], current['end'])
    merged.append(current)

    return pd.DataFrame(merged).reset_index(drop=True)


def _calc_magnitude(data: pd.DataFrame, phase: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Calculate magnitude as value_at_end - value_before_start."""
    # Use .loc with index if timestamp is the index, otherwise use efficient lookup
    if data.index.name == 'timestamp':
        try:
            value_end = data.loc[end, phase]
        except KeyError:
            value_end = 0
        before_start = start - pd.Timedelta(minutes=1)
        try:
            value_before = data.loc[before_start, phase]
        except KeyError:
            value_before = 0
    else:
        # Fallback to boolean indexing
        end_mask = data['timestamp'] == end
        value_end = data.loc[end_mask, phase].values[0] if end_mask.any() else 0

        before_start = start - pd.Timedelta(minutes=1)
        before_mask = data['timestamp'] == before_start
        value_before = data.loc[before_mask, phase].values[0] if before_mask.any() else 0

    return value_end - value_before


def merge_consecutive_on_events(on_events: pd.DataFrame, off_events: pd.DataFrame,
                                 max_gap_minutes: int = 2,
                                 data: pd.DataFrame = None, phase: str = None) -> pd.DataFrame:
    """
    Merge consecutive ON events that likely represent the same appliance activation.

    Two ON events are merged if:
    1. Gap between them is <= max_gap_minutes (default 2)
    2. No OFF event occurs between them

    This handles cases where an appliance turns on in stages (e.g., AC compressor)
    creating multiple threshold crossings over a few minutes.

    Args:
        on_events: DataFrame with ON events ('start', 'end', 'magnitude')
        off_events: DataFrame with OFF events (to check for intervening OFFs)
        max_gap_minutes: Maximum gap between ON events to consider merging
        data: Power data DataFrame (indexed by timestamp) for magnitude recalculation
        phase: Phase column name for magnitude calculation

    Returns:
        DataFrame with merged ON events
    """
    return _merge_consecutive_events(
        on_events, off_events, max_gap_minutes, data, phase
    )


def merge_consecutive_off_events(off_events: pd.DataFrame, on_events: pd.DataFrame,
                                  max_gap_minutes: int = 2,
                                  data: pd.DataFrame = None, phase: str = None) -> pd.DataFrame:
    """
    Merge consecutive OFF events that likely represent the same appliance deactivation.

    Two OFF events are merged if:
    1. Gap between them is <= max_gap_minutes (default 2)
    2. No ON event occurs between them

    This handles cases where an appliance turns off in stages.

    Args:
        off_events: DataFrame with OFF events ('start', 'end', 'magnitude')
        on_events: DataFrame with ON events (to check for intervening ONs)
        max_gap_minutes: Maximum gap between OFF events to consider merging
        data: Power data DataFrame (indexed by timestamp) for magnitude recalculation
        phase: Phase column name for magnitude calculation

    Returns:
        DataFrame with merged OFF events
    """
    return _merge_consecutive_events(
        off_events, on_events, max_gap_minutes, data, phase
    )


def _merge_consecutive_events(events: pd.DataFrame, opposite_events: pd.DataFrame,
                               max_gap_minutes: int = 2,
                               data: pd.DataFrame = None, phase: str = None) -> pd.DataFrame:
    """
    Internal function to merge consecutive events of the same type.

    Only merges events if:
    1. Both events are instantaneous (duration = 0, i.e., start == end)
    2. Events are truly consecutive (gap <= 1 minute, one ends when next starts)
    3. No opposite event occurs between them

    This is very restrictive to avoid merging events from different appliances.
    Only merges when it's clearly the same appliance turning on/off across
    consecutive minutes (e.g., threshold crossed at 23:55 and again at 23:56).

    Args:
        events: DataFrame with events to merge ('start', 'end', 'magnitude')
        opposite_events: DataFrame with opposite events (to check for intervening events)
        max_gap_minutes: Maximum gap between events to consider merging (default 2, but
                        actual merge only happens if both events have duration 0)
        data: Power data DataFrame (indexed by timestamp) for magnitude recalculation
        phase: Phase column name for magnitude calculation

    Returns:
        DataFrame with merged events
    """
    if len(events) <= 1:
        return events

    # Sort by start time
    df = events.sort_values('start').reset_index(drop=True)

    # Get opposite event starts for quick lookup
    opposite_starts = set()
    if opposite_events is not None and len(opposite_events) > 0:
        opposite_starts = set(opposite_events['start'])

    merged = []
    current = df.iloc[0].copy()
    merge_count = 0

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
            # Merge: extend end time, magnitude will be recalculated
            current['end'] = next_event['end']
            merge_count += 1
        else:
            # Finalize current event
            if merge_count > 0 and data is not None and phase is not None:
                current['magnitude'] = _calc_magnitude(data, phase, current['start'], current['end'])
            merged.append(current)
            current = next_event.copy()
            merge_count = 0

    # Don't forget the last event
    if merge_count > 0 and data is not None and phase is not None:
        current['magnitude'] = _calc_magnitude(data, phase, current['start'], current['end'])
    merged.append(current)

    return pd.DataFrame(merged).reset_index(drop=True)
