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
