"""
Event merging logic.

Merges overlapping or touching events into single events.
"""
import pandas as pd


def merge_overlapping_events(events_df: pd.DataFrame, max_gap_minutes: int = 0) -> pd.DataFrame:
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

    Returns:
        DataFrame with merged events
    """
    if len(events_df) <= 1:
        return events_df

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
            # Merge: extend end time and add magnitudes
            current['end'] = max(current['end'], next_event['end'])
            current['magnitude'] = current['magnitude'] + next_event['magnitude']
        else:
            # No overlap - save current and start new
            merged.append(current)
            current = next_event.copy()

    # Don't forget the last event
    merged.append(current)

    return pd.DataFrame(merged).reset_index(drop=True)
