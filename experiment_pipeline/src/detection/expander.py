"""
Event expansion logic.

Expands detected events to include nearby small changes.
"""
import pandas as pd


def expand_event(
    event: pd.Series,
    data: pd.DataFrame,
    event_type: str,
    diff_col: str,
    expand_factor: float = 0.05
) -> pd.Series:
    """
    Expand an event to include adjacent small changes.

    Looks 1 minute before and after the event. If there are changes
    in the same direction that exceed the threshold factor, include them.

    Args:
        event: Series with 'start', 'end', 'magnitude'
        data: Full DataFrame with timestamp and diff columns
        event_type: 'on' or 'off'
        diff_col: Name of the diff column
        expand_factor: Threshold factor for expansion (default 5% of magnitude)

    Returns:
        Updated event Series with potentially expanded start/end/magnitude
    """
    start = event['start']
    end = event['end']
    magnitude = event['magnitude']

    before_start = start - pd.Timedelta(minutes=1)
    after_end = end + pd.Timedelta(minutes=1)

    before_data = data[(data['timestamp'] >= before_start) & (data['timestamp'] < start)]
    after_data = data[(data['timestamp'] > end) & (data['timestamp'] <= after_end)]

    before_magnitude = before_data[diff_col].sum()
    after_magnitude = after_data[diff_col].sum()

    # Use absolute value for threshold calculation
    threshold = expand_factor * abs(magnitude)

    if event_type == 'on':
        # For ON events: look for positive changes
        if before_magnitude > threshold:
            start = before_start
            magnitude += before_magnitude
        if after_magnitude > threshold:
            end = after_end
            magnitude += after_magnitude

    elif event_type == 'off':
        # For OFF events: look for negative changes
        if before_magnitude < -threshold:
            start = before_start
            magnitude += before_magnitude
        if after_magnitude < -threshold:
            end = after_end
            magnitude += after_magnitude

    return pd.Series({'start': start, 'end': end, 'magnitude': magnitude})
