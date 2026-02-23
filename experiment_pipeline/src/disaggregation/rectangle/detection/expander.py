"""
Event expansion logic.

Expands detected events to include nearby small changes.
"""
import pandas as pd


def _calc_magnitude_from_phase(data: pd.DataFrame, phase: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Calculate event magnitude as the difference between power at end and power before start.
    """
    before_start = start - pd.Timedelta(minutes=1)

    # Use index-based lookup if timestamp is the index
    if data.index.name == 'timestamp':
        try:
            value_end = data.loc[end, phase]
        except KeyError:
            value_end = 0
        try:
            value_before = data.loc[before_start, phase]
        except KeyError:
            value_before = 0
    else:
        # Fallback to boolean indexing
        end_mask = data['timestamp'] == end
        value_end = data.loc[end_mask, phase].values[0] if end_mask.any() else 0

        before_mask = data['timestamp'] == before_start
        value_before = data.loc[before_mask, phase].values[0] if before_mask.any() else 0

    return value_end - value_before


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

    # First calculate actual magnitude from phase values
    phase = diff_col.replace('_diff', '')
    if phase in data.columns:
        magnitude = _calc_magnitude_from_phase(data, phase, start, end)
    else:
        magnitude = event['magnitude']

    before_start = start - pd.Timedelta(minutes=1)
    after_end = end + pd.Timedelta(minutes=1)

    # Use index-based slicing if timestamp is the index
    if data.index.name == 'timestamp':
        try:
            before_data = data.loc[before_start:start].iloc[:-1] if start in data.index else data.loc[before_start:start]
            before_magnitude = before_data[diff_col].sum() if len(before_data) > 0 else 0
        except KeyError:
            before_magnitude = 0
        try:
            after_data = data.loc[end:after_end].iloc[1:] if end in data.index else data.loc[end:after_end]
            after_magnitude = after_data[diff_col].sum() if len(after_data) > 0 else 0
        except KeyError:
            after_magnitude = 0
    else:
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
        if after_magnitude > threshold:
            end = after_end

    elif event_type == 'off':
        # For OFF events: look for negative changes
        if before_magnitude < -threshold:
            start = before_start
        if after_magnitude < -threshold:
            end = after_end

    # Recalculate magnitude if expanded
    if start != event['start'] or end != event['end']:
        if phase in data.columns:
            magnitude = _calc_magnitude_from_phase(data, phase, start, end)

    return pd.Series({'start': start, 'end': end, 'magnitude': magnitude})
