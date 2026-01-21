"""
Sharp event detection.

Detects single-minute power jumps that exceed the threshold.
"""
import pandas as pd
import numpy as np
from typing import Tuple

from .expander import expand_event


def _calc_magnitude(df: pd.DataFrame, phase: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Calculate event magnitude as the difference between power at end and power before start.

    Args:
        df: DataFrame with timestamp and phase columns
        phase: Phase column name
        start: Event start timestamp
        end: Event end timestamp

    Returns:
        Power difference (positive for ON, negative for OFF)
    """
    # Get value at end of event
    end_row = df[df['timestamp'] == end]
    value_end = end_row[phase].values[0] if len(end_row) > 0 else 0

    # Get value before start of event (1 minute before)
    before_start = start - pd.Timedelta(minutes=1)
    before_row = df[df['timestamp'] == before_start]
    value_before = before_row[phase].values[0] if len(before_row) > 0 else 0

    return value_end - value_before


def detect_sharp_events(
    data: pd.DataFrame,
    phase: str,
    threshold: int,
    off_threshold_factor: float = 0.8,
    expand_factor: float = 0.05
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect sharp ON/OFF events (single-minute jumps >= threshold).

    Args:
        data: DataFrame with timestamp and phase columns
        phase: Phase name ('w1', 'w2', 'w3')
        threshold: Power threshold for ON events (e.g., 1500W)
        off_threshold_factor: Factor for OFF threshold (e.g., 0.8 = 80%)
        expand_factor: Factor for event expansion (e.g., 0.05 = 5%)

    Returns:
        Tuple of (on_events_df, off_events_df)
    """
    df = data.copy()
    diff_col = f'{phase}_diff'

    # Calculate diff if not present
    if diff_col not in df.columns:
        df[diff_col] = df[phase].diff()

    # OFF threshold is configurable (default 80% of ON threshold)
    off_threshold = int(threshold * off_threshold_factor)

    # Mark ON/OFF events
    on_col = f"{phase}_{threshold}_on"
    off_col = f"{phase}_{threshold}_off"

    df[on_col] = np.where(df[diff_col] >= threshold, df[diff_col], 0)
    df[off_col] = np.where(df[diff_col] <= -off_threshold, df[diff_col], 0)

    # Group consecutive non-zero values
    is_on = f'{on_col}_group'
    is_off = f'{off_col}_group'

    df[is_on] = (df[on_col] == 0).cumsum() * (df[on_col] != 0)
    df[is_off] = (df[off_col] == 0).cumsum() * (df[off_col] != 0)

    # Aggregate into events - use phase value difference (end - start) for magnitude
    results_on = (
        df[df[on_col] != 0]
        .groupby(is_on)
        .agg(
            start=('timestamp', 'min'),
            end=('timestamp', 'max'),
            value_start=(phase, 'first'),
            value_end=(phase, 'last')
        )
        .reset_index(drop=True)
    )
    if len(results_on) > 0:
        # Calculate magnitude as: value at end - value at (start - 1 minute)
        results_on['magnitude'] = results_on.apply(
            lambda row: _calc_magnitude(df, phase, row['start'], row['end']), axis=1
        )
        results_on = results_on.drop(columns=['value_start', 'value_end'])

    results_off = (
        df[df[off_col] != 0]
        .groupby(is_off)
        .agg(
            start=('timestamp', 'min'),
            end=('timestamp', 'max'),
            value_start=(phase, 'first'),
            value_end=(phase, 'last')
        )
        .reset_index(drop=True)
    )
    if len(results_off) > 0:
        results_off['magnitude'] = results_off.apply(
            lambda row: _calc_magnitude(df, phase, row['start'], row['end']), axis=1
        )
        results_off = results_off.drop(columns=['value_start', 'value_end'])

    # Expand events to include adjacent small changes
    if len(results_on) > 0:
        results_on = results_on.apply(
            lambda x: expand_event(x, df, 'on', diff_col, expand_factor),
            axis=1
        )

    if len(results_off) > 0:
        results_off = results_off.apply(
            lambda x: expand_event(x, df, 'off', diff_col, expand_factor),
            axis=1
        )

    return results_on, results_off
