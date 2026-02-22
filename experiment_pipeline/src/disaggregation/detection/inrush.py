"""
Inrush spike normalization for ON and OFF events.

Detects transient inrush spikes at the start of ON events (or end of OFF events)
and extends event boundaries to include the settling period. This ensures the
cumsum-based segmentation tracks the correct steady-state device power.

Problem: An AC compressor turns on with a 4000W inrush spike that settles to
1500W steady state after 1-2 minutes. Without normalization:
  - ON magnitude = 4000W (spike)
  - Segmentation extracts at 4000W → "pit" in remaining
  - Matching fails (4000W ON vs 1500W OFF → diff > 350W)

Solution: Extend ON event boundary through the settling period:
  - Spike at T: cumsum = +4000
  - Settling at T+1: cumsum = +4000 + (-2500) = +1500
  - device_power = on_seg[-1] = 1500 ← correct steady-state
  - Magnitude recalculated from phase values = 1500W
"""
import pandas as pd
import numpy as np
from typing import Optional


def normalize_inrush_on_events(
    on_events: pd.DataFrame,
    data_indexed: pd.DataFrame,
    phase: str,
    off_events: pd.DataFrame = None,
    settling_factor: float = 0.7,
    max_settling_minutes: int = 5,
    min_threshold: float = 0,
) -> pd.DataFrame:
    """
    Detect inrush spikes in ON events and extend boundaries to include settling.

    For each ON event, looks at the power diffs in the minutes after the event
    ends. If there's a significant opposite-direction change (>30% of magnitude),
    the event boundary is extended to include the settling period, and the
    magnitude is recalculated from actual phase values.

    Args:
        on_events: DataFrame with ON events (start, end, magnitude)
        data_indexed: Power DataFrame with timestamp as index
        phase: Phase column name (w1, w2, w3)
        off_events: DataFrame with OFF events (to avoid extending into them)
        settling_factor: Minimum ratio of settled/spike magnitude to trigger
                        normalization. 0.7 means >30% drop triggers it.
        max_settling_minutes: Maximum minutes to look ahead for settling
        min_threshold: Minimum magnitude after normalization (events below
                      this are not normalized to avoid dropping below threshold)

    Returns:
        Modified ON events DataFrame with extended boundaries where inrush detected
    """
    if len(on_events) == 0:
        return on_events

    diff_col = f'{phase}_diff'
    if diff_col not in data_indexed.columns:
        return on_events

    # Build set of occupied timestamps (other ON and OFF events) to avoid extending into
    occupied = set()
    for _, ev in on_events.iterrows():
        ts = ev['start']
        while ts <= ev['end']:
            occupied.add(ts)
            ts += pd.Timedelta(minutes=1)
    if off_events is not None and len(off_events) > 0:
        for _, ev in off_events.iterrows():
            ts = ev['start']
            while ts <= ev['end']:
                occupied.add(ts)
                ts += pd.Timedelta(minutes=1)

    results = []
    for _, event in on_events.iterrows():
        event = event.copy()
        result = _normalize_single_on_event(
            event, data_indexed, phase, diff_col, occupied,
            settling_factor, max_settling_minutes, min_threshold,
        )
        results.append(result)

    return pd.DataFrame(results).reset_index(drop=True)


def _normalize_single_on_event(
    event: pd.Series,
    data_indexed: pd.DataFrame,
    phase: str,
    diff_col: str,
    occupied: set,
    settling_factor: float,
    max_settling_minutes: int,
    min_threshold: float,
) -> pd.Series:
    """Normalize a single ON event if it has an inrush spike."""
    on_end = event['end']
    magnitude = abs(event['magnitude'])

    if magnitude == 0:
        return event

    # Look at the minutes after the ON event for settling (power dropping back
    # from spike to steady state). Stop once power stabilizes (2+ consecutive
    # near-zero diffs) to avoid extending through the entire stable period
    # into the OFF event.
    settling_end = None
    cumulative_drop = 0.0
    consecutive_stable = 0

    for offset in range(1, max_settling_minutes + 1):
        check_ts = on_end + pd.Timedelta(minutes=offset)

        # Don't extend into another event
        if check_ts in occupied:
            break

        # Get the diff at this timestamp
        if check_ts not in data_indexed.index:
            break

        diff_value = data_indexed.loc[check_ts, diff_col]
        if pd.isna(diff_value):
            break

        if diff_value < 0:
            # Power is dropping — this is part of the settling
            cumulative_drop += diff_value
            settling_end = check_ts
            consecutive_stable = 0
        elif diff_value > magnitude * 0.1:
            # Power is rising significantly — settling is over
            break
        else:
            # Small positive or zero change — power is stabilizing
            if cumulative_drop == 0:
                break  # No settling started yet
            consecutive_stable += 1
            if consecutive_stable >= 2:
                break  # Power has stabilized — settling is complete

    # Check if the drop is significant enough to be an inrush
    if settling_end is None or abs(cumulative_drop) < magnitude * (1 - settling_factor):
        return event  # No significant inrush detected

    # Calculate new magnitude from actual phase values
    before_start = event['start'] - pd.Timedelta(minutes=1)
    try:
        value_before = data_indexed.loc[before_start, phase]
    except KeyError:
        return event
    try:
        value_at_settling = data_indexed.loc[settling_end, phase]
    except KeyError:
        return event

    if pd.isna(value_before) or pd.isna(value_at_settling):
        return event

    new_magnitude = value_at_settling - value_before

    # Safety: don't normalize if new magnitude would be too small
    if new_magnitude < min_threshold:
        return event

    # Safety: new magnitude must be positive (it's an ON event)
    if new_magnitude <= 0:
        return event

    # Store original values and extend
    event['inrush_original_end'] = event['end']
    event['inrush_original_magnitude'] = event['magnitude']
    event['end'] = settling_end
    event['magnitude'] = new_magnitude

    return event


def normalize_inrush_off_events(
    off_events: pd.DataFrame,
    data_indexed: pd.DataFrame,
    phase: str,
    on_events: pd.DataFrame = None,
    settling_factor: float = 0.7,
    max_settling_minutes: int = 5,
    min_threshold: float = 0,
) -> pd.DataFrame:
    """
    Detect outgoing spikes in OFF events and extend boundaries backward.

    For each OFF event, looks at the power diffs in the minutes before the
    event starts. If there's a significant opposite-direction change (power
    spikes UP before dropping), the event boundary is extended backward to
    include the spike, and magnitude is recalculated.

    Args:
        off_events: DataFrame with OFF events (start, end, magnitude)
        data_indexed: Power DataFrame with timestamp as index
        phase: Phase column name (w1, w2, w3)
        on_events: DataFrame with ON events (to avoid extending into them)
        settling_factor: Minimum ratio to trigger normalization
        max_settling_minutes: Maximum minutes to look backward for spike
        min_threshold: Minimum magnitude after normalization

    Returns:
        Modified OFF events DataFrame with extended boundaries where spike detected
    """
    if len(off_events) == 0:
        return off_events

    diff_col = f'{phase}_diff'
    if diff_col not in data_indexed.columns:
        return off_events

    # Build set of occupied timestamps
    occupied = set()
    for _, ev in off_events.iterrows():
        ts = ev['start']
        while ts <= ev['end']:
            occupied.add(ts)
            ts += pd.Timedelta(minutes=1)
    if on_events is not None and len(on_events) > 0:
        for _, ev in on_events.iterrows():
            ts = ev['start']
            while ts <= ev['end']:
                occupied.add(ts)
                ts += pd.Timedelta(minutes=1)

    results = []
    for _, event in off_events.iterrows():
        event = event.copy()
        result = _normalize_single_off_event(
            event, data_indexed, phase, diff_col, occupied,
            settling_factor, max_settling_minutes, min_threshold,
        )
        results.append(result)

    return pd.DataFrame(results).reset_index(drop=True)


def _normalize_single_off_event(
    event: pd.Series,
    data_indexed: pd.DataFrame,
    phase: str,
    diff_col: str,
    occupied: set,
    settling_factor: float,
    max_settling_minutes: int,
    min_threshold: float,
) -> pd.Series:
    """Normalize a single OFF event if it has an outgoing spike."""
    off_start = event['start']
    magnitude = abs(event['magnitude'])

    if magnitude == 0:
        return event

    # Look at the minutes before the OFF event for a pre-shutdown spike
    # (power rising before the drop). Stop once power was stable (2+
    # consecutive near-zero diffs going backward) to avoid extending
    # through the entire stable period into the ON event.
    spike_start = None
    cumulative_rise = 0.0
    consecutive_stable = 0

    for offset in range(1, max_settling_minutes + 1):
        check_ts = off_start - pd.Timedelta(minutes=offset)

        # Don't extend into another event
        if check_ts in occupied:
            break

        if check_ts not in data_indexed.index:
            break

        diff_value = data_indexed.loc[check_ts, diff_col]
        if pd.isna(diff_value):
            break

        if diff_value > 0:
            # Power was rising — this is part of the pre-shutdown spike
            cumulative_rise += diff_value
            spike_start = check_ts
            consecutive_stable = 0
        elif diff_value < -magnitude * 0.1:
            # Power was dropping significantly — not part of the spike
            break
        else:
            # Small negative or zero — power was stable
            if cumulative_rise == 0:
                break  # No spike detected yet
            consecutive_stable += 1
            if consecutive_stable >= 2:
                break  # Power was stable before the spike — done

    # Check if the spike is significant enough
    if spike_start is None or cumulative_rise < magnitude * (1 - settling_factor):
        return event

    # Calculate new magnitude from actual phase values
    before_spike = spike_start - pd.Timedelta(minutes=1)
    try:
        value_before = data_indexed.loc[before_spike, phase]
    except KeyError:
        return event

    end_ts = event['end']
    # For OFF events, get value at end (or end + 1min for the post-OFF value)
    after_end = end_ts + pd.Timedelta(minutes=1)
    try:
        value_after = data_indexed.loc[after_end, phase]
    except KeyError:
        try:
            value_after = data_indexed.loc[end_ts, phase]
        except KeyError:
            return event

    if pd.isna(value_before) or pd.isna(value_after):
        return event

    new_magnitude = value_after - value_before  # Will be negative for OFF

    # Safety: new magnitude must be negative (it's an OFF event) and significant
    if new_magnitude >= 0:
        return event
    if abs(new_magnitude) < min_threshold:
        return event

    # Store original values and extend
    event['inrush_original_start'] = event['start']
    event['inrush_original_magnitude'] = event['magnitude']
    event['start'] = spike_start
    event['magnitude'] = new_magnitude

    return event
