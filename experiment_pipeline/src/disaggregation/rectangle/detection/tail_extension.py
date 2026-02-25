"""
Tail extension for OFF events.

Extends OFF events forward through residual power decay tails.
Devices have a physical "soft landing" (fan coasting, heating element cooling)
where power doesn't drop to zero instantly but decays over several minutes.

Example: AC turns off → sharp drop from 1600W to 280W → 280W decays to 0W over 8 minutes.
Without tail extension: detected magnitude = -1300W (misses the 280W tail).
With tail extension: detected magnitude = -1600W (captures full shutdown).
"""
import pandas as pd
import numpy as np


def _calc_magnitude_from_phase(data_indexed: pd.DataFrame, phase: str,
                                start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Calculate magnitude as power(end) - power(start - 1min)."""
    before_start = start - pd.Timedelta(minutes=1)
    try:
        value_end = float(data_indexed.loc[end, phase])
    except KeyError:
        return 0.0
    try:
        value_before = float(data_indexed.loc[before_start, phase])
    except KeyError:
        return 0.0
    return value_end - value_before


def _safe_lookup(data_indexed: pd.DataFrame, phase: str, timestamp: pd.Timestamp) -> float:
    """Look up power value at timestamp, return NaN if not found."""
    try:
        return float(data_indexed.loc[timestamp, phase])
    except KeyError:
        return float('nan')


def extend_off_event_tails(off_events: pd.DataFrame, data_indexed: pd.DataFrame, phase: str,
                            max_minutes: int = 10, min_residual: int = 100,
                            noise_tolerance: int = 30, min_gain: int = 100,
                            min_residual_fraction: float = 0.05,
                            logger=None) -> pd.DataFrame:
    """
    Extend OFF events forward through monotonically-decaying residual power tails.

    After a sharp power drop (the detected OFF event), some devices leave residual
    power that decays to zero over several minutes. This function extends the event
    end time to capture that tail, increasing the detected magnitude.

    Args:
        off_events: DataFrame of OFF events with columns: start, end, magnitude
        data_indexed: Power data with timestamp index and phase columns
        phase: Phase column name (e.g., 'w1')
        max_minutes: Maximum minutes to extend forward (default 10)
        min_residual: Minimum residual power (W) at event end to trigger extension (default 100)
        noise_tolerance: Maximum allowed power rise per step in watts (default 30)
        min_gain: Minimum magnitude gain (W) to keep the extension (default 100)
        min_residual_fraction: Minimum residual as fraction of |magnitude| (default 0.05)

    Returns:
        Updated OFF events DataFrame with tail_extended and tail_original_end columns
        for events that were extended.
    """
    if len(off_events) == 0:
        return off_events

    results = off_events.copy()

    for idx in results.index:
        event_end = results.at[idx, 'end']
        event_start = results.at[idx, 'start']
        magnitude = abs(results.at[idx, 'magnitude'])

        # Read residual power at event end
        residual = _safe_lookup(data_indexed, phase, event_end)
        if np.isnan(residual):
            continue

        # Check if residual is significant enough to extend
        if residual < min_residual:
            continue
        if residual < magnitude * min_residual_fraction:
            continue

        # Scan forward for monotonic decay
        prev_power = residual
        new_end = event_end

        for i in range(1, max_minutes + 1):
            ts = event_end + pd.Timedelta(minutes=i)
            current = _safe_lookup(data_indexed, phase, ts)

            if np.isnan(current):
                break  # Data gap

            if current > prev_power + noise_tolerance:
                break  # Power rising — different device or noise

            new_end = ts
            prev_power = current

            if current < min_residual:
                break  # Reached near-zero

        # Check if extension is worthwhile
        final_power = _safe_lookup(data_indexed, phase, new_end)
        if np.isnan(final_power):
            continue

        gain = residual - final_power
        if gain < min_gain:
            continue

        # Apply extension
        new_magnitude = _calc_magnitude_from_phase(data_indexed, phase, event_start, new_end)

        # Only apply if new magnitude is actually larger (more negative for OFF)
        if abs(new_magnitude) > magnitude:
            results.at[idx, 'tail_original_end'] = event_end
            results.at[idx, 'end'] = new_end
            results.at[idx, 'magnitude'] = new_magnitude
            results.at[idx, 'tail_extended'] = True

    # Fill NaN for events that weren't extended
    if 'tail_extended' not in results.columns:
        results['tail_extended'] = False
    else:
        results.loc[results['tail_extended'].isna(), 'tail_extended'] = False

    if logger:
        extended = int(results['tail_extended'].sum()) if 'tail_extended' in results.columns else 0
        if extended:
            logger.debug(f"Tail extension {phase}: {extended}/{len(off_events)} OFF events extended")
    return results
