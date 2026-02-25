"""
Detect wave-shaped power patterns in remaining signal.

A "wave" is a sharp power rise followed by a gradual monotonic decay back to
baseline — the signature of an AC compressor cycle that starts high and
decreases as the room cools.  Rectangle matching misses these because the ON
magnitude is much larger than the OFF magnitude.

Algorithm:
    1. Compute minute-to-minute diffs of remaining power.
    2. Find sharp rises >= wave_min_rise_watts.
    3. For each rise, scan forward for a monotonic decrease (with tolerance).
    4. Validate: duration in bounds, power decays >= min_decay_fraction from peak.
    5. Find the return-to-baseline point.

Returns a list of WavePattern dataclass instances.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class WavePattern:
    """A single detected wave event."""
    start: pd.Timestamp          # Minute of the sharp rise
    peak_time: pd.Timestamp      # Minute where power is highest
    end: pd.Timestamp            # Minute where power returns to baseline
    phase: str                   # 'w1', 'w2', or 'w3'
    peak_power: float            # Power at peak (watts)
    baseline_power: float        # Estimated baseline before the rise
    duration_minutes: int        # end - start in minutes
    wave_profile: np.ndarray     # Per-minute power above baseline (len = duration)


def detect_wave_patterns(
    remaining: pd.Series,
    phase: str,
    config,
    logger=None,
) -> List[WavePattern]:
    """
    Detect wave-shaped patterns in a remaining-power time series.

    Parameters
    ----------
    remaining : pd.Series
        Remaining power for a single phase, indexed by timestamp (1-min resolution).
    phase : str
        Phase name ('w1', 'w2', 'w3').
    config : ExperimentConfig
        Must have wave_min_rise_watts, wave_min_duration_minutes,
        wave_max_duration_minutes, wave_monotonic_tolerance, wave_min_decay_fraction.
    logger : optional
        Logger instance for debug output.

    Returns
    -------
    List[WavePattern]
        Detected wave patterns, sorted by start time.
    """
    if remaining.empty or len(remaining) < 3:
        return []

    min_rise = config.wave_min_rise_watts
    min_dur = config.wave_min_duration_minutes
    max_dur = config.wave_max_duration_minutes
    mono_tol = config.wave_monotonic_tolerance
    min_decay = config.wave_min_decay_fraction

    values = remaining.values.astype(float)
    timestamps = remaining.index
    diffs = np.diff(values)

    # Neutralize diffs at time gaps (NaN regions removed by dropna upstream)
    # to prevent false wave detection from communication holes.
    MAX_CONTINUITY_GAP_MINUTES = 5
    time_gaps = np.diff(timestamps).astype('timedelta64[m]').astype(float)
    gap_mask = time_gaps > MAX_CONTINUITY_GAP_MINUTES
    diffs[gap_mask] = 0.0

    # Find candidate sharp-rise indices (minute *before* the rise ends)
    rise_indices = np.where(diffs >= min_rise)[0]

    waves: List[WavePattern] = []
    used_until = -1  # Prevent overlapping waves

    for rise_idx in rise_indices:
        if rise_idx <= used_until:
            continue

        peak_idx = rise_idx + 1  # The minute after the rise
        peak_power = values[peak_idx]

        # Baseline: mean of up to 3 minutes before the rise (excluding NaN)
        baseline_start = max(0, rise_idx - 2)
        baseline_slice = values[baseline_start:rise_idx + 1]
        valid_baseline = baseline_slice[~np.isnan(baseline_slice)]
        if len(valid_baseline) == 0:
            continue
        baseline_power = float(np.mean(valid_baseline))

        if peak_power - baseline_power < min_rise:
            continue

        # Scan forward for monotonic decay
        end_idx = _find_wave_end(
            values, peak_idx, baseline_power, max_dur, mono_tol, min_decay
        )
        if end_idx is None:
            continue

        duration = end_idx - rise_idx
        if duration < min_dur:
            continue

        # Build the wave profile (power above baseline, clipped to >= 0)
        raw_profile = values[rise_idx:end_idx + 1] - baseline_power
        wave_profile = np.clip(raw_profile, 0, None)

        wave = WavePattern(
            start=timestamps[rise_idx],
            peak_time=timestamps[peak_idx],
            end=timestamps[end_idx],
            phase=phase,
            peak_power=float(peak_power),
            baseline_power=float(baseline_power),
            duration_minutes=duration,
            wave_profile=wave_profile,
        )
        waves.append(wave)
        used_until = end_idx

        if logger:
            logger.debug(
                f"Wave detected on {phase}: {wave.start} -> {wave.end} "
                f"({duration} min), peak={peak_power:.0f}W, "
                f"baseline={baseline_power:.0f}W"
            )

    if logger:
        logger.debug(f"Wave detection {phase}: {len(rise_indices)} rise candidates -> {len(waves)} waves")

    return waves


def _find_wave_end(
    values: np.ndarray,
    peak_idx: int,
    baseline: float,
    max_dur: int,
    mono_tol: float,
    min_decay: float,
) -> Optional[int]:
    """
    Scan forward from peak to find where the wave returns to baseline.

    The decay must be "mostly monotonic" — at most ``mono_tol`` fraction of
    minute-to-minute transitions may increase.

    Returns the index of the end minute, or None if no valid wave end found.
    """
    peak_power = values[peak_idx]
    required_decay = (peak_power - baseline) * min_decay

    scan_end = min(peak_idx + max_dur, len(values) - 1)
    if scan_end <= peak_idx:
        return None

    segment = values[peak_idx:scan_end + 1]
    segment_diffs = np.diff(segment)

    # Track running minimum and find first return to baseline
    end_idx = None
    increase_count = 0
    total_steps = 0

    for i in range(1, len(segment)):
        total_steps += 1
        current = segment[i]

        # Count increases (non-monotonic steps)
        if segment_diffs[i - 1] > 0:
            increase_count += 1

        # Check if we've decayed enough and returned near baseline
        decay_from_peak = peak_power - current
        if decay_from_peak >= required_decay and current <= baseline + 0.2 * (peak_power - baseline):
            end_idx = peak_idx + i
            break

    if end_idx is None:
        # Didn't return to baseline — check if the last point is close enough
        # (wave might extend to end of data or next event)
        last_val = segment[-1] if len(segment) > 0 else peak_power
        if peak_power - last_val >= required_decay:
            end_idx = scan_end
        else:
            return None

    # Validate monotonicity
    actual_steps = end_idx - peak_idx
    if actual_steps <= 0:
        return None

    increases = np.sum(segment_diffs[:actual_steps] > 0)
    if actual_steps > 0 and increases / actual_steps > mono_tol:
        return None

    return end_idx
