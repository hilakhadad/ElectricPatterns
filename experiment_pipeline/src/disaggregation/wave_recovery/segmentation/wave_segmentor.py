"""
Wave-shaped power extraction from remaining signal.

For each validated WavePattern, extract the wave-shaped power from the
remaining signal and produce an updated remaining series.

The extraction follows the wave profile (not a flat rectangle), so
remaining[t] -= min(wave_profile[t], remaining[t] - baseline) at each
minute within the wave window.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..detection.wave_detector import WavePattern


def extract_wave_power(
    remaining: pd.Series,
    wave: WavePattern,
    logger=None,
) -> Tuple[pd.Series, pd.Series]:
    """
    Extract wave-shaped power from remaining and return updated remaining.

    Parameters
    ----------
    remaining : pd.Series
        Remaining power for the wave's phase, indexed by timestamp.
    wave : WavePattern
        The wave to extract.

    Returns
    -------
    extracted : pd.Series
        Power attributed to this wave (same index as remaining, zero outside wave).
    updated_remaining : pd.Series
        remaining - extracted (clipped so never < 0).
    """
    extracted = pd.Series(0.0, index=remaining.index)
    updated_remaining = remaining.copy()

    # Get the time range of the wave
    wave_mask = (remaining.index >= wave.start) & (remaining.index <= wave.end)
    wave_indices = remaining.index[wave_mask]

    if len(wave_indices) == 0:
        return extracted, updated_remaining

    # The wave_profile has len = wave.duration_minutes + 1 (inclusive endpoints)
    # Map each timestamp in wave_indices to its position in wave_profile
    profile = wave.wave_profile
    profile_len = len(profile)

    for i, ts in enumerate(wave_indices):
        if i >= profile_len:
            break

        # How much to extract: min of wave_profile and available remaining above baseline
        available = max(0.0, remaining.loc[ts] - wave.baseline_power)
        to_extract = min(profile[i], available)

        extracted.loc[ts] = to_extract
        updated_remaining.loc[ts] = remaining.loc[ts] - to_extract

    if logger:
        total = extracted.sum()
        logger.debug(f"Wave extraction {wave.phase}: extracted {total:.0f}W-min over {wave.duration_minutes} min")

    return extracted, updated_remaining
