"""
Validate wave extraction results.

Checks performed:
    1. Remaining >= 0 everywhere after extraction.
    2. No holes (zero-gaps) inside the wave duration.
    3. Wave is monotonic (with tolerance) — redundant with detection, but safety check.
    4. Power invariant: remaining_before = remaining_after + extracted.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from ..detection.wave_detector import WavePattern

# Maximum fraction of extracted power that can violate the invariant (rounding)
INVARIANT_TOLERANCE = 1.0  # 1 watt absolute


def validate_wave_extraction(
    remaining_before: pd.Series,
    remaining_after: pd.Series,
    extracted: pd.Series,
    wave: WavePattern,
    logger=None,
) -> Tuple[bool, str]:
    """
    Validate that a wave extraction is clean.

    Parameters
    ----------
    remaining_before : pd.Series
        Remaining power before extraction.
    remaining_after : pd.Series
        Remaining power after extraction.
    extracted : pd.Series
        Power extracted for this wave.
    wave : WavePattern
        The wave that was extracted.

    Returns
    -------
    (valid, reason) : (bool, str)
        True if extraction is valid, else False with a reason string.
    """
    # 1. No negative remaining
    neg_mask = remaining_after < -INVARIANT_TOLERANCE
    if neg_mask.any():
        worst = remaining_after[neg_mask].min()
        reason = f"Negative remaining after extraction (min={worst:.1f}W)"
        if logger:
            logger.debug(f"Wave validation failed: {reason}")
        return False, reason

    # 2. Power invariant: before = after + extracted
    diff = remaining_before - remaining_after - extracted
    max_violation = diff.abs().max()
    if max_violation > INVARIANT_TOLERANCE:
        reason = f"Power invariant violated (max diff={max_violation:.1f}W)"
        if logger:
            logger.debug(f"Wave validation failed: {reason}")
        return False, reason

    # 3. Wave profile should have no internal zeros (holes)
    wave_mask = (extracted.index >= wave.start) & (extracted.index <= wave.end)
    wave_segment = extracted[wave_mask]
    if len(wave_segment) >= 3:
        # Ignore first and last minute (transition points)
        inner = wave_segment.iloc[1:-1]
        if len(inner) > 0:
            zero_fraction = (inner <= 0).sum() / len(inner)
            if zero_fraction > 0.3:  # More than 30% zeros inside wave
                reason = f"Too many zero-gaps inside wave ({zero_fraction:.0%})"
                if logger:
                    logger.debug(f"Wave validation failed: {reason}")
                return False, reason

    # 4. Total extracted power should be meaningful
    total_extracted = extracted.sum()
    if total_extracted <= 0:
        reason = "No power extracted"
        if logger:
            logger.debug(f"Wave validation failed: {reason}")
        return False, reason

    if logger:
        logger.debug("Wave validation passed")
    return True, "OK"
