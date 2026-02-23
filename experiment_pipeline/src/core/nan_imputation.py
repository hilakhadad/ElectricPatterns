"""
Short-gap NaN imputation for power time series.

Fills short NaN gaps in-memory (does NOT modify INPUT files).
Applied before diff()-based detection to prevent false events at NaN boundaries.

Strategy by gap length:
  - <= ffill_limit minutes : forward fill (power barely changes in 1-5 min)
  - > ffill_limit minutes : remain NaN (preserve real signal structure)
"""
import numpy as np
import pandas as pd


def impute_nan_gaps(data: pd.DataFrame, phase_cols=None,
                    ffill_limit: int = 5,
                    logger=None) -> pd.DataFrame:
    """
    Short-gap NaN imputation for power time series.

    Must be called AFTER sorting by timestamp and dropping all-NaN rows.
    Returns a **new** DataFrame — does not modify the input.

    Only fills gaps of up to *ffill_limit* consecutive minutes using forward
    fill.  Longer gaps remain NaN so that cross-phase patterns (e.g. 3-phase
    devices) are not masked by artificial interpolation.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'timestamp' column and the phase columns.
    phase_cols : list[str] or None
        Columns to impute. Defaults to ['w1', 'w2', 'w3'].
    ffill_limit : int
        Maximum consecutive NaN minutes to fill with forward fill (default 5).
    logger : logging.Logger or None
        Optional logger for diagnostics.

    Returns
    -------
    pd.DataFrame
        Copy of *data* with short NaN gaps filled.
    """
    if phase_cols is None:
        phase_cols = ['w1', 'w2', 'w3']

    # Work on a copy so the caller's DataFrame is untouched
    result = data.copy()

    for col in phase_cols:
        if col not in result.columns:
            continue

        series = result[col]
        nan_mask = series.isna()
        total_nan = nan_mask.sum()

        if total_nan == 0:
            continue

        # Identify contiguous NaN groups
        # Each group gets a unique ID; non-NaN positions get 0
        group_ids = (nan_mask != nan_mask.shift()).cumsum()
        group_ids = group_ids.where(nan_mask, 0)

        # Calculate the length of each NaN group
        group_lengths = group_ids.map(group_ids.value_counts()).where(nan_mask, 0)

        filled_ffill = 0

        # --- Tier 1: short gaps (<=ffill_limit) → forward fill ---
        short_mask = nan_mask & (group_lengths <= ffill_limit) & (group_lengths > 0)
        if short_mask.any():
            # ffill only the positions belonging to short gaps
            filled_values = series.ffill(limit=ffill_limit)
            result.loc[short_mask, col] = filled_values[short_mask]
            filled_ffill = short_mask.sum() - result[col][short_mask].isna().sum()

        # --- Longer gaps remain NaN ---
        remaining_nan = result[col].isna().sum()

        if logger:
            logger.info(
                f"[NaN Imputation] {col}: {total_nan} NaN total → "
                f"ffill={filled_ffill}, remaining={remaining_nan}"
            )

    return result
