"""
Tiered NaN imputation for power time series.

Fills short NaN gaps in-memory (does NOT modify INPUT files).
Applied before diff()-based detection to prevent false events at NaN boundaries.

Strategy by gap length:
  - <= ffill_limit minutes : forward fill (power barely changes in 1-5 min)
  - ffill_limit+1 to interp_limit minutes : linear interpolation
  - > interp_limit minutes : remain NaN (too long to guess reliably)
"""
import numpy as np
import pandas as pd


def impute_nan_gaps(data: pd.DataFrame, phase_cols=None,
                    ffill_limit: int = 5, interp_limit: int = 60,
                    logger=None) -> pd.DataFrame:
    """
    Tiered NaN imputation for power time series.

    Must be called AFTER sorting by timestamp and dropping all-NaN rows.
    Returns a **new** DataFrame — does not modify the input.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain a 'timestamp' column and the phase columns.
    phase_cols : list[str] or None
        Columns to impute. Defaults to ['w1', 'w2', 'w3'].
    ffill_limit : int
        Maximum consecutive NaN minutes to fill with forward fill (default 5).
    interp_limit : int
        Maximum consecutive NaN minutes to fill with linear interpolation (default 60).
    logger : logging.Logger or None
        Optional logger for diagnostics.

    Returns
    -------
    pd.DataFrame
        Copy of *data* with short/medium NaN gaps filled.
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
        filled_interp = 0

        # --- Tier 1: short gaps (<=ffill_limit) → forward fill ---
        short_mask = nan_mask & (group_lengths <= ffill_limit) & (group_lengths > 0)
        if short_mask.any():
            # ffill only the positions belonging to short gaps
            filled_values = series.ffill(limit=ffill_limit)
            result.loc[short_mask, col] = filled_values[short_mask]
            filled_ffill = short_mask.sum() - result[col][short_mask].isna().sum()

        # --- Tier 2: medium gaps (ffill_limit < gap <= interp_limit) → linear interpolation ---
        medium_mask = nan_mask & (group_lengths > ffill_limit) & (group_lengths <= interp_limit)
        if medium_mask.any():
            # Interpolate on the partially-filled series so short gaps don't interfere
            interp_values = result[col].interpolate(method='linear', limit=interp_limit,
                                                     limit_direction='forward')
            result.loc[medium_mask, col] = interp_values[medium_mask]
            filled_interp = medium_mask.sum() - result[col][medium_mask].isna().sum()

        # --- Tier 3: long gaps (> interp_limit) → remain NaN ---
        remaining_nan = result[col].isna().sum()

        if logger:
            logger.info(
                f"[NaN Imputation] {col}: {total_nan} NaN total → "
                f"ffill={filled_ffill}, interp={filled_interp}, "
                f"remaining={remaining_nan}"
            )

    return result
