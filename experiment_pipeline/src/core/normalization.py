"""
Normalization preprocessing for power data.

Three scale-preserving normalization methods that remove noise/drift
while keeping absolute watt values intact for threshold-based detection.

Methods:
    - MA Detrending: removes slow baseline drift (2-hour moving average)
    - Phase Balancing: equalizes baselines across 3 phases
    - MAD Outlier Cleaning: removes measurement spikes using robust statistics

All methods preserve the absolute watt scale so existing detection
thresholds (2000, 1500, 1100, 800W) work without adjustment.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from .data_loader import find_house_data_path, load_power_data, get_monthly_files


PHASES = ['w1', 'w2', 'w3']


def detrend_moving_average(
    data: pd.DataFrame,
    window_minutes: int = 120,
    phases: List[str] = None,
) -> pd.DataFrame:
    """Remove baseline drift using centered moving average.

    Computes: detrended = signal - rolling_mean + global_mean

    This removes slow drift (time-of-day changes, seasonal trends) while
    preserving the absolute watt scale. Since the rolling mean changes slowly,
    diff() values are approximately unchanged, so detection thresholds still work.

    Args:
        data: DataFrame with timestamp, w1, w2, w3 columns.
        window_minutes: Rolling window size in minutes (default 120 = 2 hours).
        phases: Phase columns to normalize (default ['w1', 'w2', 'w3']).

    Returns:
        DataFrame with detrended phase columns (timestamp unchanged).
    """
    if phases is None:
        phases = PHASES
    result = data.copy()

    for phase in phases:
        series = result[phase]
        rolling_mean = series.rolling(
            window=window_minutes, min_periods=1, center=True
        ).mean()
        global_mean = series.mean()
        result[phase] = (series - rolling_mean + global_mean).clip(lower=0)

    return result


def balance_phases(
    data: pd.DataFrame,
    phases: List[str] = None,
) -> pd.DataFrame:
    """Equalize phase baselines by shifting to a common median.

    Computes: balanced = phase - phase_median + global_median

    Israeli households often have imbalanced loads across 3 phases.
    This shifts each phase so all share the same median baseline,
    making detection equally sensitive on all phases.
    Event magnitudes (diff values) are unchanged since this is a constant shift.

    Args:
        data: DataFrame with timestamp, w1, w2, w3 columns.
        phases: Phase columns to balance (default ['w1', 'w2', 'w3']).

    Returns:
        DataFrame with balanced phase columns.
    """
    if phases is None:
        phases = PHASES
    result = data.copy()

    medians = {phase: result[phase].median() for phase in phases}
    global_median = np.mean(list(medians.values()))

    for phase in phases:
        result[phase] = (result[phase] - medians[phase] + global_median).clip(lower=0)

    return result


def mad_outlier_cleaning(
    data: pd.DataFrame,
    window_minutes: int = 240,
    k: float = 5.0,
    phases: List[str] = None,
) -> pd.DataFrame:
    """Remove measurement outliers using rolling MAD threshold.

    For each point, if |value - rolling_median| > k * rolling_MAD,
    replace it with the rolling median. This removes sensor spikes
    and measurement noise while preserving real device events.

    Uses a 4-hour window by default and k=5.0 (very conservative:
    only extreme outliers are replaced, real events are kept).

    Args:
        data: DataFrame with timestamp, w1, w2, w3 columns.
        window_minutes: Rolling window for median/MAD calculation (default 240 = 4 hours).
        k: Outlier threshold in MAD units (default 5.0).
        phases: Phase columns to clean (default ['w1', 'w2', 'w3']).

    Returns:
        DataFrame with outliers replaced by rolling median.
    """
    if phases is None:
        phases = PHASES
    result = data.copy()

    for phase in phases:
        series = result[phase]
        rolling_med = series.rolling(
            window=window_minutes, min_periods=1, center=True
        ).median()
        deviations = (series - rolling_med).abs()
        rolling_mad = deviations.rolling(
            window=window_minutes, min_periods=1, center=True
        ).median()

        # Where MAD > 0: outlier if deviation > k * MAD
        # Where MAD == 0 (constant signal): any deviation > 0 is an outlier
        is_outlier = ((rolling_mad > 0) & (deviations > k * rolling_mad)) | (
            (rolling_mad == 0) & (deviations > 0)
        )
        result.loc[is_outlier, phase] = rolling_med[is_outlier]

    return result


def apply_normalization(
    data: pd.DataFrame,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Apply a normalization method to power data.

    Args:
        data: DataFrame with timestamp, w1, w2, w3 columns.
        method: One of 'none', 'ma_detrend', 'phase_balance', 'mad_clean', 'combined'.
        params: Method-specific parameters. For 'combined', can contain
                sub-dicts keyed by method name.
        logger: Optional logger for status messages.

    Returns:
        Normalized DataFrame.
    """
    if params is None:
        params = {}

    methods = {
        'ma_detrend': detrend_moving_average,
        'phase_balance': balance_phases,
        'mad_clean': mad_outlier_cleaning,
    }

    if method == 'none':
        return data.copy()

    if method == 'combined':
        # Apply in sequence: MA detrending → phase balancing → MAD cleaning
        if logger:
            logger.info("  Applying combined normalization: MA detrend → phase balance → MAD clean")
        result = detrend_moving_average(data, **params.get('ma_detrend', {}))
        result = balance_phases(result, **params.get('phase_balance', {}))
        result = mad_outlier_cleaning(result, **params.get('mad_clean', {}))
        return result

    if method not in methods:
        raise ValueError(f"Unknown normalization method: '{method}'. "
                         f"Available: ['none'] + {list(methods.keys()) + ['combined']}")

    if logger:
        logger.info(f"  Applying normalization: {method}")
    return methods[method](data, **params.get(method, {}))


def preprocess_normalize(
    input_path: str,
    house_id: str,
    output_path: str,
    method: str,
    params: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Load raw monthly data, normalize, and save to preprocessed directory.

    Processes each monthly pkl file independently, applies normalization,
    and saves to {output_path}/preprocessed/{house_id}/.

    Args:
        input_path: Base input directory (e.g., INPUT/HouseholdData).
        house_id: House identifier.
        output_path: Experiment output directory.
        method: Normalization method name.
        params: Method-specific parameters.
        logger: Logger instance.

    Returns:
        Path to preprocessed parent directory (to override RAW_INPUT_DIRECTORY).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Find raw data
    raw_path = find_house_data_path(input_path, house_id)

    # Create preprocessed output directory
    preprocessed_dir = Path(output_path) / "preprocessed" / house_id
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Get monthly files
    raw_path = Path(raw_path)
    if raw_path.is_dir():
        monthly_files = sorted(raw_path.glob("*.pkl"))
    else:
        monthly_files = [raw_path]

    if not monthly_files:
        raise FileNotFoundError(f"No pkl files found for house {house_id} at {raw_path}")

    logger.info(f"Normalizing {len(monthly_files)} monthly files with method '{method}'")

    total_outliers = 0
    for monthly_file in monthly_files:
        data = load_power_data(monthly_file)

        rows_before = len(data)
        data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)

        if data.empty:
            logger.info(f"  Skipping {monthly_file.name}: empty after dropping NaN rows")
            continue

        # Apply normalization
        normalized = apply_normalization(data, method, params, logger=None)

        # Save original diffs for per-timestamp threshold detection.
        # Detection will use these to compare against thresholds (exact, not approximated),
        # while using the normalized signal for magnitude recording and segmentation.
        for phase in PHASES:
            if phase in data.columns:
                normalized[f'{phase}_orig_diff'] = data[phase].diff()

        # Count changes for logging (MAD cleaning)
        if method in ('mad_clean', 'combined'):
            for phase in PHASES:
                changed = (normalized[phase] != data[phase]).sum()
                total_outliers += changed

        # Save with same filename to preprocessed directory
        output_file = preprocessed_dir / monthly_file.name
        normalized.to_pickle(output_file)

    if method in ('mad_clean', 'combined') and total_outliers > 0:
        logger.info(f"  MAD cleaning replaced {total_outliers} outlier points across all months")

    logger.info(f"Normalized data saved to {preprocessed_dir}")

    # Return parent directory (preprocessed/) so find_house_data_path finds {house_id}/ subfolder
    preprocessed_base = str(Path(output_path) / "preprocessed")
    return preprocessed_base


def compute_threshold_scaling(
    original_input_path: str,
    house_id: str,
    preprocessed_path: str,
    min_diff_watts: int = 500,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Compute how much normalization attenuated event-sized diffs.

    Compares abs(diff()) between original and normalized signals for each
    phase, focusing on diffs >= min_diff_watts (i.e., real events).

    Returns a scaling factor in [0.5, 1.0] to multiply thresholds by.
    If normalization didn't change diffs significantly, returns 1.0.

    Args:
        original_input_path: Base directory with original (raw) house data.
        house_id: House identifier.
        preprocessed_path: Base directory with normalized house data.
        min_diff_watts: Minimum abs(diff) to consider as an event (default 500W).
        logger: Optional logger.

    Returns:
        Scaling factor (0.5 <= scale <= 1.0).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Find original and normalized data paths
    orig_path = find_house_data_path(original_input_path, house_id)
    norm_path = find_house_data_path(preprocessed_path, house_id)

    orig_path = Path(orig_path)
    norm_path = Path(norm_path)

    # Get monthly files
    if orig_path.is_dir():
        orig_files = sorted(orig_path.glob("*.pkl"))
    else:
        orig_files = [orig_path]

    if norm_path.is_dir():
        norm_files = sorted(norm_path.glob("*.pkl"))
    else:
        norm_files = [norm_path]

    # Build lookup by filename for matching
    norm_lookup = {f.name: f for f in norm_files}

    all_ratios = []

    for orig_file in orig_files:
        norm_file = norm_lookup.get(orig_file.name)
        if norm_file is None:
            continue

        orig_data = load_power_data(orig_file)
        norm_data = load_power_data(norm_file)

        if orig_data.empty or norm_data.empty:
            continue

        for phase in PHASES:
            if phase not in orig_data.columns or phase not in norm_data.columns:
                continue

            orig_diff = orig_data[phase].diff().abs()
            norm_diff = norm_data[phase].diff().abs()

            # Filter to event-sized diffs in the original signal
            mask = orig_diff >= min_diff_watts
            if not mask.any():
                continue

            orig_vals = orig_diff[mask].values
            norm_vals = norm_diff[mask].values

            # Compute per-timestamp ratios (avoid division by zero)
            valid = orig_vals > 0
            if valid.any():
                ratios = norm_vals[valid] / orig_vals[valid]
                all_ratios.extend(ratios.tolist())

    if not all_ratios:
        logger.info(f"  No diffs >= {min_diff_watts}W found, no threshold adaptation")
        return 1.0

    scale = float(np.median(all_ratios))

    # Clamp to [0.5, 1.0]
    if scale < 0.5:
        logger.warning(f"  Threshold scale {scale:.3f} is very low (< 0.5), clamping to 0.5")
        scale = 0.5
    elif scale > 1.0:
        scale = 1.0

    if scale < 0.7:
        logger.warning(f"  Normalization significantly attenuated diffs (scale={scale:.3f})")

    logger.info(f"  Threshold scaling: {scale:.3f} (from {len(all_ratios)} event-sized diffs)")
    return scale
