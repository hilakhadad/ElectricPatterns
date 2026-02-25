"""
Evaluation metrics for segmentation results.

Calculates power and time metrics comparing original vs remaining consumption.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def calculate_phase_metrics(
    baseline_original: pd.Series,
    current_remaining: pd.Series,
    prev_remaining: Optional[pd.Series],
    threshold: int,
    run_number: int,
    logger=None
) -> Dict:
    """
    Calculate evaluation metrics for a single phase.

    Args:
        baseline_original: Original power from run_0
        current_remaining: Current remaining power
        prev_remaining: Previous run remaining power (None for run_0)
        threshold: Power threshold in watts
        run_number: Current run number

    Returns:
        Dict with all metrics
    """
    # Valid data mask
    valid_mask = baseline_original.notna() & current_remaining.notna()
    above_th_mask = (baseline_original > threshold) & valid_mask

    # === POWER METRICS ===
    total_power_all = baseline_original.fillna(0).sum()
    total_power_above_th = baseline_original[above_th_mask].sum()

    # Cumulative explained power
    explained_power_cumulative = (
        baseline_original[above_th_mask] - current_remaining[above_th_mask]
    ).clip(lower=0).sum()

    explained_power_cumulative_pct = (
        (explained_power_cumulative / total_power_above_th * 100)
        if total_power_above_th > 0 else 0.0
    )

    # Iteration-specific power
    if run_number == 0 or prev_remaining is None:
        explained_power_iteration = explained_power_cumulative
        explained_power_iteration_pct = explained_power_cumulative_pct
    else:
        valid_iter_mask = above_th_mask & prev_remaining.notna()
        explained_power_iteration = (
            prev_remaining[valid_iter_mask] - current_remaining[valid_iter_mask]
        ).clip(lower=0).sum()
        explained_power_iteration_pct = (
            (explained_power_iteration / total_power_above_th * 100)
            if total_power_above_th > 0 else 0.0
        )

    # === NEGATIVE VALUES (sanity check) ===
    negative_mask = current_remaining < 0
    minutes_negative = negative_mask.sum()
    power_negative = current_remaining[negative_mask].sum()

    # === MISSING VALUES ===
    minutes_missing = (baseline_original.notna() & current_remaining.isna()).sum()

    # === TIME METRICS ===
    minutes_above_th = above_th_mask.sum()
    minutes_below_th_cumulative = (above_th_mask & (current_remaining <= threshold)).sum()

    time_pct_cumulative = (
        (minutes_below_th_cumulative / minutes_above_th * 100)
        if minutes_above_th > 0 else 0.0
    )

    # Iteration-specific time
    if run_number == 0 or prev_remaining is None:
        minutes_below_th_iteration = minutes_below_th_cumulative
        time_pct_iteration = time_pct_cumulative
    else:
        valid_time_mask = above_th_mask & prev_remaining.notna()
        minutes_below_th_iteration = (
            valid_time_mask & (prev_remaining > threshold) & (current_remaining <= threshold)
        ).sum()
        time_pct_iteration = (
            (minutes_below_th_iteration / minutes_above_th * 100)
            if minutes_above_th > 0 else 0.0
        )

    if logger:
        logger.debug(
            f"Phase metrics: explained_pwr={explained_power_iteration_pct:.1f}%, "
            f"explained_time={time_pct_iteration:.1f}%, negative={minutes_negative} min"
        )

    return {
        'total_power_all': total_power_all,
        'total_power_above_th': total_power_above_th,
        'explained_power': explained_power_iteration,
        'explained_power_pct': round(explained_power_iteration_pct, 2),
        'explained_power_cumulative': explained_power_cumulative,
        'explained_power_cumulative_pct': round(explained_power_cumulative_pct, 2),
        'minutes_above_th': minutes_above_th,
        'minutes_explained': minutes_below_th_iteration,
        'minutes_explained_pct': round(time_pct_iteration, 2),
        'minutes_explained_cumulative': minutes_below_th_cumulative,
        'minutes_explained_cumulative_pct': round(time_pct_cumulative, 2),
        'threshold_explanation_pct': round(time_pct_cumulative, 2),  # Alias for clarity
        'minutes_negative': minutes_negative,
        'power_negative': round(power_negative, 2),
        'minutes_missing': minutes_missing,
    }


def save_negative_values(
    data: pd.DataFrame,
    remaining_values: pd.Series,
    negative_mask: pd.Series,
    house_id: str,
    run_number: int,
    phase: str,
    output_base_path: str
) -> None:
    """Save negative value details to experiment-level CSV for debugging."""
    negative_values_path = Path(output_base_path) / "negative_values.csv"

    negative_rows = data.loc[negative_mask, ['timestamp']].copy()
    negative_rows['house_id'] = house_id
    negative_rows['run_number'] = run_number
    negative_rows['phase'] = phase
    negative_rows['remaining_power'] = remaining_values[negative_mask].values

    file_exists = negative_values_path.exists()
    negative_rows.to_csv(negative_values_path, mode='a', header=not file_exists, index=False)
