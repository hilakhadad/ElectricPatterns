"""
Power statistics metrics for household data.

Calculates statistical measures of power consumption.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def calculate_power_statistics(data: pd.DataFrame, phase_cols: list = None) -> Dict[str, Any]:
    """
    Calculate power consumption statistics for household data.

    Args:
        data: DataFrame with phase power columns
        phase_cols: List of phase column names

    Returns:
        Dictionary with power statistics
    """
    logger.debug("calculate_power_statistics: %d rows", len(data))
    if phase_cols is None:
        if 'w1' in data.columns:
            phase_cols = ['w1', 'w2', 'w3']
        elif '1' in data.columns:
            phase_cols = ['1', '2', '3']
        else:
            phase_cols = [c for c in data.columns if c not in ['timestamp', 'sum']]

    metrics = {}

    for col in phase_cols:
        if col not in data.columns:
            continue

        phase_data = data[col].dropna()
        if len(phase_data) == 0:
            continue

        prefix = f'phase_{col}'

        # Mean power (used by phase balance chart, phase power chart)
        metrics[f'{prefix}_mean'] = phase_data.mean()
        metrics[f'{prefix}_max'] = phase_data.max()

        # Power range distribution (used by power distribution chart, histogram)
        total = len(phase_data)
        metrics[f'{prefix}_share_0_100'] = (phase_data < 100).sum() / total
        metrics[f'{prefix}_share_100_500'] = ((phase_data >= 100) & (phase_data < 500)).sum() / total
        metrics[f'{prefix}_share_500_1000'] = ((phase_data >= 500) & (phase_data < 1000)).sum() / total
        metrics[f'{prefix}_share_1000_2000'] = ((phase_data >= 1000) & (phase_data < 2000)).sum() / total
        metrics[f'{prefix}_share_2000_plus'] = (phase_data >= 2000).sum() / total

    # High-power density: % of time above each pipeline threshold
    PIPELINE_THRESHOLDS = [800, 1100, 1500, 2000]
    phase_above_800 = []
    for col in phase_cols:
        if col not in data.columns:
            continue
        phase_data = data[col].dropna()
        if len(phase_data) == 0:
            continue
        prefix = f'phase_{col}'
        total = len(phase_data)
        for th in PIPELINE_THRESHOLDS:
            above_pct = (phase_data >= th).sum() / total
            metrics[f'{prefix}_above_{th}_pct'] = above_pct
        phase_above_800.append(metrics[f'{prefix}_above_800_pct'])
        # Energy concentration: what % of total energy comes from minutes >= 2000W
        total_energy = phase_data.sum()
        if total_energy > 0:
            metrics[f'{prefix}_above_2000_energy_pct'] = phase_data[phase_data >= 2000].sum() / total_energy
        else:
            metrics[f'{prefix}_above_2000_energy_pct'] = 0.0

    # Aggregate: average across phases of above_800_pct
    metrics['high_power_density'] = np.mean(phase_above_800) if phase_above_800 else 0.0

    # Total power (sum of phases)
    sum_cols = [c for c in phase_cols if c in data.columns]
    if sum_cols:
        total_power = data[sum_cols].sum(axis=1)
        metrics['total_mean'] = total_power.mean()
        metrics['total_max'] = total_power.max()

    # Phase balance
    if len(sum_cols) >= 2:
        phase_means = [data[c].mean() for c in sum_cols if c in data.columns]
        if min(phase_means) > 0:
            metrics['phase_balance_ratio'] = max(phase_means) / min(phase_means)
        else:
            metrics['phase_balance_ratio'] = float('inf')

        metrics['active_phases'] = sum(1 for m in phase_means if m > 50)  # >50W considered active

    logger.debug("calculate_power_statistics: total_mean=%.1f, high_power_density=%.4f",
                 metrics.get('total_mean', 0), metrics.get('high_power_density', 0))
    return metrics


def calculate_phase_imbalance(data: pd.DataFrame, phase_cols: list = None) -> Dict[str, Any]:
    """
    Calculate per-minute phase imbalance score.

    Imbalance = std(w1, w2, w3) / mean(w1, w2, w3) per minute.
    Score of 0 = perfectly balanced, higher = more imbalanced.

    Args:
        data: DataFrame with phase power columns
        phase_cols: List of phase column names

    Returns:
        Dictionary with imbalance statistics
    """
    logger.debug("calculate_phase_imbalance: %d rows", len(data))
    if phase_cols is None:
        if 'w1' in data.columns:
            phase_cols = ['w1', 'w2', 'w3']
        elif '1' in data.columns:
            phase_cols = ['1', '2', '3']
        else:
            phase_cols = [c for c in data.columns if c not in ['timestamp', 'sum']]

    sum_cols = [c for c in phase_cols if c in data.columns]
    if len(sum_cols) < 2:
        return {
            'imbalance_mean': 0.0,
            'imbalance_median': 0.0,
            'imbalance_p95': 0.0,
            'imbalance_max': 0.0,
            'high_imbalance_pct': 0.0,
            'imbalance_label': 'balanced',
        }

    phase_data = data[sum_cols].dropna(how='all')
    if len(phase_data) == 0:
        return {
            'imbalance_mean': 0.0,
            'imbalance_median': 0.0,
            'imbalance_p95': 0.0,
            'imbalance_max': 0.0,
            'high_imbalance_pct': 0.0,
            'imbalance_label': 'balanced',
        }

    row_means = phase_data.mean(axis=1)
    row_stds = phase_data.std(axis=1, ddof=0)

    # Avoid division by zero: where mean is 0, imbalance is 0
    imbalance = row_stds / row_means.replace(0, np.nan)
    imbalance = imbalance.fillna(0.0)

    imb_mean = float(imbalance.mean())
    imb_median = float(imbalance.median())
    imb_p95 = float(imbalance.quantile(0.95))
    imb_max = float(imbalance.max())
    high_pct = float((imbalance > 0.5).sum() / len(imbalance) * 100)

    if imb_mean < 0.2:
        label = 'balanced'
    elif imb_mean < 0.5:
        label = 'moderate'
    else:
        label = 'imbalanced'

    return {
        'imbalance_mean': imb_mean,
        'imbalance_median': imb_median,
        'imbalance_p95': imb_p95,
        'imbalance_max': imb_max,
        'high_imbalance_pct': high_pct,
        'imbalance_label': label,
    }


def calculate_power_ranges(data: pd.DataFrame, phase_cols: list = None,
                           ranges: List[tuple] = None) -> Dict[str, Any]:
    """
    Calculate time spent in different power ranges.

    Args:
        data: DataFrame with phase power columns
        phase_cols: List of phase column names
        ranges: List of (min, max) tuples for power ranges

    Returns:
        Dictionary with time in each range
    """
    logger.debug("calculate_power_ranges: %d rows", len(data))
    if ranges is None:
        ranges = [(0, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, float('inf'))]

    if phase_cols is None:
        if 'w1' in data.columns:
            phase_cols = ['w1', 'w2', 'w3']
        elif '1' in data.columns:
            phase_cols = ['1', '2', '3']
        else:
            phase_cols = [c for c in data.columns if c not in ['timestamp', 'sum']]

    metrics = {}

    for col in phase_cols:
        if col not in data.columns:
            continue

        phase_data = data[col].dropna()
        total = len(phase_data)
        if total == 0:
            continue

        for min_val, max_val in ranges:
            range_name = f'{int(min_val)}_{int(max_val) if max_val != float("inf") else "inf"}'
            count = ((phase_data >= min_val) & (phase_data < max_val)).sum()
            metrics[f'{col}_range_{range_name}_pct'] = count / total * 100
            metrics[f'{col}_range_{range_name}_minutes'] = count  # Assuming 1-min resolution

    return metrics
