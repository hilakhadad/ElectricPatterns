"""
Power statistics metrics for household data.

Calculates statistical measures of power consumption.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def calculate_power_statistics(data: pd.DataFrame, phase_cols: list = None) -> Dict[str, Any]:
    """
    Calculate power consumption statistics for household data.

    Args:
        data: DataFrame with phase power columns
        phase_cols: List of phase column names

    Returns:
        Dictionary with power statistics
    """
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

        # Power range distribution (used by power distribution chart, histogram)
        total = len(phase_data)
        metrics[f'{prefix}_share_0_100'] = (phase_data < 100).sum() / total
        metrics[f'{prefix}_share_100_500'] = ((phase_data >= 100) & (phase_data < 500)).sum() / total
        metrics[f'{prefix}_share_500_1000'] = ((phase_data >= 500) & (phase_data < 1000)).sum() / total
        metrics[f'{prefix}_share_1000_2000'] = ((phase_data >= 1000) & (phase_data < 2000)).sum() / total
        metrics[f'{prefix}_share_2000_plus'] = (phase_data >= 2000).sum() / total

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

    return metrics


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
