"""
Coverage metrics for household data.

Calculates data availability and completeness metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_coverage_metrics(data: pd.DataFrame, phase_cols: list = None) -> Dict[str, Any]:
    """
    Calculate coverage and completeness metrics for household data.

    Args:
        data: DataFrame with timestamp column and phase power columns
        phase_cols: List of phase column names (default: ['w1', 'w2', 'w3'] or ['1', '2', '3'])

    Returns:
        Dictionary with coverage metrics
    """
    if phase_cols is None:
        # Try common column naming conventions
        if 'w1' in data.columns:
            phase_cols = ['w1', 'w2', 'w3']
        elif '1' in data.columns:
            phase_cols = ['1', '2', '3']
        else:
            phase_cols = [c for c in data.columns if c not in ['timestamp', 'sum']]

    metrics = {}

    # Date range
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        metrics['days_span'] = (data['timestamp'].max() - data['timestamp'].min()).days + 1

    # Coverage per phase and NaN tracking
    phase_coverages = []
    nan_pcts = []
    for col in phase_cols:
        if col in data.columns:
            total = len(data)
            non_zero = (data[col] > 0).sum()
            phase_coverages.append(non_zero / total if total > 0 else 0)
            nan_pcts.append(data[col].isna().sum() / total * 100 if total > 0 else 0)

    metrics['coverage_ratio'] = np.mean(phase_coverages) if phase_coverages else 0
    metrics['avg_nan_pct'] = np.mean(nan_pcts) if nan_pcts else 0

    # Time gaps analysis
    if 'timestamp' in data.columns and len(data) > 1:
        time_diffs = data['timestamp'].diff().dt.total_seconds()
        metrics['max_gap_minutes'] = time_diffs.max() / 60
        metrics['pct_gaps_over_2min'] = (time_diffs > 120).sum() / len(time_diffs) * 100

    # Duplicate timestamps
    if 'timestamp' in data.columns:
        duplicate_count = data['timestamp'].duplicated(keep=False).sum()
        metrics['duplicate_timestamps_count'] = int(duplicate_count)
        metrics['has_duplicate_timestamps'] = duplicate_count > 0

    return metrics
