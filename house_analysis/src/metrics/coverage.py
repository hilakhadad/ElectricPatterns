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

    # Date range and span-based coverage
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        ts = data['timestamp']
        metrics['days_span'] = (ts.max() - ts.min()).days + 1

        # Span-based coverage: actual data rows / expected minutes in time range
        # Consistent with disaggregation_analysis dynamic report denominator
        expected_minutes = int((ts.max() - ts.min()).total_seconds() / 60) + 1
        actual_minutes = len(data)
        metrics['expected_minutes'] = expected_minutes
        metrics['actual_minutes'] = actual_minutes
        metrics['coverage_ratio'] = min(actual_minutes / expected_minutes, 1.0) if expected_minutes > 0 else 1.0
        metrics['no_data_pct'] = round((1 - metrics['coverage_ratio']) * 100, 1)

    # NaN tracking per phase (within loaded rows)
    nan_pcts = []
    for col in phase_cols:
        if col in data.columns:
            total = len(data)
            nan_pcts.append(data[col].isna().sum() / total * 100 if total > 0 else 0)

    metrics['avg_nan_pct'] = np.mean(nan_pcts) if nan_pcts else 0

    # Time gaps analysis
    if 'timestamp' in data.columns and len(data) > 1:
        time_diffs = data['timestamp'].diff().dt.total_seconds()
        metrics['max_gap_minutes'] = time_diffs.max() / 60
        metrics['pct_gaps_over_2min'] = (time_diffs > 120).sum() / len(time_diffs) * 100

    # Per-phase max consecutive NaN gap (in minutes, assuming 1-min resolution)
    for col in phase_cols:
        if col not in data.columns:
            continue
        is_nan = data[col].isna()
        if not is_nan.any():
            metrics[f'{col}_max_nan_gap_minutes'] = 0
            continue
        # Find consecutive NaN streaks
        groups = (is_nan != is_nan.shift()).cumsum()
        nan_streaks = is_nan.groupby(groups).sum()
        metrics[f'{col}_max_nan_gap_minutes'] = int(nan_streaks.max())

    # Duplicate timestamps
    if 'timestamp' in data.columns:
        duplicate_count = data['timestamp'].duplicated(keep=False).sum()
        metrics['duplicate_timestamps_count'] = int(duplicate_count)
        metrics['has_duplicate_timestamps'] = duplicate_count > 0

    return metrics
