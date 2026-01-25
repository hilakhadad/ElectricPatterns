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

    # Basic counts
    metrics['total_rows'] = len(data)

    # Date range
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        metrics['start_date'] = data['timestamp'].min().isoformat()
        metrics['end_date'] = data['timestamp'].max().isoformat()
        metrics['days_span'] = (data['timestamp'].max() - data['timestamp'].min()).days + 1

        # Expected rows (assuming 1-minute resolution)
        expected_rows = metrics['days_span'] * 24 * 60
        metrics['expected_rows'] = expected_rows
        metrics['coverage_ratio'] = metrics['total_rows'] / expected_rows if expected_rows > 0 else 0

        # Per-month coverage
        data['year_month'] = data['timestamp'].dt.to_period('M')
        monthly_counts = data.groupby('year_month').size()
        metrics['months_count'] = len(monthly_counts)
        metrics['months_list'] = [str(m) for m in monthly_counts.index.tolist()]

        # Days with data per month
        data['date'] = data['timestamp'].dt.date
        days_per_month = data.groupby('year_month')['date'].nunique()
        metrics['avg_days_per_month'] = days_per_month.mean()
        metrics['min_days_in_month'] = days_per_month.min()
        metrics['max_days_in_month'] = days_per_month.max()

    # Missing values per phase
    for col in phase_cols:
        if col in data.columns:
            missing = data[col].isna().sum()
            metrics[f'{col}_missing_count'] = int(missing)
            metrics[f'{col}_missing_pct'] = missing / len(data) * 100 if len(data) > 0 else 0

    # Time gaps analysis
    if 'timestamp' in data.columns and len(data) > 1:
        time_diffs = data['timestamp'].diff().dt.total_seconds()
        metrics['median_step_seconds'] = time_diffs.median()
        metrics['max_gap_seconds'] = time_diffs.max()
        metrics['max_gap_minutes'] = time_diffs.max() / 60

        # Percentage of gaps over thresholds
        metrics['pct_gaps_over_2min'] = (time_diffs > 120).sum() / len(time_diffs) * 100
        metrics['pct_gaps_over_10min'] = (time_diffs > 600).sum() / len(time_diffs) * 100
        metrics['pct_gaps_over_60min'] = (time_diffs > 3600).sum() / len(time_diffs) * 100

    return metrics
