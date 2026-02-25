"""
Coverage metrics for household data.

Calculates data availability and completeness metrics.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)


def calculate_coverage_metrics(data: pd.DataFrame, phase_cols: list = None) -> Dict[str, Any]:
    """
    Calculate coverage and completeness metrics for household data.

    Args:
        data: DataFrame with timestamp column and phase power columns
        phase_cols: List of phase column names (default: ['w1', 'w2', 'w3'] or ['1', '2', '3'])

    Returns:
        Dictionary with coverage metrics
    """
    logger.debug("calculate_coverage_metrics: %d rows, columns=%s", len(data), list(data.columns))
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
        raw_no_data_pct = (1 - metrics['coverage_ratio']) * 100
        metrics['no_data_pct'] = round(raw_no_data_pct, 1)
        # Track if there are missing minutes even when pct rounds to 0.0%
        metrics['no_data_pct_raw'] = raw_no_data_pct

    # NaN tracking per phase (within loaded rows)
    nan_pcts = []
    for col in phase_cols:
        if col in data.columns:
            total = len(data)
            nan_pcts.append(data[col].isna().sum() / total * 100 if total > 0 else 0)

    metrics['avg_nan_pct'] = np.mean(nan_pcts) if nan_pcts else 0

    # NaN rows: rows that exist but have NaN in at least one phase
    active_cols = [c for c in phase_cols if c in data.columns]
    if active_cols and len(data) > 0:
        nan_any_phase = data[active_cols].isna().any(axis=1)
        metrics['nan_rows_count'] = int(nan_any_phase.sum())
        metrics['nan_rows_pct'] = round(nan_any_phase.sum() / len(data) * 100, 1)
    else:
        metrics['nan_rows_count'] = 0
        metrics['nan_rows_pct'] = 0.0

    # No-data gap vs NaN gap distinction (as % of expected minutes)
    expected = metrics.get('expected_minutes', len(data))
    actual = metrics.get('actual_minutes', len(data))
    no_data_minutes = max(0, expected - actual)  # missing rows entirely
    nan_minutes = metrics['nan_rows_count']  # rows that exist but have NaN
    metrics['no_data_gap_minutes'] = no_data_minutes
    metrics['nan_gap_minutes'] = nan_minutes
    metrics['no_data_gap_pct'] = round(no_data_minutes / expected * 100, 1) if expected > 0 else 0.0
    metrics['nan_gap_pct'] = round(nan_minutes / expected * 100, 1) if expected > 0 else 0.0

    # Time gaps analysis
    if 'timestamp' in data.columns and len(data) > 1:
        time_diffs = data['timestamp'].diff().dt.total_seconds()
        metrics['max_gap_minutes'] = time_diffs.max() / 60
        metrics['pct_gaps_over_2min'] = (time_diffs > 120).sum() / len(time_diffs) * 100

        # No-data gap breakdown: count and max of gaps > 1 minute
        gap_mask = time_diffs > 60  # gaps > 1 minute (missing rows)
        metrics['no_data_gap_count'] = int(gap_mask.sum())
        if gap_mask.any():
            metrics['max_no_data_gap_minutes'] = round(time_diffs[gap_mask].max() / 60, 0)
        else:
            metrics['max_no_data_gap_minutes'] = 0

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

    # Anomaly detection: extreme outlier values per phase
    # A single-phase household reading > 20kW is almost certainly a sensor error
    ANOMALY_THRESHOLD = 20000  # 20kW per single phase
    anomaly_count = 0
    anomaly_phases = {}
    for col in phase_cols:
        if col not in data.columns:
            continue
        phase_data = data[col].dropna()
        phase_anomalies = int((phase_data.abs() > ANOMALY_THRESHOLD).sum())
        if phase_anomalies > 0:
            anomaly_phases[col] = phase_anomalies
        anomaly_count += phase_anomalies

    metrics['anomaly_count'] = anomaly_count
    metrics['anomaly_phases'] = anomaly_phases
    metrics['has_anomalies'] = anomaly_count > 0

    # Max observed value (to flag extreme outliers like 51M watts)
    max_vals = {}
    for col in phase_cols:
        if col in data.columns:
            max_vals[col] = float(data[col].max()) if not data[col].isna().all() else 0
    metrics['phase_max_values'] = max_vals

    logger.debug("calculate_coverage_metrics: coverage_ratio=%.3f, days_span=%s, avg_nan_pct=%.2f",
                 metrics.get('coverage_ratio', 0), metrics.get('days_span', 'N/A'), metrics.get('avg_nan_pct', 0))
    return metrics
