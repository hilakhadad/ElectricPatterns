"""
Data quality metrics for household data.

Detects anomalies, outliers, and potential data issues.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def calculate_data_quality_metrics(data: pd.DataFrame, phase_cols: list = None) -> Dict[str, Any]:
    """
    Calculate data quality metrics for household data.

    Args:
        data: DataFrame with timestamp and phase power columns
        phase_cols: List of phase column names

    Returns:
        Dictionary with data quality metrics
    """
    if phase_cols is None:
        if 'w1' in data.columns:
            phase_cols = ['w1', 'w2', 'w3']
        elif '1' in data.columns:
            phase_cols = ['1', '2', '3']
        else:
            phase_cols = [c for c in data.columns if c not in ['timestamp', 'sum']]

    metrics = {}

    # Negative values (should not exist in power data)
    for col in phase_cols:
        if col not in data.columns:
            continue

        negative_count = (data[col] < 0).sum()
        metrics[f'{col}_negative_count'] = int(negative_count)
        metrics[f'{col}_negative_pct'] = negative_count / len(data) * 100 if len(data) > 0 else 0

    # Zero values (potential metering issues if too many)
    for col in phase_cols:
        if col not in data.columns:
            continue

        zero_count = (data[col] == 0).sum()
        metrics[f'{col}_zero_count'] = int(zero_count)
        metrics[f'{col}_zero_pct'] = zero_count / len(data) * 100 if len(data) > 0 else 0

    # Outliers detection (values beyond 3 standard deviations)
    for col in phase_cols:
        if col not in data.columns:
            continue

        phase_data = data[col].dropna()
        if len(phase_data) == 0:
            continue

        mean = phase_data.mean()
        std = phase_data.std()
        if std > 0:
            z_scores = np.abs((phase_data - mean) / std)
            outliers_3sd = (z_scores > 3).sum()
            outliers_4sd = (z_scores > 4).sum()
            metrics[f'{col}_outliers_3sd'] = int(outliers_3sd)
            metrics[f'{col}_outliers_4sd'] = int(outliers_4sd)
            metrics[f'{col}_outliers_3sd_pct'] = outliers_3sd / len(phase_data) * 100
        else:
            metrics[f'{col}_outliers_3sd'] = 0
            metrics[f'{col}_outliers_4sd'] = 0
            metrics[f'{col}_outliers_3sd_pct'] = 0

    # Sudden jumps detection (large changes between consecutive readings)
    for col in phase_cols:
        if col not in data.columns:
            continue

        phase_data = data[col].dropna()
        if len(phase_data) < 2:
            continue

        diffs = phase_data.diff().abs()
        large_jumps_1000 = (diffs > 1000).sum()
        large_jumps_2000 = (diffs > 2000).sum()
        large_jumps_5000 = (diffs > 5000).sum()

        metrics[f'{col}_jumps_over_1000W'] = int(large_jumps_1000)
        metrics[f'{col}_jumps_over_2000W'] = int(large_jumps_2000)
        metrics[f'{col}_jumps_over_5000W'] = int(large_jumps_5000)

        # Max single jump
        metrics[f'{col}_max_jump'] = diffs.max()

    # Duplicate timestamps
    if 'timestamp' in data.columns:
        dup_timestamps = data['timestamp'].duplicated().sum()
        metrics['duplicate_timestamps'] = int(dup_timestamps)

    # Overall quality score (0-100)
    quality_score = 100

    # Deduct for missing data
    for col in phase_cols:
        if f'{col}_missing_pct' in metrics:
            quality_score -= min(metrics.get(f'{col}_missing_pct', 0), 10)

    # Deduct for negative values
    total_negatives = sum(metrics.get(f'{col}_negative_count', 0) for col in phase_cols)
    if total_negatives > 0:
        quality_score -= min(total_negatives / 10, 20)

    # Deduct for excessive zeros
    for col in phase_cols:
        zero_pct = metrics.get(f'{col}_zero_pct', 0)
        if zero_pct > 50:  # More than 50% zeros is suspicious
            quality_score -= 10

    # Deduct for outliers
    total_outliers = sum(metrics.get(f'{col}_outliers_3sd', 0) for col in phase_cols)
    quality_score -= min(total_outliers / 100, 10)

    metrics['quality_score'] = max(0, min(100, quality_score))

    return metrics


def detect_anomalous_periods(data: pd.DataFrame, phase_cols: list = None,
                              window_hours: int = 24) -> List[Dict[str, Any]]:
    """
    Detect periods with anomalous behavior.

    Args:
        data: DataFrame with timestamp and phase power columns
        phase_cols: List of phase column names
        window_hours: Size of rolling window in hours

    Returns:
        List of anomalous periods with details
    """
    if 'timestamp' not in data.columns:
        return []

    if phase_cols is None:
        if 'w1' in data.columns:
            phase_cols = ['w1', 'w2', 'w3']
        elif '1' in data.columns:
            phase_cols = ['1', '2', '3']
        else:
            phase_cols = [c for c in data.columns if c not in ['timestamp', 'sum']]

    data = data.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.set_index('timestamp')

    anomalies = []

    for col in phase_cols:
        if col not in data.columns:
            continue

        # Rolling statistics
        window = f'{window_hours}H'
        rolling_mean = data[col].rolling(window).mean()
        rolling_std = data[col].rolling(window).std()

        # Detect periods where values deviate significantly from rolling stats
        if rolling_std.mean() > 0:
            z_scores = (data[col] - rolling_mean) / rolling_std.replace(0, np.nan)

            # Find periods with sustained anomalies
            anomaly_mask = z_scores.abs() > 3
            if anomaly_mask.any():
                # Group consecutive anomalies
                anomaly_groups = (anomaly_mask != anomaly_mask.shift()).cumsum()
                for group_id, group in data[anomaly_mask].groupby(anomaly_groups[anomaly_mask]):
                    if len(group) >= 5:  # At least 5 minutes of anomaly
                        anomalies.append({
                            'phase': col,
                            'start': group.index.min().isoformat(),
                            'end': group.index.max().isoformat(),
                            'duration_minutes': len(group),
                            'mean_value': group[col].mean(),
                            'expected_mean': rolling_mean.loc[group.index].mean(),
                            'type': 'sustained_deviation'
                        })

    return anomalies
