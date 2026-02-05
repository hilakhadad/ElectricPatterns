"""
Data quality metrics for household data.

Detects anomalies, outliers, and potential data issues.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def calculate_data_quality_metrics(data: pd.DataFrame, phase_cols: list = None,
                                    coverage_ratio: float = None,
                                    days_span: int = None) -> Dict[str, Any]:
    """
    Calculate data quality metrics for household data.

    Args:
        data: DataFrame with timestamp and phase power columns
        phase_cols: List of phase column names
        coverage_ratio: Optional coverage ratio (0-1) to include in quality score
        days_span: Optional number of days of data

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
    metrics['coverage_ratio'] = coverage_ratio if coverage_ratio is not None else 1.0

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

    # Detect dead/faulty phases using relative threshold (same logic as experiment_analysis)
    # A phase is considered damaged if its total power is less than 1% of the maximum phase's power
    dead_phases = []
    phase_powers = {}

    for col in phase_cols:
        if col not in data.columns:
            phase_powers[col] = 0
            continue
        phase_data = data[col].dropna()
        if len(phase_data) == 0:
            phase_powers[col] = 0
        else:
            phase_powers[col] = phase_data.sum()

    # Find max power among phases
    max_power = max(phase_powers.values()) if phase_powers else 0
    threshold_ratio = 0.01  # 1% threshold (same as experiment_analysis)

    for col, power in phase_powers.items():
        ratio = power / max_power if max_power > 0 else 0
        if ratio < threshold_ratio:
            dead_phases.append(col)

    metrics['dead_phases'] = dead_phases
    metrics['phase_powers'] = phase_powers
    metrics['has_dead_phase'] = len(dead_phases) > 0
    metrics['active_phase_count'] = len(phase_cols) - len(dead_phases)

    # Store days_span in metrics
    metrics['days_span'] = days_span if days_span is not None else 0

    # Overall quality score (0-100)
    # PRIMARY FACTORS: Coverage (60%) and Days of data (30%)
    # SECONDARY FACTORS: Other issues (10%)

    quality_score = 0

    # ===== PRIMARY FACTOR 1: Coverage (up to 60 points) =====
    coverage = metrics.get('coverage_ratio', 1.0)
    # Linear scale: 0% coverage = 0 points, 100% coverage = 60 points
    coverage_score = coverage * 60
    quality_score += coverage_score
    metrics['coverage_score_contribution'] = coverage_score

    # ===== PRIMARY FACTOR 2: Days of data (up to 30 points) =====
    days = days_span if days_span is not None else 0
    days_score = 0

    if days >= 730:  # 2+ years - excellent
        days_score = 30
    elif days >= 365:  # 1-2 years - good
        # Linear scale from 20 to 30 points
        days_score = 20 + (days - 365) / 365 * 10
    elif days >= 180:  # 6 months to 1 year - acceptable
        # Linear scale from 10 to 20 points
        days_score = 10 + (days - 180) / 185 * 10
    elif days >= 30:  # 1-6 months - limited
        # Linear scale from 5 to 10 points
        days_score = 5 + (days - 30) / 150 * 5
    else:  # Less than 1 month - very limited
        days_score = days / 30 * 5

    quality_score += days_score
    metrics['days_score_contribution'] = days_score

    # ===== SECONDARY FACTORS: Other issues (up to 10 points) =====
    # Start with 10 and deduct for issues
    secondary_score = 10

    # Deduct for dead phases (up to 3 points)
    if len(dead_phases) > 0:
        secondary_score -= min(len(dead_phases) * 1, 3)

    # Deduct for negative values (up to 2 points)
    total_negatives = sum(metrics.get(f'{col}_negative_count', 0) for col in phase_cols)
    if total_negatives > 0:
        secondary_score -= min(total_negatives / 100, 2)

    # Deduct for outliers (up to 2 points)
    outlier_pct = sum(metrics.get(f'{col}_outliers_3sd_pct', 0) for col in phase_cols) / len(phase_cols) if phase_cols else 0
    if outlier_pct > 1:
        secondary_score -= min(outlier_pct * 0.5, 2)

    # Deduct for large jumps (up to 2 points)
    total_large_jumps = sum(metrics.get(f'{col}_jumps_over_2000W', 0) for col in phase_cols)
    if total_large_jumps > 500:
        secondary_score -= min((total_large_jumps - 500) / 500, 2)

    # Deduct for duplicate timestamps (up to 1 point)
    dup_ts = metrics.get('duplicate_timestamps', 0)
    if dup_ts > 0:
        secondary_score -= min(dup_ts / 100, 1)

    secondary_score = max(0, secondary_score)
    quality_score += secondary_score
    metrics['secondary_score_contribution'] = secondary_score

    # ===== PHASE BALANCE PENALTY =====
    # Calculate phase balance ratio and penalize unbalanced phases
    phase_means = []
    for col in phase_cols:
        if col in data.columns:
            phase_mean = data[col].mean()
            phase_means.append(phase_mean)

    balance_penalty = 0
    if len(phase_means) >= 2 and min(phase_means) > 0:
        balance_ratio = max(phase_means) / min(phase_means)
        metrics['phase_balance_ratio'] = balance_ratio

        # Penalty scale:
        # ratio 1-2: no penalty (normal variation)
        # ratio 2-3: 0-5 points penalty
        # ratio 3-5: 5-10 points penalty
        # ratio 5+: 10-15 points penalty (capped)
        if balance_ratio > 5:
            balance_penalty = 10 + min((balance_ratio - 5), 5)
        elif balance_ratio > 3:
            balance_penalty = 5 + (balance_ratio - 3) * 2.5
        elif balance_ratio > 2:
            balance_penalty = (balance_ratio - 2) * 5
        # else: no penalty for ratio <= 2
    else:
        metrics['phase_balance_ratio'] = 1.0  # Default if can't calculate

    metrics['balance_penalty'] = balance_penalty
    quality_score -= balance_penalty

    # Ensure bounds
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
