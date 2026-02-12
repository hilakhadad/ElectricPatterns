"""
Data quality metrics for household data.

Detects anomalies, outliers, and potential data issues.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def calculate_data_quality_metrics(data: pd.DataFrame, phase_cols: list = None,
                                    coverage_ratio: float = None,
                                    days_span: int = None,
                                    max_gap_minutes: float = None,
                                    pct_gaps_over_2min: float = None,
                                    avg_nan_pct: float = None) -> Dict[str, Any]:
    """
    Calculate data quality metrics for household data.

    Args:
        data: DataFrame with timestamp and phase power columns
        phase_cols: List of phase column names
        coverage_ratio: Optional coverage ratio (0-1) to include in quality score
        days_span: Optional number of days of data
        max_gap_minutes: Optional maximum gap in minutes
        pct_gaps_over_2min: Optional percentage of gaps over 2 minutes
        avg_nan_pct: Optional average NaN percentage across phases

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

    # Negative values count (used by has_negative_values flag)
    for col in phase_cols:
        if col not in data.columns:
            continue
        metrics[f'{col}_negative_count'] = int((data[col] < 0).sum())

    # Outliers detection - 3sd percentage (used by many_outliers flag)
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
            metrics[f'{col}_outliers_3sd_pct'] = (z_scores > 3).sum() / len(phase_data) * 100
        else:
            metrics[f'{col}_outliers_3sd_pct'] = 0

    # Sudden jumps - >2000W count (used by many_large_jumps flag)
    for col in phase_cols:
        if col not in data.columns:
            continue
        phase_data = data[col].dropna()
        if len(phase_data) < 2:
            continue
        diffs = phase_data.diff().abs()
        metrics[f'{col}_jumps_over_2000W'] = int((diffs > 2000).sum())

    # Detect dead/faulty phases using relative threshold
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
    threshold_ratio = 0.01  # 1% threshold

    for col, power in phase_powers.items():
        ratio = power / max_power if max_power > 0 else 0
        if ratio < threshold_ratio:
            dead_phases.append(col)

    metrics['dead_phases'] = dead_phases
    metrics['has_dead_phase'] = len(dead_phases) > 0

    # ===== FAULTY PHASE DETECTION (NaN >= 20%) =====
    # A phase with >= 20% NaN values is considered faulty (תקולה)
    # Houses with faulty phases get quality_label='faulty' instead of numeric score
    FAULTY_NAN_THRESHOLD = 20.0  # percent

    faulty_nan_phases = []
    for col in phase_cols:
        if col not in data.columns:
            continue
        col_nan_pct = data[col].isna().sum() / len(data) * 100 if len(data) > 0 else 0
        metrics[f'{col}_nan_pct'] = col_nan_pct
        if col_nan_pct >= FAULTY_NAN_THRESHOLD:
            faulty_nan_phases.append(col)

    metrics['faulty_nan_phases'] = faulty_nan_phases
    metrics['has_faulty_nan_phase'] = len(faulty_nan_phases) > 0

    # ===== QUALITY SCORING SYSTEM (0-100) =====
    # Based on: Completeness, Gap Quality, Phase Balance, Monthly Balance, Noise Level

    quality_score = 0

    # ===== 1. DATA COMPLETENESS (up to 30 points) =====
    coverage = metrics.get('coverage_ratio', 1.0)
    completeness_score = coverage * 30
    quality_score += completeness_score
    metrics['completeness_score'] = completeness_score

    # ===== 2. GAP QUALITY (up to 20 points) =====
    gap_score = 20

    # Penalty for max gap > 60 min (up to 10 pts)
    max_gap = max_gap_minutes if max_gap_minutes is not None else 0
    if max_gap > 60:
        gap_score -= min((max_gap - 60) / 138, 10)

    # Penalty for high gap percentage (up to 10 pts)
    gap_pct = pct_gaps_over_2min if pct_gaps_over_2min is not None else 0
    if gap_pct > 2:
        gap_score -= min((gap_pct - 2) / 1.8, 10)

    # Penalty for NaN values within existing rows (up to 10 pts)
    nan_pct = avg_nan_pct if avg_nan_pct is not None else 0
    if nan_pct > 1:
        gap_score -= min((nan_pct - 1) / 1.9, 10)

    gap_score = max(0, gap_score)
    quality_score += gap_score
    metrics['gap_score'] = gap_score

    # ===== 3. PHASE BALANCE (up to 15 points) =====
    phase_means = []
    for col in phase_cols:
        if col in data.columns:
            phase_mean = data[col].mean()
            if phase_mean > 0:  # Only include active phases
                phase_means.append(phase_mean)

    if len(phase_means) >= 2 and min(phase_means) > 0:
        balance_ratio = max(phase_means) / min(phase_means)

        # Score based on ratio
        if balance_ratio <= 2:
            balance_score = 15  # Excellent
        elif balance_ratio <= 3:
            balance_score = 10  # Good
        elif balance_ratio <= 5:
            balance_score = 5   # Acceptable
        else:
            balance_score = 0   # Poor
    else:
        balance_score = 15  # Default for single phase or can't calculate

    quality_score += balance_score
    metrics['balance_score'] = balance_score

    # ===== 4. MONTHLY COVERAGE BALANCE (up to 20 points) =====
    monthly_balance_score = 20  # Start with max

    if 'timestamp' in data.columns:
        data_temp = data.copy()
        data_temp['timestamp'] = pd.to_datetime(data_temp['timestamp'])
        data_temp['year_month'] = data_temp['timestamp'].dt.to_period('M')

        monthly_counts = data_temp.groupby('year_month').size()

        if len(monthly_counts) > 1:
            monthly_coverage = []
            for period, count in monthly_counts.items():
                expected = period.days_in_month * 24 * 60  # minutes in month
                monthly_coverage.append(min(count / expected, 1.0))

            monthly_std = np.std(monthly_coverage)
            monthly_balance_score = max(0, 20 * (1 - monthly_std * 2))

    quality_score += monthly_balance_score
    metrics['monthly_balance_score'] = monthly_balance_score

    # ===== 5. LOW NOISE / STABILITY (up to 15 points) =====
    noise_score = 15  # Start with max

    hourly_cvs = []
    if 'timestamp' in data.columns:
        data_temp = data.copy()
        data_temp['timestamp'] = pd.to_datetime(data_temp['timestamp'])
        data_temp['hour'] = data_temp['timestamp'].dt.hour

        for col in phase_cols:
            if col not in data.columns:
                continue

            # Calculate hourly mean and std
            hourly_stats = data_temp.groupby('hour')[col].agg(['mean', 'std'])
            hourly_mean = hourly_stats['mean'].mean()
            hourly_std = hourly_stats['std'].mean()

            if hourly_mean > 0:
                cv = hourly_std / hourly_mean
                hourly_cvs.append(cv)

    if hourly_cvs:
        avg_cv = np.mean(hourly_cvs)

        # Score based on CV (coefficient of variation)
        if 0.3 <= avg_cv <= 0.8:
            noise_score = 15
        elif avg_cv < 0.3:
            noise_score = 10
        elif avg_cv <= 1.5:
            noise_score = 10
        elif avg_cv <= 2.0:
            noise_score = 5
        else:
            noise_score = 0

    quality_score += noise_score
    metrics['noise_score'] = noise_score

    # Ensure bounds
    metrics['quality_score'] = max(0, min(100, quality_score))

    # Mark faulty phases via label (keep numeric score for sorting/comparison)
    if metrics['has_faulty_nan_phase']:
        metrics['quality_label'] = 'faulty'
    else:
        metrics['quality_label'] = None

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
