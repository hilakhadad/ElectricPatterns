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
                                    pct_gaps_over_2min: float = None) -> Dict[str, Any]:
    """
    Calculate data quality metrics for household data.

    Args:
        data: DataFrame with timestamp and phase power columns
        phase_cols: List of phase column names
        coverage_ratio: Optional coverage ratio (0-1) to include in quality score
        days_span: Optional number of days of data
        max_gap_minutes: Optional maximum gap in minutes
        pct_gaps_over_2min: Optional percentage of gaps over 2 minutes

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

    # ===== NEW QUALITY SCORING SYSTEM (0-100) =====
    # Based on: Completeness, Gap Quality, Phase Balance, Monthly Balance, Noise Level

    quality_score = 0

    # ===== 1. DATA COMPLETENESS (up to 30 points) =====
    coverage = metrics.get('coverage_ratio', 1.0)
    completeness_score = coverage * 30
    quality_score += completeness_score
    metrics['completeness_score'] = completeness_score

    # ===== 2. GAP QUALITY (up to 20 points) =====
    # Start with 20, deduct for gaps
    gap_score = 20

    # Store gap metrics
    metrics['max_gap_minutes'] = max_gap_minutes if max_gap_minutes is not None else 0
    metrics['pct_gaps_over_2min'] = pct_gaps_over_2min if pct_gaps_over_2min is not None else 0

    # Penalty for max gap > 60 min (up to 10 pts)
    max_gap = max_gap_minutes if max_gap_minutes is not None else 0
    if max_gap > 60:
        # Scale: 60 min = 0 penalty, 1440 min (1 day) = 10 penalty
        gap_penalty = min((max_gap - 60) / 138, 10)
        gap_score -= gap_penalty

    # Penalty for high gap percentage (up to 10 pts)
    gap_pct = pct_gaps_over_2min if pct_gaps_over_2min is not None else 0
    if gap_pct > 2:
        # Scale: 2% = 0 penalty, 20% = 10 penalty
        gap_pct_penalty = min((gap_pct - 2) / 1.8, 10)
        gap_score -= gap_pct_penalty

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
        metrics['phase_balance_ratio'] = balance_ratio

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
        metrics['phase_balance_ratio'] = 1.0
        balance_score = 15  # Default for single phase or can't calculate

    quality_score += balance_score
    metrics['balance_score'] = balance_score

    # ===== 4. MONTHLY COVERAGE BALANCE (up to 20 points) =====
    # Calculate coverage per month and check variance
    monthly_balance_score = 20  # Start with max

    if 'timestamp' in data.columns:
        data_temp = data.copy()
        data_temp['timestamp'] = pd.to_datetime(data_temp['timestamp'])
        data_temp['year_month'] = data_temp['timestamp'].dt.to_period('M')

        # Calculate expected vs actual rows per month
        monthly_counts = data_temp.groupby('year_month').size()

        if len(monthly_counts) > 1:
            # Calculate coverage ratio per month (assuming 1-min resolution)
            monthly_coverage = []
            for period, count in monthly_counts.items():
                expected = period.days_in_month * 24 * 60  # minutes in month
                coverage_ratio = min(count / expected, 1.0)
                monthly_coverage.append(coverage_ratio)

            # Calculate std dev of monthly coverage
            monthly_std = np.std(monthly_coverage)
            metrics['monthly_coverage_std'] = monthly_std
            metrics['monthly_coverage_values'] = monthly_coverage

            # Score: low std = high score
            # std 0 = 20 pts, std 0.5 = 0 pts
            monthly_balance_score = max(0, 20 * (1 - monthly_std * 2))
        else:
            metrics['monthly_coverage_std'] = 0
            # Single month - give full points

    quality_score += monthly_balance_score
    metrics['monthly_balance_score'] = monthly_balance_score

    # ===== 5. LOW NOISE / STABILITY (up to 15 points) =====
    # Based on hourly coefficient of variation
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
        metrics['avg_hourly_cv'] = avg_cv

        # Score based on CV (coefficient of variation)
        # Normal CV (0.3-0.8): 15 pts (healthy daily variation)
        # Very low CV (<0.3): 10 pts (suspicious - too flat)
        # High CV (0.8-1.5): 10 pts (some noise)
        # Very high CV (1.5-2.0): 5 pts (noisy)
        # Extreme CV (>2.0): 0 pts (very noisy/problematic)
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
    else:
        metrics['avg_hourly_cv'] = 0

    quality_score += noise_score
    metrics['noise_score'] = noise_score

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
