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
    # Optimized based on correlation analysis with actual algorithm performance.
    # Categories weighted by predictive power (Spearman rho with algo_overall_score):
    #   1. Event Detectability (35 pts) - strongest predictor (rho 0.30-0.53)
    #   2. Power Profile (20 pts) - mid-range avoidance (rho -0.43)
    #   3. Variability (20 pts) - CV predicts success (rho +0.41)
    #   4. Data Volume (15 pts) - days_span + monthly balance (rho +0.25)
    #   5. Data Integrity (10 pts) - NaN, negatives, critical gaps

    DETECTION_THRESHOLD = 1300  # algorithm's detection threshold in watts

    quality_score = 0

    # ===== 1. EVENT DETECTABILITY (up to 35 points) =====
    # How many detectable events (jumps >= threshold) exist per phase.
    # This is the single best predictor of algorithm success (rho up to +0.53).
    event_detectability_score = 0

    phase_event_densities = []
    phase_jumps_total = 0
    for col in phase_cols:
        if col not in data.columns:
            continue
        phase_data = data[col].dropna()
        if len(phase_data) < 2:
            continue
        diffs = phase_data.diff().dropna()
        # Count jumps >= threshold (both ON and OFF)
        jumps = int((diffs.abs() >= DETECTION_THRESHOLD).sum())
        phase_jumps_total += jumps
        # Event density: jumps per 10,000 minutes of data
        density = jumps / (len(phase_data) / 10000) if len(phase_data) > 0 else 0
        phase_event_densities.append(density)

    if phase_event_densities:
        avg_density = np.mean(phase_event_densities)
        # Calibrated on 171 houses: p10=5, p25=20, p50=50, p75=95, p90=160
        # Score: 0 density = 0 pts, density >= 120 = full 35 pts
        if avg_density < 5:
            event_detectability_score = avg_density / 5 * 5  # 0-5 pts
        elif avg_density < 20:
            event_detectability_score = 5 + (avg_density - 5) / 15 * 8  # 5-13 pts
        elif avg_density < 50:
            event_detectability_score = 13 + (avg_density - 20) / 30 * 9  # 13-22 pts
        elif avg_density < 120:
            event_detectability_score = 22 + (avg_density - 50) / 70 * 13  # 22-35 pts
        else:
            event_detectability_score = 35

    quality_score += event_detectability_score
    metrics['event_detectability_score'] = round(event_detectability_score, 1)
    metrics['total_threshold_jumps'] = phase_jumps_total

    # ===== 2. POWER PROFILE (up to 20 points) =====
    # Penalize houses stuck in 500-1000W range (rho=-0.43 with overall_score).
    # Reward houses with clear low-power baseline (< 500W) allowing sharp events.
    power_profile_score = 20

    phase_mid_shares = []
    phase_low_shares = []
    for col in phase_cols:
        if col not in data.columns:
            continue
        phase_data = data[col].dropna()
        if len(phase_data) == 0:
            continue
        total = len(phase_data)
        mid_share = ((phase_data >= 500) & (phase_data < 1000)).sum() / total
        low_share = (phase_data < 100).sum() / total
        phase_mid_shares.append(mid_share)
        phase_low_shares.append(low_share)

    if phase_mid_shares:
        avg_mid_share = np.mean(phase_mid_shares)
        avg_low_share = np.mean(phase_low_shares)

        # Penalty for high mid-range share (500-1000W)
        # Calibrated: avg_mid_share p25=0.10, p50=0.16, p75=0.22
        if avg_mid_share > 0.10:
            power_profile_score -= min(15, (avg_mid_share - 0.10) / 0.25 * 15)

        # Penalty for very little quiet time (< 500W)
        if avg_low_share < 0.15:
            power_profile_score -= min(5, (0.15 - avg_low_share) / 0.15 * 5)

    power_profile_score = max(0, power_profile_score)
    quality_score += power_profile_score
    metrics['power_profile_score'] = round(power_profile_score, 1)

    # ===== 3. VARIABILITY (up to 20 points) =====
    # Higher CV = more device activity = better algorithm performance (rho=+0.41).
    # This is the OPPOSITE of the old noise_score which penalized variability.
    variability_score = 0

    total_power = None
    active_cols = [c for c in phase_cols if c in data.columns]
    if active_cols:
        total_power = data[active_cols].sum(axis=1)
        total_mean = total_power.mean()
        total_std = total_power.std()
        total_cv = total_std / total_mean if total_mean > 0 else 0

        # Calibrated on 171 houses: CV p10=0.98, p25=1.13, p50=1.40, p75=1.86, p90=2.65
        # Higher CV = more device switching = better for algorithm
        if total_cv < 0.8:
            variability_score = total_cv / 0.8 * 4  # 0-4 pts
        elif total_cv < 1.1:
            variability_score = 4 + (total_cv - 0.8) / 0.3 * 4  # 4-8 pts
        elif total_cv < 1.4:
            variability_score = 8 + (total_cv - 1.1) / 0.3 * 4  # 8-12 pts
        elif total_cv < 2.0:
            variability_score = 12 + (total_cv - 1.4) / 0.6 * 5  # 12-17 pts
        elif total_cv < 3.0:
            variability_score = 17 + (total_cv - 2.0) / 1.0 * 3  # 17-20 pts
        else:
            variability_score = 20

    quality_score += variability_score
    metrics['variability_score'] = round(variability_score, 1)

    # ===== 4. DATA VOLUME (up to 15 points) =====
    # More days of data = better (rho=+0.25), plus monthly balance (rho=+0.14).
    data_volume_score = 0

    # Days span: 0-30 days = 0-3 pts, 30-180 = 3-7 pts, 180-365 = 7-10 pts, 365+ = 10 pts
    days = days_span if days_span is not None else 0
    if days >= 365:
        days_score = 10
    elif days >= 180:
        days_score = 7 + (days - 180) / 185 * 3
    elif days >= 30:
        days_score = 3 + (days - 30) / 150 * 4
    else:
        days_score = days / 30 * 3
    data_volume_score += days_score

    # Monthly coverage balance (up to 5 pts)
    monthly_balance_pts = 5
    if 'timestamp' in data.columns:
        data_temp = data.copy()
        data_temp['timestamp'] = pd.to_datetime(data_temp['timestamp'])
        data_temp['year_month'] = data_temp['timestamp'].dt.to_period('M')

        monthly_counts = data_temp.groupby('year_month').size()

        if len(monthly_counts) > 1:
            monthly_coverage = []
            for period, count in monthly_counts.items():
                expected = period.days_in_month * 24 * 60
                monthly_coverage.append(min(count / expected, 1.0))

            monthly_std = np.std(monthly_coverage)
            monthly_balance_pts = max(0, 5 * (1 - monthly_std * 2))

    data_volume_score += monthly_balance_pts
    data_volume_score = min(15, data_volume_score)

    quality_score += data_volume_score
    metrics['data_volume_score'] = round(data_volume_score, 1)
    metrics['monthly_balance_score'] = round(monthly_balance_pts, 1)

    # ===== 5. DATA INTEGRITY (up to 10 points) =====
    # Basic data quality: NaN, negative values, critical gaps.
    # Low weight because gaps/coverage barely correlate with performance (rho<0.05).
    integrity_score = 10

    # Penalty for NaN (up to 4 pts)
    nan_pct = avg_nan_pct if avg_nan_pct is not None else 0
    if nan_pct > 1:
        integrity_score -= min(4, (nan_pct - 1) / 4 * 4)

    # Penalty for many gaps > 2min (up to 3 pts)
    gap_pct = pct_gaps_over_2min if pct_gaps_over_2min is not None else 0
    if gap_pct > 5:
        integrity_score -= min(3, (gap_pct - 5) / 10 * 3)

    # Penalty for negative values (up to 3 pts)
    total_negatives = sum(
        metrics.get(f'{col}_negative_count', 0)
        for col in phase_cols
    )
    if total_negatives > 100:
        integrity_score -= min(3, (total_negatives - 100) / 500 * 3)

    integrity_score = max(0, integrity_score)
    quality_score += integrity_score
    metrics['integrity_score'] = round(integrity_score, 1)

    # Keep old component names for backward compatibility in reports
    metrics['completeness_score'] = round(data_volume_score, 1)
    metrics['gap_score'] = round(integrity_score, 1)
    metrics['balance_score'] = round(power_profile_score, 1)
    metrics['noise_score'] = round(variability_score, 1)

    # Ensure bounds
    metrics['quality_score'] = max(0, min(100, round(quality_score, 1)))

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
