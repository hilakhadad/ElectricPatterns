"""
Data quality metrics for household data.

Detects anomalies, outliers, and potential data issues.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple


def _calc_sharp_entry_rate(vals, threshold):
    """
    Sharp Entry Rate: fraction of threshold crossings caused by single-minute jumps.

    Finds points where power crosses above threshold, then checks how many
    had a single-minute jump >= threshold. Returns sharp_crossings / total_crossings.
    """
    if len(vals) < 2:
        return np.nan

    above = vals >= threshold
    prev_above = np.roll(above, 1)
    prev_above[0] = False
    is_valid = ~np.isnan(vals)
    crossings = (~prev_above) & above & is_valid

    total_crossings = np.sum(crossings)
    if total_crossings == 0:
        return np.nan

    prev_vals = np.roll(vals, 1)
    prev_vals[0] = 0.0
    jumps = vals - prev_vals
    sharp_crossings = np.sum(crossings & (jumps >= threshold))

    return sharp_crossings / total_crossings


def _calc_sustained_high_power(vals, threshold=2000, min_duration=20):
    """
    Count periods where power stays >= threshold for >= min_duration consecutive minutes.
    Returns count normalized per 10K minutes.
    """
    total_valid = np.sum(~np.isnan(vals))
    if total_valid < 100:
        return np.nan

    above = vals >= threshold
    above[np.isnan(vals)] = False

    count = 0
    run_length = 0
    for v in above:
        if v:
            run_length += 1
        else:
            if run_length >= min_duration:
                count += 1
            run_length = 0
    if run_length >= min_duration:
        count += 1

    return count / (total_valid / 10000.0)


def _calc_compressor_cycles(vals, min_w=1300, max_w=3000,
                            tolerance=0.30, min_dur=3, max_dur=30):
    """
    Count ON/OFF pairs that look like compressor cycles.
    Returns count normalized per 10K minutes.
    """
    total_valid = np.sum(~np.isnan(vals))
    if total_valid < 100:
        return np.nan

    diffs = np.diff(vals)
    diffs = np.nan_to_num(diffs, nan=0.0)

    on_mask = (diffs >= min_w) & (diffs <= max_w)
    on_indices = np.where(on_mask)[0]

    count = 0
    used_off = set()

    for on_idx in on_indices:
        on_magnitude = diffs[on_idx]
        search_start = on_idx + min_dur
        search_end = min(on_idx + max_dur + 1, len(diffs))

        for off_idx in range(search_start, search_end):
            if off_idx in used_off:
                continue
            off_magnitude = -diffs[off_idx]
            if off_magnitude < min_w:
                continue
            ratio = off_magnitude / on_magnitude if on_magnitude > 0 else 0
            if (1 - tolerance) <= ratio <= (1 + tolerance):
                count += 1
                used_off.add(off_idx)
                break

    return count / (total_valid / 10000.0)


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
    # Optimized based on correlation analysis with actual algorithm performance (161 houses).
    # Categories weighted by predictive power (Spearman rho with algo scores):
    #   1. Sharp Entry Rate (20 pts) - best single predictor (rho +0.576 overall, +0.606 th_expl)
    #   2. Device Signature (15 pts) - boiler + AC patterns (rho +0.611 seg_ratio)
    #   3. Power Profile (20 pts) - mid-range avoidance (rho -0.43)
    #   4. Variability (20 pts) - CV predicts success (rho +0.41)
    #   5. Data Volume (15 pts) - days_span + monthly balance (rho +0.25)
    #   6. Data Integrity (10 pts) - NaN, negatives, critical gaps

    DETECTION_THRESHOLD = 1300  # algorithm's detection threshold in watts

    quality_score = 0

    # ===== 1. SHARP ENTRY RATE (up to 20 points) =====
    # Fraction of threshold crossings caused by single-minute sharp jumps >= threshold.
    # Best predictor of algorithm success (rho=+0.576 overall, +0.606 th_explanation_rate).
    # Houses where power reaches threshold via device stacking (gradual) score low here.
    sharp_entry_score = 0

    phase_sharp_rates = []
    phase_jumps_total = 0
    for col in phase_cols:
        if col not in data.columns:
            continue
        phase_data = data[col].dropna()
        if len(phase_data) < 2:
            continue
        diffs = phase_data.diff().dropna()
        jumps = int((diffs.abs() >= DETECTION_THRESHOLD).sum())
        phase_jumps_total += jumps
        sr = _calc_sharp_entry_rate(phase_data.values, threshold=DETECTION_THRESHOLD)
        if not np.isnan(sr):
            phase_sharp_rates.append(sr)

    if phase_sharp_rates:
        avg_sharp_rate = np.mean(phase_sharp_rates)
        # Calibrated on 161 houses: p5=0.15, p25=0.24, p50=0.32, p75=0.43, p90=0.52
        if avg_sharp_rate < 0.15:
            sharp_entry_score = avg_sharp_rate / 0.15 * 3  # 0-3 pts
        elif avg_sharp_rate < 0.25:
            sharp_entry_score = 3 + (avg_sharp_rate - 0.15) / 0.10 * 5  # 3-8 pts
        elif avg_sharp_rate < 0.35:
            sharp_entry_score = 8 + (avg_sharp_rate - 0.25) / 0.10 * 5  # 8-13 pts
        elif avg_sharp_rate < 0.50:
            sharp_entry_score = 13 + (avg_sharp_rate - 0.35) / 0.15 * 5  # 13-18 pts
        else:
            sharp_entry_score = 20
    else:
        avg_sharp_rate = 0.0

    quality_score += sharp_entry_score
    metrics['sharp_entry_score'] = round(sharp_entry_score, 1)
    metrics['sharp_entry_rate'] = round(avg_sharp_rate, 4)
    metrics['total_threshold_jumps'] = phase_jumps_total

    # ===== 2. DEVICE SIGNATURE (up to 15 points) =====
    # Detects boiler-like (sustained high power) and AC-like (compressor cycles) patterns.
    # Sustained high power: rho=+0.611 with seg_ratio (best predictor for segregation).
    # Compressor cycles: rho=+0.424 with overall_score.

    # --- Sustained High Power (boiler-like): up to 8 points ---
    sustained_score = 0
    phase_sustained = []
    for col in phase_cols:
        if col not in data.columns:
            continue
        vals = data[col].dropna().values
        if len(vals) < 100:
            continue
        sh = _calc_sustained_high_power(vals)
        if sh is not None and not np.isnan(sh):
            phase_sustained.append(sh)

    if phase_sustained:
        avg_sustained = np.mean(phase_sustained)
        # Calibrated on 161 houses: p5=0.39, p25=1.57, p50=3.17, p75=5.56, p90=9.06
        if avg_sustained < 0.5:
            sustained_score = avg_sustained / 0.5 * 1  # 0-1 pts
        elif avg_sustained < 2.0:
            sustained_score = 1 + (avg_sustained - 0.5) / 1.5 * 2  # 1-3 pts
        elif avg_sustained < 5.0:
            sustained_score = 3 + (avg_sustained - 2.0) / 3.0 * 3  # 3-6 pts
        elif avg_sustained < 9.0:
            sustained_score = 6 + (avg_sustained - 5.0) / 4.0 * 2  # 6-8 pts
        else:
            sustained_score = 8
    else:
        avg_sustained = 0.0

    # --- Compressor Cycles (AC-like): up to 7 points ---
    compressor_score = 0
    phase_compressor = []
    for col in phase_cols:
        if col not in data.columns:
            continue
        vals = data[col].dropna().values
        if len(vals) < 100:
            continue
        cc = _calc_compressor_cycles(vals)
        if cc is not None and not np.isnan(cc):
            phase_compressor.append(cc)

    if phase_compressor:
        avg_compressor = np.mean(phase_compressor)
        # Calibrated on 161 houses: p5=2.04, p25=5.90, p50=10.66, p75=19.84, p90=36.72
        if avg_compressor < 2.0:
            compressor_score = avg_compressor / 2.0 * 1  # 0-1 pts
        elif avg_compressor < 6.0:
            compressor_score = 1 + (avg_compressor - 2.0) / 4.0 * 2  # 1-3 pts
        elif avg_compressor < 15.0:
            compressor_score = 3 + (avg_compressor - 6.0) / 9.0 * 2  # 3-5 pts
        elif avg_compressor < 35.0:
            compressor_score = 5 + (avg_compressor - 15.0) / 20.0 * 2  # 5-7 pts
        else:
            compressor_score = 7
    else:
        avg_compressor = 0.0

    device_signature_score = sustained_score + compressor_score
    quality_score += device_signature_score
    metrics['device_signature_score'] = round(device_signature_score, 1)
    metrics['sustained_high_power_density'] = round(avg_sustained, 2)
    metrics['compressor_cycle_density'] = round(avg_compressor, 2)

    # ===== 3. POWER PROFILE (up to 20 points) =====
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

    # ===== 4. VARIABILITY (up to 20 points) =====
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

    # ===== 5. DATA VOLUME (up to 15 points) =====
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

    # ===== 6. DATA INTEGRITY (up to 10 points) =====
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
    metrics['event_detectability_score'] = round(sharp_entry_score + device_signature_score, 1)
    metrics['completeness_score'] = round(data_volume_score, 1)
    metrics['gap_score'] = round(integrity_score, 1)
    metrics['balance_score'] = round(power_profile_score, 1)
    metrics['noise_score'] = round(variability_score, 1)

    # Ensure bounds
    metrics['quality_score'] = max(0, min(100, round(quality_score, 1)))

    # ===== QUALITY FLAGS (per-component tags for insufficient scores) =====
    # Each flag indicates the house doesn't meet sufficient level for that component.
    # Thresholds calibrated at approximately the p25 level of 161 houses.
    quality_flags = []
    if sharp_entry_score < 8:  # ~p25 sharp entry rate
        quality_flags.append('low_sharp_entry')
    if device_signature_score < 4:  # ~p25 device signature
        quality_flags.append('low_device_signature')
    if power_profile_score < 10:  # below half of max
        quality_flags.append('low_power_profile')
    if variability_score < 8:  # ~p25 variability
        quality_flags.append('low_variability')
    if data_volume_score < 7:  # below half of max
        quality_flags.append('low_data_volume')
    if integrity_score < 5:  # below half of max
        quality_flags.append('low_data_integrity')
    metrics['quality_flags'] = quality_flags

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
