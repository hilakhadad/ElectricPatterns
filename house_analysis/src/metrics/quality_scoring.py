"""
Quality scoring sub-functions for household data quality assessment.

Extracted from quality.py to keep scoring logic modular and maintainable.
Each function computes one component of the quality score (0-100).

Score components (total = 100 points):
  1. Sharp Entry Rate   (20 pts) -- _score_sharp_entry()
  2. Device Signature   (15 pts) -- _score_device_signature()
  3. Power Profile      (20 pts) -- _score_power_profile()
  4. Variability        (20 pts) -- _score_variability()
  5. Data Volume        (15 pts) -- _score_data_volume()
  6. Data Integrity     (10 pts) -- _score_data_integrity()

After the base score, anomaly penalties are applied via _apply_anomaly_penalties().
Quality tier is assigned via _compute_quality_tier().
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from metrics.quality import _calc_sharp_entry_rate, _calc_sustained_high_power, _calc_compressor_cycles

logger = logging.getLogger(__name__)

DETECTION_THRESHOLD = 1300  # algorithm's detection threshold in watts


def _score_sharp_entry(data: pd.DataFrame, phase_cols: list, metrics: dict) -> float:
    """
    Score sharp entry rate (up to 20 points).

    Fraction of threshold crossings caused by single-minute sharp jumps >= threshold.
    Best predictor of algorithm success (rho=+0.576 overall, +0.606 th_explanation_rate).
    """
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
            sharp_entry_score = avg_sharp_rate / 0.15 * 3
        elif avg_sharp_rate < 0.25:
            sharp_entry_score = 3 + (avg_sharp_rate - 0.15) / 0.10 * 5
        elif avg_sharp_rate < 0.35:
            sharp_entry_score = 8 + (avg_sharp_rate - 0.25) / 0.10 * 5
        elif avg_sharp_rate < 0.50:
            sharp_entry_score = 13 + (avg_sharp_rate - 0.35) / 0.15 * 5
        else:
            sharp_entry_score = 20
    else:
        avg_sharp_rate = 0.0

    metrics['sharp_entry_score'] = round(sharp_entry_score, 1)
    metrics['sharp_entry_rate'] = round(avg_sharp_rate, 4)
    metrics['total_threshold_jumps'] = phase_jumps_total

    return sharp_entry_score


def _score_device_signature(data: pd.DataFrame, phase_cols: list, metrics: dict) -> float:
    """
    Score device signature (up to 15 points).

    Detects boiler-like (sustained high power, up to 8 pts) and
    AC-like (compressor cycles, up to 7 pts) patterns.
    """
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
        if avg_sustained < 0.5:
            sustained_score = avg_sustained / 0.5 * 1
        elif avg_sustained < 2.0:
            sustained_score = 1 + (avg_sustained - 0.5) / 1.5 * 2
        elif avg_sustained < 5.0:
            sustained_score = 3 + (avg_sustained - 2.0) / 3.0 * 3
        elif avg_sustained < 9.0:
            sustained_score = 6 + (avg_sustained - 5.0) / 4.0 * 2
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
        if avg_compressor < 2.0:
            compressor_score = avg_compressor / 2.0 * 1
        elif avg_compressor < 6.0:
            compressor_score = 1 + (avg_compressor - 2.0) / 4.0 * 2
        elif avg_compressor < 15.0:
            compressor_score = 3 + (avg_compressor - 6.0) / 9.0 * 2
        elif avg_compressor < 35.0:
            compressor_score = 5 + (avg_compressor - 15.0) / 20.0 * 2
        else:
            compressor_score = 7
    else:
        avg_compressor = 0.0

    device_signature_score = sustained_score + compressor_score
    metrics['device_signature_score'] = round(device_signature_score, 1)
    metrics['sustained_high_power_density'] = round(avg_sustained, 2)
    metrics['compressor_cycle_density'] = round(avg_compressor, 2)

    return device_signature_score


def _score_power_profile(data: pd.DataFrame, phase_cols: list, metrics: dict) -> float:
    """
    Score power profile (up to 20 points).

    Penalizes houses stuck in 500-1000W range (rho=-0.43 with overall_score).
    Rewards houses with clear low-power baseline (< 500W).
    """
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

        if avg_mid_share > 0.10:
            power_profile_score -= min(15, (avg_mid_share - 0.10) / 0.25 * 15)
        if avg_low_share < 0.15:
            power_profile_score -= min(5, (0.15 - avg_low_share) / 0.15 * 5)

    power_profile_score = max(0, power_profile_score)
    metrics['power_profile_score'] = round(power_profile_score, 1)

    return power_profile_score


def _score_variability(data: pd.DataFrame, phase_cols: list, metrics: dict) -> float:
    """
    Score variability (up to 20 points).

    Higher CV = more device activity = better algorithm performance (rho=+0.41).
    """
    variability_score = 0

    active_cols = [c for c in phase_cols if c in data.columns]
    if active_cols:
        total_power = data[active_cols].sum(axis=1)
        total_mean = total_power.mean()
        total_std = total_power.std()
        total_cv = total_std / total_mean if total_mean > 0 else 0

        if total_cv < 0.8:
            variability_score = total_cv / 0.8 * 4
        elif total_cv < 1.1:
            variability_score = 4 + (total_cv - 0.8) / 0.3 * 4
        elif total_cv < 1.4:
            variability_score = 8 + (total_cv - 1.1) / 0.3 * 4
        elif total_cv < 2.0:
            variability_score = 12 + (total_cv - 1.4) / 0.6 * 5
        elif total_cv < 3.0:
            variability_score = 17 + (total_cv - 2.0) / 1.0 * 3
        else:
            variability_score = 20

    metrics['variability_score'] = round(variability_score, 1)

    return variability_score


def _score_data_volume(data: pd.DataFrame, days_span: Optional[int],
                       metrics: dict) -> float:
    """
    Score data volume (up to 15 points).

    More days of data = better (rho=+0.25), plus monthly balance (rho=+0.14).
    """
    data_volume_score = 0

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

    metrics['data_volume_score'] = round(data_volume_score, 1)
    metrics['monthly_balance_score'] = round(monthly_balance_pts, 1)

    return data_volume_score


def _score_data_integrity(data: pd.DataFrame, phase_cols: list,
                          metrics: dict,
                          avg_nan_pct: Optional[float] = None,
                          pct_gaps_over_2min: Optional[float] = None,
                          coverage_ratio: Optional[float] = None,
                          anomaly_count: Optional[int] = None) -> float:
    """
    Score data integrity (up to 10 points).

    Basic data quality: NaN, negative values, critical gaps, zero-power months.
    """
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

    # Penalty for anomalous readings (up to 2 pts)
    n_anomalies = anomaly_count if anomaly_count is not None else 0
    if n_anomalies > 0:
        integrity_score -= min(2, n_anomalies / 50 * 2)
    metrics['anomaly_penalty'] = round(min(2, n_anomalies / 50 * 2) if n_anomalies > 0 else 0, 1)

    # Penalty for zero-power months (up to 3 pts)
    zero_power_months = 0
    total_months = 0
    if 'timestamp' in data.columns:
        data_zp = data.copy()
        data_zp['timestamp'] = pd.to_datetime(data_zp['timestamp'])
        data_zp['year_month'] = data_zp['timestamp'].dt.to_period('M')
        for period, group in data_zp.groupby('year_month'):
            total_months += 1
            sum_cols = [c for c in phase_cols if c in group.columns]
            if sum_cols:
                total_power_col = group[sum_cols].sum(axis=1)
                if total_power_col.mean() < 1.0 and len(group) > 100:
                    zero_power_months += 1
    metrics['zero_power_months'] = zero_power_months
    metrics['total_months'] = total_months
    if zero_power_months > 0 and total_months > 0:
        zero_ratio = zero_power_months / total_months
        zero_penalty = min(3, zero_ratio * 10)
        integrity_score -= zero_penalty
        metrics['zero_power_penalty'] = round(zero_penalty, 1)
    else:
        metrics['zero_power_penalty'] = 0.0

    integrity_score = max(0, integrity_score)
    metrics['integrity_score'] = round(integrity_score, 1)

    return integrity_score


def _apply_anomaly_penalties(metrics: dict,
                             coverage_ratio: Optional[float] = None,
                             anomaly_count: Optional[int] = None,
                             high_power_density: Optional[float] = None) -> float:
    """
    Apply anomaly penalties to the base quality score.

    Returns:
        Final quality score after penalties.
    """
    base_score = metrics.get('base_quality_score', 0)
    anomaly_penalties = []

    # Dead phase penalty (-15 per phase)
    n_dead = len(metrics.get('dead_phases', []))
    if n_dead > 0:
        dead_names = ', '.join(metrics['dead_phases'])
        deduction = min(30, n_dead * 15)
        anomaly_penalties.append({
            'reason': f'dead_phase ({dead_names})',
            'deduction': deduction,
        })

    # Faulty NaN phase penalty (-10 per phase)
    n_faulty = len(metrics.get('faulty_nan_phases', []))
    if n_faulty > 0:
        faulty_names = ', '.join(metrics['faulty_nan_phases'])
        deduction = min(20, n_faulty * 10)
        anomaly_penalties.append({
            'reason': f'faulty_nan_phase ({faulty_names})',
            'deduction': deduction,
        })

    # Coverage penalties
    cov = coverage_ratio if coverage_ratio is not None else 1.0
    if cov < 0.50:
        anomaly_penalties.append({
            'reason': f'very_low_coverage ({cov:.0%})',
            'deduction': 15,
        })
    elif cov < 0.70:
        anomaly_penalties.append({
            'reason': f'low_coverage ({cov:.0%})',
            'deduction': 8,
        })

    # Extreme outliers
    n_anomalies = anomaly_count if anomaly_count is not None else 0
    if n_anomalies > 0:
        anomaly_penalties.append({
            'reason': f'extreme_outliers ({n_anomalies} readings >20kW)',
            'deduction': 5,
        })

    # High power density — most of the time above pipeline thresholds
    hpd = high_power_density if high_power_density is not None else 0
    if hpd > 0.40:
        anomaly_penalties.append({
            'reason': f'high_power_density ({hpd:.0%} of time above 800W)',
            'deduction': 8,
        })
    elif hpd > 0.25:
        anomaly_penalties.append({
            'reason': f'high_power_density ({hpd:.0%} of time above 800W)',
            'deduction': 4,
        })

    # Fragmented data
    total_loss = metrics.get('total_data_loss_pct', 0)
    if total_loss >= 40:
        anomaly_penalties.append({
            'reason': f'fragmented_data ({total_loss:.0f}% data loss)',
            'deduction': 10,
        })
    elif total_loss >= 15:
        anomaly_penalties.append({
            'reason': f'discontinuous_data ({total_loss:.0f}% data loss)',
            'deduction': 5,
        })

    total_penalty = sum(p['deduction'] for p in anomaly_penalties)
    final_score = max(0, base_score - total_penalty)

    metrics['anomaly_penalties'] = total_penalty
    metrics['anomaly_penalty_details'] = anomaly_penalties
    metrics['quality_score'] = round(final_score, 1)

    logger.info("quality_score=%.1f, label=%s (base=%.1f, penalty=%.1f)",
                final_score, metrics.get('quality_label', 'N/A'), base_score, total_penalty)

    return final_score


def _compute_quality_tier(metrics: dict) -> None:
    """
    Compute quality flags and faulty phase label.

    Updates metrics dict in-place with 'quality_flags' and 'quality_label'.
    """
    sharp_entry_score = metrics.get('sharp_entry_score', 0)
    device_signature_score = metrics.get('device_signature_score', 0)
    power_profile_score = metrics.get('power_profile_score', 0)
    variability_score = metrics.get('variability_score', 0)
    data_volume_score = metrics.get('data_volume_score', 0)
    integrity_score = metrics.get('integrity_score', 0)

    quality_flags = []
    if sharp_entry_score < 8:
        quality_flags.append('low_sharp_entry')
    if device_signature_score < 4:
        quality_flags.append('low_device_signature')
    if power_profile_score < 10:
        quality_flags.append('low_power_profile')
    if variability_score < 8:
        quality_flags.append('low_variability')
    if data_volume_score < 7:
        quality_flags.append('low_data_volume')
    if integrity_score < 5:
        quality_flags.append('low_data_integrity')
    metrics['quality_flags'] = quality_flags

    # Mark faulty phases
    has_dead = metrics.get('has_dead_phase', False)
    has_nan = metrics.get('has_faulty_nan_phase', False)
    non_dead_nan = [p for p in metrics.get('faulty_nan_phases', [])
                    if p not in metrics.get('dead_phases', [])]
    if has_dead and len(non_dead_nan) > 0:
        metrics['quality_label'] = 'faulty_both'
    elif has_dead:
        metrics['quality_label'] = 'faulty_dead_phase'
    elif has_nan:
        metrics['quality_label'] = 'faulty_high_nan'
    else:
        metrics['quality_label'] = None
