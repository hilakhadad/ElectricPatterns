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
                                    avg_nan_pct: float = None,
                                    anomaly_count: int = None) -> Dict[str, Any]:
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
    # A phase is dead if its total power is less than 2% of the average of the other two phases
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

    DEAD_PHASE_RATIO = 0.02  # 2% of sisters' average

    for col, power in phase_powers.items():
        sisters = [v for k, v in phase_powers.items() if k != col]
        sisters_avg = sum(sisters) / len(sisters) if sisters else 0
        if sisters_avg > 0 and power / sisters_avg < DEAD_PHASE_RATIO:
            dead_phases.append(col)

    metrics['dead_phases'] = dead_phases
    metrics['has_dead_phase'] = len(dead_phases) > 0

    # ===== FAULTY PHASE DETECTION (NaN >= 10%) =====
    # A non-dead phase with >= 10% NaN values is considered faulty.
    # Dead phases are excluded — their NaN is expected and already flagged separately.
    FAULTY_NAN_THRESHOLD = 10.0  # percent

    faulty_nan_phases = []
    for col in phase_cols:
        if col not in data.columns:
            continue
        col_nan_pct = data[col].isna().sum() / len(data) * 100 if len(data) > 0 else 0
        metrics[f'{col}_nan_pct'] = col_nan_pct
        if col not in dead_phases and col_nan_pct >= FAULTY_NAN_THRESHOLD:
            faulty_nan_phases.append(col)

    metrics['faulty_nan_phases'] = faulty_nan_phases
    metrics['has_faulty_nan_phase'] = len(faulty_nan_phases) > 0

    # ===== DATA CONTINUITY CLASSIFICATION =====
    # Classify data continuity based on BOTH:
    # 1. NaN percentage within existing rows (max across phases)
    # 2. No-data percentage (missing rows entirely, from coverage_ratio)
    # A house can have 0% NaN but 50% missing rows if the sensor was offline for months.
    phase_nan_pcts = [metrics.get(f'{col}_nan_pct', 0) for col in phase_cols if col in data.columns]
    max_phase_nan_pct = max(phase_nan_pcts) if phase_nan_pcts else 0
    metrics['max_phase_nan_pct'] = round(max_phase_nan_pct, 2)

    # Compute total data loss: missing rows + NaN within existing rows
    no_data_pct = (1 - (coverage_ratio if coverage_ratio is not None else 1.0)) * 100
    # NaN affects only existing rows, so scale by coverage
    effective_nan_pct = max_phase_nan_pct * (coverage_ratio if coverage_ratio is not None else 1.0)
    total_data_loss_pct = no_data_pct + effective_nan_pct
    metrics['no_data_pct'] = round(no_data_pct, 2)
    metrics['total_data_loss_pct'] = round(total_data_loss_pct, 2)

    if total_data_loss_pct < 5:
        nan_continuity = 'continuous'       # Continuous data
    elif total_data_loss_pct < 15:
        nan_continuity = 'minor_gaps'       # Minor gaps
    elif total_data_loss_pct < 40:
        nan_continuity = 'discontinuous'    # Discontinuous
    else:
        nan_continuity = 'fragmented'       # Heavily fragmented

    metrics['nan_continuity_label'] = nan_continuity

    # ===== QUALITY SCORING SYSTEM (0-100) =====
    # Scoring logic extracted to quality_scoring.py for modularity.
    # Import scoring sub-functions
    from metrics.quality_scoring import (
        _score_sharp_entry,
        _score_device_signature,
        _score_power_profile,
        _score_variability,
        _score_data_volume,
        _score_data_integrity,
        _apply_anomaly_penalties,
        _compute_quality_tier,
    )

    quality_score = 0

    sharp_entry_score = _score_sharp_entry(data, phase_cols, metrics)
    quality_score += sharp_entry_score

    device_signature_score = _score_device_signature(data, phase_cols, metrics)
    quality_score += device_signature_score

    power_profile_score = _score_power_profile(data, phase_cols, metrics)
    quality_score += power_profile_score

    variability_score = _score_variability(data, phase_cols, metrics)
    quality_score += variability_score

    data_volume_score = _score_data_volume(data, days_span, metrics)
    quality_score += data_volume_score

    integrity_score = _score_data_integrity(
        data, phase_cols, metrics,
        avg_nan_pct=avg_nan_pct,
        pct_gaps_over_2min=pct_gaps_over_2min,
        coverage_ratio=coverage_ratio,
        anomaly_count=anomaly_count,
    )
    quality_score += integrity_score

    # Keep old component names for backward compatibility in reports
    metrics['event_detectability_score'] = round(sharp_entry_score + device_signature_score, 1)
    metrics['completeness_score'] = round(data_volume_score, 1)
    metrics['gap_score'] = round(integrity_score, 1)
    metrics['balance_score'] = round(power_profile_score, 1)
    metrics['noise_score'] = round(variability_score, 1)

    # Save base score before anomaly penalties
    base_score = max(0, min(100, round(quality_score, 1)))
    metrics['base_quality_score'] = base_score

    # Apply anomaly penalties and compute final score
    _apply_anomaly_penalties(metrics, coverage_ratio=coverage_ratio, anomaly_count=anomaly_count)

    # Compute quality flags and faulty phase label
    _compute_quality_tier(metrics)

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
