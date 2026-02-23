"""
Wave behavior pre-analysis for household power data.

Detects periodic cycling patterns (AC compressor) in raw power data using
autocorrelation of the minute-to-minute diff signal. Classifies each house
as wave_dominant / has_waves / no_waves.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional


# --- Constants ---
AC_MIN_WATTS = 800
AC_MAX_WATTS = 4000
MIN_WINDOW_MINUTES = 30
MAX_WINDOW_MINUTES = 180
MAX_NAN_FRACTION = 0.20
ACF_MIN_LAG = 3
ACF_MAX_LAG = 30
MIN_ACF_DATAPOINTS = 10  # minimum points after removing lag for valid ACF

WAVE_DOMINANT_THRESHOLD = 0.50
WAVE_PRESENT_THRESHOLD = 0.25
MIN_WAVE_MINUTES_FOR_DOMINANT = 120

SUMMER_MONTHS = {6, 7, 8, 9}
WINTER_MONTHS = {12, 1, 2, 3}


def _find_wave_windows(vals: np.ndarray, min_w: float = AC_MIN_WATTS,
                       max_w: float = AC_MAX_WATTS,
                       min_dur: int = MIN_WINDOW_MINUTES,
                       max_dur: int = MAX_WINDOW_MINUTES,
                       max_nan_frac: float = MAX_NAN_FRACTION
                       ) -> List[Tuple[int, int]]:
    """
    Find contiguous segments where power is in [min_w, max_w] range.

    Returns list of (start_idx, end_idx) for qualifying segments.
    Segments longer than max_dur are split at max_dur boundaries.
    """
    n = len(vals)
    if n < min_dur:
        return []

    # Mark valid positions: in range and not NaN
    is_nan = np.isnan(vals)
    in_range = (~is_nan) & (vals >= min_w) & (vals <= max_w)

    # Find contiguous runs of in_range OR NaN (NaN allowed up to fraction)
    # We include NaN positions within runs and check fraction afterward
    eligible = in_range | is_nan
    eligible[~is_nan & ((vals < min_w) | (vals > max_w))] = False

    windows = []
    start = None

    for i in range(n):
        if in_range[i]:
            if start is None:
                start = i
        elif is_nan[i] and start is not None:
            # Allow NaN within a run
            continue
        else:
            if start is not None:
                _add_window(vals, start, i, min_dur, max_dur, max_nan_frac, windows)
                start = None

    if start is not None:
        _add_window(vals, start, n, min_dur, max_dur, max_nan_frac, windows)

    return windows


def _add_window(vals, start, end, min_dur, max_dur, max_nan_frac, windows):
    """Validate and split a candidate window, appending qualifying segments."""
    length = end - start
    if length < min_dur:
        return

    segment = vals[start:end]
    nan_frac = np.sum(np.isnan(segment)) / length
    if nan_frac > max_nan_frac:
        return

    # Split long windows
    if length > max_dur:
        for chunk_start in range(0, length, max_dur):
            chunk_end = min(chunk_start + max_dur, length)
            if chunk_end - chunk_start >= min_dur:
                chunk = vals[start + chunk_start:start + chunk_end]
                chunk_nan_frac = np.sum(np.isnan(chunk)) / (chunk_end - chunk_start)
                if chunk_nan_frac <= max_nan_frac:
                    windows.append((start + chunk_start, start + chunk_end))
    else:
        windows.append((start, end))


def _compute_acf_peak(diffs: np.ndarray,
                      min_lag: int = ACF_MIN_LAG,
                      max_lag: int = ACF_MAX_LAG
                      ) -> Tuple[float, Optional[int]]:
    """
    Compute peak autocorrelation of diff signal at lags min_lag..max_lag.

    Returns (peak_abs_acf, dominant_lag) or (0.0, None) if insufficient data.
    """
    n = len(diffs)
    # Ensure enough data points after removing lag
    effective_max_lag = min(max_lag, n - MIN_ACF_DATAPOINTS)
    if effective_max_lag < min_lag:
        return 0.0, None

    best_acf = 0.0
    best_lag = None

    for lag in range(min_lag, effective_max_lag + 1):
        x = diffs[:-lag]
        y = diffs[lag:]

        # Remove NaN pairs
        valid = ~(np.isnan(x) | np.isnan(y))
        x_valid = x[valid]
        y_valid = y[valid]

        if len(x_valid) < MIN_ACF_DATAPOINTS:
            continue

        # Pearson correlation
        x_mean = x_valid.mean()
        y_mean = y_valid.mean()
        x_dev = x_valid - x_mean
        y_dev = y_valid - y_mean
        denom = np.sqrt(np.sum(x_dev ** 2) * np.sum(y_dev ** 2))
        if denom == 0:
            continue

        corr = np.sum(x_dev * y_dev) / denom
        abs_corr = abs(corr)

        if abs_corr > best_acf:
            best_acf = abs_corr
            best_lag = lag

    return best_acf, best_lag


def _compute_phase_wave_metrics(vals: np.ndarray,
                                timestamps: np.ndarray
                                ) -> Dict[str, Any]:
    """
    Full per-phase wave behavior computation.

    Args:
        vals: Power values (1-min resolution) for a single phase
        timestamps: Corresponding datetime64 timestamps

    Returns:
        Dict with wave_score, dominant_period, prevalence, etc.
    """
    result = {
        'wave_score': 0.0,
        'dominant_period_minutes': None,
        'total_wave_minutes': 0,
        'wave_prevalence_pct': 0.0,
        'avg_amplitude': 0.0,
        'monthly_minutes': {},
        'hourly_minutes': {},
    }

    total_valid = np.sum(~np.isnan(vals))
    if total_valid < 200:
        return result

    # Find wave windows
    windows = _find_wave_windows(vals)
    if not windows:
        return result

    # Compute per-window metrics
    window_scores = []
    window_durations = []
    window_lags = []
    window_amplitudes = []

    for start, end in windows:
        segment = vals[start:end]
        diffs = np.diff(segment)

        acf_peak, dominant_lag = _compute_acf_peak(diffs)

        duration = end - start
        window_scores.append(acf_peak)
        window_durations.append(duration)
        if dominant_lag is not None:
            window_lags.append(dominant_lag)

        # Amplitude: mean power in window minus overall median
        seg_valid = segment[~np.isnan(segment)]
        if len(seg_valid) > 0:
            window_amplitudes.append(seg_valid.mean())

    durations = np.array(window_durations, dtype=float)
    scores = np.array(window_scores)
    total_duration = durations.sum()

    if total_duration == 0:
        return result

    # Duration-weighted wave score
    wave_score = float(np.average(scores, weights=durations))

    # Dominant period: weighted mode of lags
    dominant_period = None
    if window_lags:
        lag_arr = np.array(window_lags)
        # Use the lag from the highest-scoring window
        best_window_idx = np.argmax(scores)
        if window_lags and best_window_idx < len(windows):
            seg_start, seg_end = windows[best_window_idx]
            seg_diffs = np.diff(vals[seg_start:seg_end])
            _, best_lag = _compute_acf_peak(seg_diffs)
            if best_lag is not None:
                dominant_period = int(best_lag)

    # Amplitude
    overall_median = np.nanmedian(vals)
    avg_amplitude = 0.0
    if window_amplitudes:
        avg_amplitude = float(np.mean(window_amplitudes) - overall_median)
        avg_amplitude = max(0.0, avg_amplitude)

    # Temporal distribution
    monthly_minutes = {}
    hourly_minutes = {}

    ts = pd.DatetimeIndex(timestamps)
    for start, end in windows:
        window_ts = ts[start:end]
        if len(window_ts) == 0:
            continue

        # Monthly
        months = window_ts.month
        for m in months:
            monthly_minutes[int(m)] = monthly_minutes.get(int(m), 0) + 1

        # Hourly
        hours = window_ts.hour
        for h in hours:
            hourly_minutes[int(h)] = hourly_minutes.get(int(h), 0) + 1

    result['wave_score'] = round(wave_score, 4)
    result['dominant_period_minutes'] = dominant_period
    result['total_wave_minutes'] = int(total_duration)
    result['wave_prevalence_pct'] = round(total_duration / total_valid * 100, 2)
    result['avg_amplitude'] = round(avg_amplitude, 1)
    result['monthly_minutes'] = monthly_minutes
    result['hourly_minutes'] = hourly_minutes

    return result


def calculate_wave_behavior_metrics(data: pd.DataFrame,
                                    phase_cols: list = None
                                    ) -> Dict[str, Any]:
    """
    Main entry point: compute wave behavior metrics for all phases.

    Args:
        data: DataFrame with timestamp and phase power columns
        phase_cols: List of phase column names (auto-detected if None)

    Returns:
        Dictionary with per-phase metrics, house-level classification,
        and temporal distribution.
    """
    if phase_cols is None:
        if 'w1' in data.columns:
            phase_cols = ['w1', 'w2', 'w3']
        elif '1' in data.columns:
            phase_cols = ['1', '2', '3']
        else:
            phase_cols = [c for c in data.columns if c not in ['timestamp', 'sum']]

    timestamps = pd.to_datetime(data['timestamp']).values if 'timestamp' in data.columns else None

    phases = {}
    for col in phase_cols:
        if col not in data.columns:
            continue
        vals = data[col].values.astype(float)
        if timestamps is not None:
            phase_metrics = _compute_phase_wave_metrics(vals, timestamps)
        else:
            phase_metrics = {
                'wave_score': 0.0, 'dominant_period_minutes': None,
                'total_wave_minutes': 0, 'wave_prevalence_pct': 0.0,
                'avg_amplitude': 0.0, 'monthly_minutes': {}, 'hourly_minutes': {},
            }
        phases[col] = phase_metrics

    # House-level summary
    max_wave_score = max((p['wave_score'] for p in phases.values()), default=0.0)
    total_wave_minutes = sum(p['total_wave_minutes'] for p in phases.values())

    dominant_phases = [
        col for col, p in phases.items()
        if p['wave_score'] >= WAVE_PRESENT_THRESHOLD
    ]

    # Classification
    if max_wave_score >= WAVE_DOMINANT_THRESHOLD and total_wave_minutes >= MIN_WAVE_MINUTES_FOR_DOMINANT:
        wave_classification = 'wave_dominant'
    elif max_wave_score >= WAVE_PRESENT_THRESHOLD:
        wave_classification = 'has_waves'
    else:
        wave_classification = 'no_waves'

    has_wave_behavior = wave_classification != 'no_waves'

    # Peak season
    peak_season = _determine_peak_season(phases, dominant_phases)

    return {
        'phases': phases,
        'max_wave_score': round(max_wave_score, 4),
        'total_wave_minutes': total_wave_minutes,
        'dominant_phases': dominant_phases,
        'wave_classification': wave_classification,
        'has_wave_behavior': has_wave_behavior,
        'peak_season': peak_season,
    }


def _determine_peak_season(phases: Dict[str, Dict],
                           dominant_phases: List[str]) -> Optional[str]:
    """Determine when wave behavior peaks: summer, winter, seasonal, year_round."""
    if not dominant_phases:
        return None

    # Aggregate monthly minutes across dominant phases
    total_monthly = {}
    for col in dominant_phases:
        monthly = phases[col].get('monthly_minutes', {})
        for m, mins in monthly.items():
            total_monthly[m] = total_monthly.get(m, 0) + mins

    total_mins = sum(total_monthly.values())
    if total_mins == 0:
        return None

    summer_mins = sum(total_monthly.get(m, 0) for m in SUMMER_MONTHS)
    winter_mins = sum(total_monthly.get(m, 0) for m in WINTER_MONTHS)

    summer_ratio = summer_mins / total_mins
    winter_ratio = winter_mins / total_mins

    if summer_ratio > 0.60:
        return 'summer'
    elif winter_ratio > 0.60:
        return 'winter'
    elif (summer_ratio + winter_ratio) > 0.80:
        return 'seasonal'
    else:
        return 'year_round'
