"""
Temporal pattern metrics for household data.

Analyzes day/night patterns, weekly patterns, and seasonal variations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_temporal_patterns(data: pd.DataFrame, phase_cols: list = None) -> Dict[str, Any]:
    """
    Calculate temporal patterns in power consumption.

    Args:
        data: DataFrame with timestamp and phase power columns
        phase_cols: List of phase column names

    Returns:
        Dictionary with temporal pattern metrics
    """
    if 'timestamp' not in data.columns:
        return {}

    if phase_cols is None:
        if 'w1' in data.columns:
            phase_cols = ['w1', 'w2', 'w3']
        elif '1' in data.columns:
            phase_cols = ['1', '2', '3']
        else:
            phase_cols = [c for c in data.columns if c not in ['timestamp', 'sum', 'year_month', 'date']]

    data = data.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([4, 5])  # Friday, Saturday in Israel
    data['month'] = data['timestamp'].dt.month

    # Day hours: 6:00-22:00, Night hours: 22:00-6:00
    data['is_day'] = (data['hour'] >= 6) & (data['hour'] < 22)

    metrics = {}

    for col in phase_cols:
        if col not in data.columns:
            continue

        prefix = f'phase_{col}'

        # Day vs Night
        day_mean = data.loc[data['is_day'], col].mean()
        night_mean = data.loc[~data['is_day'], col].mean()
        metrics[f'{prefix}_day_mean'] = day_mean
        metrics[f'{prefix}_night_mean'] = night_mean
        if day_mean > 0:
            metrics[f'{prefix}_night_day_ratio'] = night_mean / day_mean
        else:
            metrics[f'{prefix}_night_day_ratio'] = float('inf') if night_mean > 0 else 1.0

        # Weekday vs Weekend
        weekday_mean = data.loc[~data['is_weekend'], col].mean()
        weekend_mean = data.loc[data['is_weekend'], col].mean()
        metrics[f'{prefix}_weekday_mean'] = weekday_mean
        metrics[f'{prefix}_weekend_mean'] = weekend_mean
        if weekday_mean > 0:
            metrics[f'{prefix}_weekend_weekday_ratio'] = weekend_mean / weekday_mean
        else:
            metrics[f'{prefix}_weekend_weekday_ratio'] = 1.0

        # Hourly pattern (peak hours)
        hourly_mean = data.groupby('hour')[col].mean()
        metrics[f'{prefix}_peak_hour'] = hourly_mean.idxmax()
        metrics[f'{prefix}_peak_hour_mean'] = hourly_mean.max()
        metrics[f'{prefix}_min_hour'] = hourly_mean.idxmin()
        metrics[f'{prefix}_min_hour_mean'] = hourly_mean.min()

        # Hour-to-hour variability
        metrics[f'{prefix}_hourly_std'] = hourly_mean.std()
        if hourly_mean.mean() > 0:
            metrics[f'{prefix}_hourly_cv'] = hourly_mean.std() / hourly_mean.mean()
        else:
            metrics[f'{prefix}_hourly_cv'] = 0

        # Monthly pattern (seasonal)
        if data['month'].nunique() > 1:
            monthly_mean = data.groupby('month')[col].mean()
            metrics[f'{prefix}_peak_month'] = int(monthly_mean.idxmax())
            metrics[f'{prefix}_min_month'] = int(monthly_mean.idxmin())
            metrics[f'{prefix}_monthly_range'] = monthly_mean.max() - monthly_mean.min()

    # Total power patterns
    sum_cols = [c for c in phase_cols if c in data.columns]
    if sum_cols:
        data['total_power'] = data[sum_cols].sum(axis=1)

        day_total = data.loc[data['is_day'], 'total_power'].mean()
        night_total = data.loc[~data['is_day'], 'total_power'].mean()
        metrics['total_day_mean'] = day_total
        metrics['total_night_mean'] = night_total
        if day_total > 0:
            metrics['total_night_day_ratio'] = night_total / day_total

        hourly_total = data.groupby('hour')['total_power'].mean()
        metrics['total_peak_hour'] = int(hourly_total.idxmax())
        metrics['total_min_hour'] = int(hourly_total.idxmin())

    return metrics


def calculate_flat_segments(data: pd.DataFrame, phase_cols: list = None,
                            tolerance: float = 10) -> Dict[str, Any]:
    """
    Analyze flat (constant) power segments.

    Flat segments may indicate metering issues or standby power.

    Args:
        data: DataFrame with phase power columns
        phase_cols: List of phase column names
        tolerance: Maximum variation to consider segment as flat (watts)

    Returns:
        Dictionary with flat segment metrics
    """
    if phase_cols is None:
        if 'w1' in data.columns:
            phase_cols = ['w1', 'w2', 'w3']
        elif '1' in data.columns:
            phase_cols = ['1', '2', '3']
        else:
            phase_cols = [c for c in data.columns if c not in ['timestamp', 'sum']]

    metrics = {}

    for col in phase_cols:
        if col not in data.columns:
            continue

        phase_data = data[col].dropna().values
        if len(phase_data) < 2:
            continue

        # Find flat segments
        diffs = np.abs(np.diff(phase_data))
        is_flat = diffs <= tolerance

        # Count segments and their lengths
        segment_lengths = []
        current_length = 1

        for flat in is_flat:
            if flat:
                current_length += 1
            else:
                if current_length > 1:
                    segment_lengths.append(current_length)
                current_length = 1

        if current_length > 1:
            segment_lengths.append(current_length)

        prefix = f'phase_{col}'
        metrics[f'{prefix}_flat_segments_count'] = len(segment_lengths)
        if segment_lengths:
            metrics[f'{prefix}_max_flat_length'] = max(segment_lengths)
            metrics[f'{prefix}_avg_flat_length'] = np.mean(segment_lengths)
            metrics[f'{prefix}_total_flat_minutes'] = sum(segment_lengths)
            metrics[f'{prefix}_flat_pct'] = sum(segment_lengths) / len(phase_data) * 100
        else:
            metrics[f'{prefix}_max_flat_length'] = 0
            metrics[f'{prefix}_avg_flat_length'] = 0
            metrics[f'{prefix}_total_flat_minutes'] = 0
            metrics[f'{prefix}_flat_pct'] = 0

    return metrics
