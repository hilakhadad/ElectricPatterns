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
        if not hourly_mean.isna().all() and len(hourly_mean.dropna()) > 0:
            metrics[f'{prefix}_peak_hour'] = int(hourly_mean.idxmax())
            metrics[f'{prefix}_peak_hour_mean'] = hourly_mean.max()
            metrics[f'{prefix}_min_hour'] = int(hourly_mean.idxmin())
            metrics[f'{prefix}_min_hour_mean'] = hourly_mean.min()
        else:
            metrics[f'{prefix}_peak_hour'] = 0
            metrics[f'{prefix}_peak_hour_mean'] = 0
            metrics[f'{prefix}_min_hour'] = 0
            metrics[f'{prefix}_min_hour_mean'] = 0

        # Hour-to-hour variability
        metrics[f'{prefix}_hourly_std'] = hourly_mean.std()
        if hourly_mean.mean() > 0:
            metrics[f'{prefix}_hourly_cv'] = hourly_mean.std() / hourly_mean.mean()
        else:
            metrics[f'{prefix}_hourly_cv'] = 0

        # Store detailed hourly pattern for charts
        hourly_std = data.groupby('hour')[col].std()
        metrics[f'{prefix}_hourly_pattern'] = {
            'hours': list(range(24)),
            'mean': [float(hourly_mean.get(h, 0)) if not pd.isna(hourly_mean.get(h, 0)) else 0 for h in range(24)],
            'std': [float(hourly_std.get(h, 0)) if not pd.isna(hourly_std.get(h, 0)) else 0 for h in range(24)]
        }

        # Weekly pattern (by day of week)
        weekly_mean = data.groupby('day_of_week')[col].mean()
        metrics[f'{prefix}_weekly_pattern'] = {
            'days': list(range(7)),
            'mean': [float(weekly_mean.get(d, 0)) if not pd.isna(weekly_mean.get(d, 0)) else 0 for d in range(7)]
        }

        # Monthly pattern (seasonal)
        if data['month'].nunique() > 1:
            monthly_mean = data.groupby('month')[col].mean()
            if not monthly_mean.isna().all() and len(monthly_mean.dropna()) > 0:
                metrics[f'{prefix}_peak_month'] = int(monthly_mean.idxmax())
                metrics[f'{prefix}_min_month'] = int(monthly_mean.idxmin())
                metrics[f'{prefix}_monthly_range'] = monthly_mean.max() - monthly_mean.min()
            else:
                metrics[f'{prefix}_peak_month'] = 1
                metrics[f'{prefix}_min_month'] = 1
                metrics[f'{prefix}_monthly_range'] = 0
            # Store detailed monthly pattern for charts
            metrics[f'{prefix}_monthly_pattern'] = {
                'months': sorted(monthly_mean.index.tolist()),
                'mean': [float(monthly_mean.get(m, 0)) if not pd.isna(monthly_mean.get(m, 0)) else 0 for m in sorted(monthly_mean.index)]
            }

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
        if not hourly_total.isna().all() and len(hourly_total.dropna()) > 0:
            metrics['total_peak_hour'] = int(hourly_total.idxmax())
            metrics['total_min_hour'] = int(hourly_total.idxmin())
        else:
            metrics['total_peak_hour'] = 0
            metrics['total_min_hour'] = 0

        # Store detailed total hourly pattern for charts
        hourly_total_std = data.groupby('hour')['total_power'].std()
        metrics['total_hourly_pattern'] = {
            'hours': list(range(24)),
            'mean': [float(hourly_total.get(h, 0)) if not pd.isna(hourly_total.get(h, 0)) else 0 for h in range(24)],
            'std': [float(hourly_total_std.get(h, 0)) if not pd.isna(hourly_total_std.get(h, 0)) else 0 for h in range(24)]
        }

        # Total weekly pattern
        weekly_total = data.groupby('day_of_week')['total_power'].mean()
        metrics['total_weekly_pattern'] = {
            'days': list(range(7)),
            'mean': [float(weekly_total.get(d, 0)) if not pd.isna(weekly_total.get(d, 0)) else 0 for d in range(7)]
        }

        # Total monthly pattern
        if data['month'].nunique() > 1:
            monthly_total = data.groupby('month')['total_power'].mean()
            metrics['total_monthly_pattern'] = {
                'months': sorted(monthly_total.index.tolist()),
                'mean': [float(monthly_total.get(m, 0)) if not pd.isna(monthly_total.get(m, 0)) else 0 for m in sorted(monthly_total.index)]
            }

        # Heatmap data: hour x day_of_week matrix
        heatmap_data = data.groupby(['day_of_week', 'hour'])['total_power'].mean().unstack(fill_value=0)
        metrics['power_heatmap'] = {
            'days': list(range(7)),
            'hours': list(range(24)),
            'values': [[float(heatmap_data.loc[d, h]) if (d in heatmap_data.index and h in heatmap_data.columns) else 0
                       for h in range(24)] for d in range(7)]
        }

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


def calculate_temporal_patterns_by_period(data: pd.DataFrame, phase_cols: list = None) -> Dict[str, Any]:
    """
    Calculate temporal patterns grouped by year and month.

    Args:
        data: DataFrame with timestamp and phase power columns
        phase_cols: List of phase column names

    Returns:
        Dictionary with patterns per year and per month
    """
    if 'timestamp' not in data.columns:
        return {'by_year': {}}

    if phase_cols is None:
        if 'w1' in data.columns:
            phase_cols = ['w1', 'w2', 'w3']
        elif '1' in data.columns:
            phase_cols = ['1', '2', '3']
        else:
            phase_cols = [c for c in data.columns if c not in ['timestamp', 'sum', 'year_month', 'date']]

    data = data.copy()
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['year'] = data['timestamp'].dt.year
    data['month'] = data['timestamp'].dt.month
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    data['is_day'] = (data['hour'] >= 6) & (data['hour'] < 22)

    # Calculate total power
    sum_cols = [c for c in phase_cols if c in data.columns]
    if sum_cols:
        data['total_power'] = data[sum_cols].sum(axis=1)
    else:
        data['total_power'] = 0

    result = {'by_year': {}}

    # Get unique years
    years = sorted(data['year'].unique())

    for year in years:
        year_data = data[data['year'] == year]
        year_metrics = {
            'year': int(year),
            'row_count': len(year_data),
        }

        # Calculate days and coverage for this year
        year_data_dates = year_data['timestamp'].dt.date.nunique()
        year_metrics['days'] = year_data_dates

        # Expected minutes in this year (accounting for partial years)
        first_date = year_data['timestamp'].min()
        last_date = year_data['timestamp'].max()
        expected_minutes = int((last_date - first_date).total_seconds() / 60) + 1
        year_metrics['coverage_ratio'] = min(len(year_data) / expected_minutes, 1.0) if expected_minutes > 0 else 0

        # Average power
        year_metrics['avg_power'] = float(year_data['total_power'].mean())
        year_metrics['max_power'] = float(year_data['total_power'].max())

        # Day/Night
        day_mean = year_data.loc[year_data['is_day'], 'total_power'].mean()
        night_mean = year_data.loc[~year_data['is_day'], 'total_power'].mean()
        year_metrics['day_mean'] = float(day_mean) if not pd.isna(day_mean) else 0
        year_metrics['night_mean'] = float(night_mean) if not pd.isna(night_mean) else 0

        # Hourly pattern for this year
        hourly_mean = year_data.groupby('hour')['total_power'].mean()
        hourly_std = year_data.groupby('hour')['total_power'].std()
        year_metrics['hourly_pattern'] = {
            'hours': list(range(24)),
            'mean': [float(hourly_mean.get(h, 0)) if not pd.isna(hourly_mean.get(h, 0)) else 0 for h in range(24)],
            'std': [float(hourly_std.get(h, 0)) if not pd.isna(hourly_std.get(h, 0)) else 0 for h in range(24)]
        }

        # Weekly pattern for this year
        weekly_mean = year_data.groupby('day_of_week')['total_power'].mean()
        year_metrics['weekly_pattern'] = {
            'days': list(range(7)),
            'mean': [float(weekly_mean.get(d, 0)) if not pd.isna(weekly_mean.get(d, 0)) else 0 for d in range(7)]
        }

        # Heatmap for this year
        heatmap_data = year_data.groupby(['day_of_week', 'hour'])['total_power'].mean().unstack(fill_value=0)
        year_metrics['power_heatmap'] = {
            'days': list(range(7)),
            'hours': list(range(24)),
            'values': [[float(heatmap_data.loc[d, h]) if (d in heatmap_data.index and h in heatmap_data.columns) else 0
                       for h in range(24)] for d in range(7)]
        }

        # Per-month metrics within this year
        year_metrics['months'] = {}
        months_in_year = sorted(year_data['month'].unique())

        for month in months_in_year:
            month_data = year_data[year_data['month'] == month]
            month_metrics = {
                'month': int(month),
                'row_count': len(month_data),
            }

            # Days in this month
            month_dates = month_data['timestamp'].dt.date.nunique()
            month_metrics['days'] = month_dates

            # Coverage for this month
            # Get days in month from first timestamp
            days_in_month = month_data['timestamp'].iloc[0].days_in_month if len(month_data) > 0 else 30
            expected_month_minutes = days_in_month * 24 * 60
            month_metrics['coverage_ratio'] = min(len(month_data) / expected_month_minutes, 1.0)

            # Average power
            month_metrics['avg_power'] = float(month_data['total_power'].mean())

            # Day/Night
            day_m = month_data.loc[month_data['is_day'], 'total_power'].mean()
            night_m = month_data.loc[~month_data['is_day'], 'total_power'].mean()
            month_metrics['day_mean'] = float(day_m) if not pd.isna(day_m) else 0
            month_metrics['night_mean'] = float(night_m) if not pd.isna(night_m) else 0

            # Hourly pattern for this month (for mini chart)
            hourly_month = month_data.groupby('hour')['total_power'].mean()
            month_metrics['hourly_pattern'] = {
                'hours': list(range(24)),
                'mean': [float(hourly_month.get(h, 0)) if not pd.isna(hourly_month.get(h, 0)) else 0 for h in range(24)]
            }

            year_metrics['months'][int(month)] = month_metrics

        result['by_year'][int(year)] = year_metrics

    return result
