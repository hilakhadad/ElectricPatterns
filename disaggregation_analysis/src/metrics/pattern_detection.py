"""
Pattern detection and clustering utilities.

Extracted from patterns.py -- contains recurring pattern detection,
event clustering, proximity stats, and time distribution analysis.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict


def _calculate_daily_stats(on_off_df: pd.DataFrame,
                           matches_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Calculate daily averages and statistics."""
    stats = {}

    if 'start' not in on_off_df.columns or on_off_df['start'].isna().all():
        return stats

    # Add date column
    on_off_df['date'] = on_off_df['start'].dt.date

    # Daily event counts
    daily_events = on_off_df.groupby('date').size()
    stats['total_days'] = len(daily_events)
    stats['avg_events_per_day'] = daily_events.mean()
    stats['min_events_per_day'] = daily_events.min()
    stats['max_events_per_day'] = daily_events.max()
    stats['std_events_per_day'] = daily_events.std()

    # ON events per day
    on_events = on_off_df[on_off_df['event'] == 'on'].groupby('date').size()
    stats['avg_on_events_per_day'] = on_events.mean() if len(on_events) > 0 else 0

    # OFF events per day
    off_events = on_off_df[on_off_df['event'] == 'off'].groupby('date').size()
    stats['avg_off_events_per_day'] = off_events.mean() if len(off_events) > 0 else 0

    # Matches per day
    if matches_df is not None and 'on_start' in matches_df.columns:
        matches_df['date'] = matches_df['on_start'].dt.date
        daily_matches = matches_df.groupby('date').size()
        stats['avg_matches_per_day'] = daily_matches.mean() if len(daily_matches) > 0 else 0
        stats['total_match_days'] = len(daily_matches)

        # Power matched per day
        if 'on_magnitude' in matches_df.columns:
            daily_power = matches_df.groupby('date')['on_magnitude'].sum().abs()
            stats['avg_power_matched_per_day'] = daily_power.mean()
    else:
        stats['avg_matches_per_day'] = 0

    return stats


def _find_recurring_events(on_off_df: pd.DataFrame,
                           time_tolerance_minutes: int = 30,
                           min_occurrences: int = 3) -> Dict[str, Any]:
    """
    Find events that recur at similar times across multiple days.

    Args:
        on_off_df: Events DataFrame
        time_tolerance_minutes: How close times must be to be considered "same time"
        min_occurrences: Minimum days an event must appear to be considered recurring

    Returns:
        Dictionary with recurring event statistics
    """
    result = {
        'recurring_patterns': [],
        'total_recurring_events': 0,
        'recurring_event_percentage': 0,
    }

    if 'start' not in on_off_df.columns or on_off_df['start'].isna().all():
        return result

    # Extract time components
    on_off_df = on_off_df.copy()
    on_off_df['time_minutes'] = on_off_df['start'].dt.hour * 60 + on_off_df['start'].dt.minute
    on_off_df['date'] = on_off_df['start'].dt.date

    # Group events by approximate time and magnitude (to identify same "appliance")
    # Round magnitude to nearest 100W for grouping
    on_off_df['magnitude_group'] = (on_off_df['magnitude'].abs() / 100).round() * 100

    # Find recurring patterns
    recurring_patterns = []
    events_in_patterns = set()

    # Group by phase, event type, and magnitude group
    for (phase, event_type, mag_group), group in on_off_df.groupby(['phase', 'event', 'magnitude_group']):
        if len(group) < min_occurrences:
            continue

        # Find time clusters
        times = group['time_minutes'].values
        dates = group['date'].values
        indices = group.index.values

        # Simple clustering: group times within tolerance
        time_clusters = _cluster_times(times, dates, indices, time_tolerance_minutes)

        for cluster in time_clusters:
            if len(cluster['dates']) >= min_occurrences:
                pattern = {
                    'phase': phase,
                    'event_type': event_type,
                    'magnitude': mag_group,
                    'avg_time': _minutes_to_time_str(cluster['avg_time']),
                    'time_range': f"{_minutes_to_time_str(cluster['min_time'])} - {_minutes_to_time_str(cluster['max_time'])}",
                    'occurrences': len(cluster['dates']),
                    'unique_days': len(set(cluster['dates'])),
                }
                recurring_patterns.append(pattern)
                events_in_patterns.update(cluster['indices'])

    # Sort by occurrences
    recurring_patterns.sort(key=lambda x: x['occurrences'], reverse=True)

    result['recurring_patterns'] = recurring_patterns  # Keep all patterns
    result['total_recurring_events'] = len(events_in_patterns)
    result['recurring_event_percentage'] = (
        len(events_in_patterns) / len(on_off_df) * 100 if len(on_off_df) > 0 else 0
    )
    result['pattern_count'] = len(recurring_patterns)

    return result


def _find_recurring_matches(matches_df: pd.DataFrame,
                            time_tolerance_minutes: int = 60,
                            duration_tolerance_pct: float = 0.10,
                            min_occurrences: int = 3,
                            min_duration_minutes: int = 20) -> Dict[str, Any]:
    """
    Find matched events (ON+OFF pairs) that recur at similar times.

    This analyzes MATCHES not individual events - so we see complete
    appliance usage cycles that repeat.

    Args:
        matches_df: Matches DataFrame with on_start, off_end, duration, etc.
        time_tolerance_minutes: How close start times must be to be considered same pattern
        duration_tolerance_pct: Duration tolerance as percentage (0.10 = ±10%)
        min_occurrences: Minimum days a match must appear to be considered recurring
        min_duration_minutes: Minimum duration in minutes to be considered (default 20)

    Returns:
        Dictionary with recurring match patterns including dates
    """
    result = {
        'patterns': [],
        'total_recurring': 0,
        'recurring_percentage': 0,
    }

    if matches_df is None or len(matches_df) == 0:
        return result

    if 'on_start' not in matches_df.columns:
        return result

    # Work with a copy
    df = matches_df.copy()

    # Filter by minimum duration (only consider long events)
    if 'duration' in df.columns and min_duration_minutes > 0:
        df = df[df['duration'] >= min_duration_minutes]
        if len(df) == 0:
            return result

    # Parse timestamps if needed
    if not pd.api.types.is_datetime64_any_dtype(df['on_start']):
        df['on_start'] = pd.to_datetime(df['on_start'], format='%d/%m/%Y %H:%M', errors='coerce')
    if 'off_end' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['off_end']):
        df['off_end'] = pd.to_datetime(df['off_end'], format='%d/%m/%Y %H:%M', errors='coerce')

    # Extract time components
    df['time_minutes'] = df['on_start'].dt.hour * 60 + df['on_start'].dt.minute
    df['date'] = df['on_start'].dt.date
    df['weekday'] = df['on_start'].dt.dayofweek  # 0=Monday, 6=Sunday

    # Round magnitude to nearest 100W for grouping
    if 'on_magnitude' in df.columns:
        df['magnitude_group'] = (df['on_magnitude'].abs() / 100).round() * 100
    else:
        df['magnitude_group'] = 0

    # Keep actual duration for ±10% tolerance matching (don't round to fixed bins)
    if 'duration' not in df.columns:
        df['duration'] = 0

    # Find recurring patterns
    recurring_patterns = []
    matches_in_patterns = set()

    # Group by phase and magnitude only (duration will be checked with ±10% tolerance)
    for (phase, mag_group), group in df.groupby(['phase', 'magnitude_group']):
        if len(group) < min_occurrences:
            continue

        # Find time clusters with duration tolerance
        times = group['time_minutes'].values
        dates = group['date'].values
        indices = group.index.values
        weekdays = group['weekday'].values
        durations = group['duration'].values

        # Cluster by start time AND duration (±10% tolerance)
        time_clusters = _cluster_times_with_duration(
            times, dates, indices, weekdays, durations,
            time_tolerance_minutes, duration_tolerance_pct
        )

        for cluster in time_clusters:
            if len(cluster['dates']) >= min_occurrences:
                # Calculate interval between occurrences
                sorted_dates = sorted(cluster['dates'])
                if len(sorted_dates) >= 2:
                    intervals_days = []
                    for i in range(1, len(sorted_dates)):
                        delta = (sorted_dates[i] - sorted_dates[i-1]).days
                        if delta > 0:
                            intervals_days.append(delta)

                    if intervals_days:
                        avg_interval = np.mean(intervals_days)
                        # Determine pattern type
                        if 0.8 <= avg_interval <= 1.2:
                            interval_type = "daily"
                        elif 6 <= avg_interval <= 8:
                            interval_type = "weekly"
                        elif 13 <= avg_interval <= 15:
                            interval_type = "bi-weekly"
                        elif 28 <= avg_interval <= 32:
                            interval_type = "monthly"
                        else:
                            interval_type = f"every ~{avg_interval:.0f} days"
                    else:
                        avg_interval = 0
                        interval_type = "unknown"
                else:
                    avg_interval = 0
                    interval_type = "unknown"

                # Check if it's weekday-specific
                weekday_counts = {}
                for wd in cluster['weekdays']:
                    weekday_counts[wd] = weekday_counts.get(wd, 0) + 1
                dominant_weekday = max(weekday_counts, key=weekday_counts.get) if weekday_counts else None
                weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

                # Format dates for display
                date_strs = [d.strftime('%Y-%m-%d') for d in sorted(cluster['dates'])]

                # Calculate average duration from cluster (not fixed bins)
                avg_duration = int(round(cluster['avg_duration']))

                pattern = {
                    'phase': phase,
                    'magnitude': int(mag_group),
                    'duration_minutes': avg_duration,
                    'avg_start_time': _minutes_to_time_str(cluster['avg_time']),
                    'time_range': f"{_minutes_to_time_str(cluster['min_time'])} - {_minutes_to_time_str(cluster['max_time'])}",
                    'occurrences': len(cluster['dates']),
                    'unique_days': len(set(cluster['dates'])),
                    'interval_type': interval_type,
                    'avg_interval_days': round(avg_interval, 1) if avg_interval > 0 else None,
                    'dominant_weekday': weekday_names[dominant_weekday] if dominant_weekday is not None else None,
                    'dates': date_strs,  # List of all dates when this pattern occurred
                    'first_date': date_strs[0] if date_strs else None,
                    'last_date': date_strs[-1] if date_strs else None,
                }
                recurring_patterns.append(pattern)
                matches_in_patterns.update(cluster['indices'])

    # Sort by occurrences
    recurring_patterns.sort(key=lambda x: x['occurrences'], reverse=True)

    result['patterns'] = recurring_patterns  # Keep all patterns
    result['total_recurring'] = len(matches_in_patterns)
    result['recurring_percentage'] = (
        len(matches_in_patterns) / len(df) * 100 if len(df) > 0 else 0
    )
    result['pattern_count'] = len(recurring_patterns)

    return result


def _cluster_times_with_dates(times: np.ndarray, dates: np.ndarray, indices: np.ndarray,
                               weekdays: np.ndarray, tolerance: int) -> List[Dict]:
    """Cluster times that are within tolerance, tracking dates and weekdays."""
    if len(times) == 0:
        return []

    # Sort by time
    sorted_idx = np.argsort(times)
    times = times[sorted_idx]
    dates = dates[sorted_idx]
    indices = indices[sorted_idx]
    weekdays = weekdays[sorted_idx]

    clusters = []
    current_cluster = {
        'times': [times[0]],
        'dates': [dates[0]],
        'indices': [indices[0]],
        'weekdays': [weekdays[0]]
    }

    for i in range(1, len(times)):
        if times[i] - np.mean(current_cluster['times']) <= tolerance:
            current_cluster['times'].append(times[i])
            current_cluster['dates'].append(dates[i])
            current_cluster['indices'].append(indices[i])
            current_cluster['weekdays'].append(weekdays[i])
        else:
            # Save current cluster and start new one
            if len(current_cluster['times']) > 0:
                clusters.append({
                    'avg_time': np.mean(current_cluster['times']),
                    'min_time': min(current_cluster['times']),
                    'max_time': max(current_cluster['times']),
                    'dates': current_cluster['dates'],
                    'indices': current_cluster['indices'],
                    'weekdays': current_cluster['weekdays']
                })
            current_cluster = {
                'times': [times[i]],
                'dates': [dates[i]],
                'indices': [indices[i]],
                'weekdays': [weekdays[i]]
            }

    # Don't forget last cluster
    if len(current_cluster['times']) > 0:
        clusters.append({
            'avg_time': np.mean(current_cluster['times']),
            'min_time': min(current_cluster['times']),
            'max_time': max(current_cluster['times']),
            'dates': current_cluster['dates'],
            'indices': current_cluster['indices'],
            'weekdays': current_cluster['weekdays']
        })

    return clusters


def _cluster_times_with_duration(times: np.ndarray, dates: np.ndarray, indices: np.ndarray,
                                  weekdays: np.ndarray, durations: np.ndarray,
                                  time_tolerance: int, duration_tolerance_pct: float) -> List[Dict]:
    """
    Cluster times that are within tolerance, also checking duration compatibility.

    Duration tolerance uses ±X% of the cluster's average duration.
    For example, with 10% tolerance and avg duration of 60 min, accepts 54-66 min.

    Args:
        times: Array of start times (minutes from midnight)
        dates: Array of dates
        indices: Array of DataFrame indices
        weekdays: Array of weekday numbers (0=Monday)
        durations: Array of durations in minutes
        time_tolerance: Max time difference in minutes
        duration_tolerance_pct: Duration tolerance as fraction (0.10 = ±10%)

    Returns:
        List of clusters with avg_time, avg_duration, dates, etc.
    """
    if len(times) == 0:
        return []

    # Sort by time
    sorted_idx = np.argsort(times)
    times = times[sorted_idx]
    dates = dates[sorted_idx]
    indices = indices[sorted_idx]
    weekdays = weekdays[sorted_idx]
    durations = durations[sorted_idx]

    clusters = []
    current_cluster = {
        'times': [times[0]],
        'dates': [dates[0]],
        'indices': [indices[0]],
        'weekdays': [weekdays[0]],
        'durations': [durations[0]]
    }

    for i in range(1, len(times)):
        # Check time tolerance
        time_ok = times[i] - np.mean(current_cluster['times']) <= time_tolerance

        # Check duration tolerance (±X% of average duration)
        avg_dur = np.mean(current_cluster['durations'])
        if avg_dur > 0:
            dur_tolerance = avg_dur * duration_tolerance_pct
            duration_ok = abs(durations[i] - avg_dur) <= dur_tolerance
        else:
            # If average duration is 0, only accept other 0 durations
            duration_ok = durations[i] == 0

        if time_ok and duration_ok:
            current_cluster['times'].append(times[i])
            current_cluster['dates'].append(dates[i])
            current_cluster['indices'].append(indices[i])
            current_cluster['weekdays'].append(weekdays[i])
            current_cluster['durations'].append(durations[i])
        else:
            # Save current cluster and start new one
            if len(current_cluster['times']) > 0:
                clusters.append({
                    'avg_time': np.mean(current_cluster['times']),
                    'min_time': min(current_cluster['times']),
                    'max_time': max(current_cluster['times']),
                    'avg_duration': np.mean(current_cluster['durations']),
                    'dates': current_cluster['dates'],
                    'indices': current_cluster['indices'],
                    'weekdays': current_cluster['weekdays']
                })
            current_cluster = {
                'times': [times[i]],
                'dates': [dates[i]],
                'indices': [indices[i]],
                'weekdays': [weekdays[i]],
                'durations': [durations[i]]
            }

    # Don't forget last cluster
    if len(current_cluster['times']) > 0:
        clusters.append({
            'avg_time': np.mean(current_cluster['times']),
            'min_time': min(current_cluster['times']),
            'max_time': max(current_cluster['times']),
            'avg_duration': np.mean(current_cluster['durations']),
            'dates': current_cluster['dates'],
            'indices': current_cluster['indices'],
            'weekdays': current_cluster['weekdays']
        })

    return clusters


def _cluster_times(times: np.ndarray, dates: np.ndarray, indices: np.ndarray,
                   tolerance: int) -> List[Dict]:
    """Cluster times that are within tolerance of each other."""
    if len(times) == 0:
        return []

    # Sort by time
    sorted_idx = np.argsort(times)
    times = times[sorted_idx]
    dates = dates[sorted_idx]
    indices = indices[sorted_idx]

    clusters = []
    current_cluster = {
        'times': [times[0]],
        'dates': [dates[0]],
        'indices': [indices[0]]
    }

    for i in range(1, len(times)):
        if times[i] - np.mean(current_cluster['times']) <= tolerance:
            current_cluster['times'].append(times[i])
            current_cluster['dates'].append(dates[i])
            current_cluster['indices'].append(indices[i])
        else:
            # Save current cluster and start new one
            if len(current_cluster['times']) > 0:
                clusters.append({
                    'avg_time': np.mean(current_cluster['times']),
                    'min_time': min(current_cluster['times']),
                    'max_time': max(current_cluster['times']),
                    'dates': current_cluster['dates'],
                    'indices': current_cluster['indices']
                })
            current_cluster = {
                'times': [times[i]],
                'dates': [dates[i]],
                'indices': [indices[i]]
            }

    # Don't forget last cluster
    if len(current_cluster['times']) > 0:
        clusters.append({
            'avg_time': np.mean(current_cluster['times']),
            'min_time': min(current_cluster['times']),
            'max_time': max(current_cluster['times']),
            'dates': current_cluster['dates'],
            'indices': current_cluster['indices']
        })

    return clusters


def _minutes_to_time_str(minutes: float) -> str:
    """Convert minutes from midnight to HH:MM string."""
    h = int(minutes // 60) % 24
    m = int(minutes % 60)
    return f"{h:02d}:{m:02d}"


def _calculate_proximity_stats(on_off_df: pd.DataFrame,
                               proximity_threshold_minutes: int = 5) -> Dict[str, Any]:
    """
    Calculate statistics about events that occur close together.

    Uses binary search (searchsorted) for O(n log n) instead of O(n²).

    Args:
        on_off_df: Events DataFrame
        proximity_threshold_minutes: Events within this time are considered "close"

    Returns:
        Dictionary with proximity statistics
    """
    stats = {
        'events_with_close_neighbors': 0,
        'percentage_with_close_neighbors': 0,
        'avg_neighbors_per_event': 0,
        'clusters': [],
    }

    if 'start' not in on_off_df.columns or on_off_df['start'].isna().all():
        return stats

    # Sort by start time
    df = on_off_df.sort_values('start').reset_index(drop=True)
    threshold = pd.Timedelta(minutes=proximity_threshold_minutes)

    # Convert to numpy array of timestamps (nanoseconds) for fast searchsorted
    starts = df['start'].values.astype('int64')  # nanoseconds
    threshold_ns = int(threshold.total_seconds() * 1e9)

    # Use searchsorted for O(n log n) neighbor counting
    # For each event, find range of events within [start - threshold, start + threshold]
    left_indices = np.searchsorted(starts, starts - threshold_ns, side='left')
    right_indices = np.searchsorted(starts, starts + threshold_ns, side='right')

    # Neighbor count = events in range minus self
    neighbor_counts = (right_indices - left_indices) - 1
    events_with_neighbors = int((neighbor_counts > 0).sum())

    stats['events_with_close_neighbors'] = events_with_neighbors
    stats['percentage_with_close_neighbors'] = (
        events_with_neighbors / len(df) * 100 if len(df) > 0 else 0
    )
    stats['avg_neighbors_per_event'] = float(np.mean(neighbor_counts)) if len(neighbor_counts) > 0 else 0

    # Find event clusters (groups of events close together)
    clusters = _find_event_clusters(df, threshold)
    stats['cluster_count'] = len(clusters)
    stats['avg_cluster_size'] = np.mean([c['size'] for c in clusters]) if clusters else 0
    stats['max_cluster_size'] = max([c['size'] for c in clusters]) if clusters else 0

    return stats


def _find_event_clusters(df: pd.DataFrame, threshold: pd.Timedelta) -> List[Dict]:
    """Find clusters of events that occur close together.

    Optimized: since data is sorted, only need to check distance to previous event.
    """
    if len(df) == 0:
        return []

    df = df.sort_values('start').reset_index(drop=True)

    # Vectorized: calculate time diff to previous event
    time_diffs = df['start'].diff()

    # An event starts a new cluster if gap to previous > threshold (or it's the first)
    new_cluster_mask = (time_diffs > threshold) | time_diffs.isna()

    # Assign cluster IDs
    cluster_ids = new_cluster_mask.cumsum()

    # Group by cluster and build cluster info
    clusters = []
    for cluster_id, group in df.groupby(cluster_ids):
        if len(group) > 1:
            clusters.append({
                'size': len(group),
                'start': group['start'].min(),
                'end': group['start'].max(),
                'duration_minutes': (group['start'].max() - group['start'].min()).total_seconds() / 60,
                'phases': group['phase'].unique().tolist() if 'phase' in group.columns else [],
            })

    return clusters


def _calculate_time_distribution(on_off_df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate distribution of events across time of day."""
    distribution = {
        'by_hour': {},
        'by_period': {},
        'by_period_on': {},
        'by_period_off': {},
        'peak_hour': None,
        'quiet_hour': None,
    }

    if 'start' not in on_off_df.columns or on_off_df['start'].isna().all():
        return distribution

    # Extract hour
    on_off_df = on_off_df.copy()
    on_off_df['hour'] = on_off_df['start'].dt.hour

    # Count by hour
    hourly_counts = on_off_df.groupby('hour').size()
    distribution['by_hour'] = hourly_counts.to_dict()

    # Find peak and quiet hours
    if len(hourly_counts) > 0:
        distribution['peak_hour'] = int(hourly_counts.idxmax())
        distribution['quiet_hour'] = int(hourly_counts.idxmin())
        distribution['peak_hour_count'] = int(hourly_counts.max())
        distribution['quiet_hour_count'] = int(hourly_counts.min())

    # Group by period
    periods = {
        'night': (0, 6),      # 00:00-06:00
        'morning': (6, 12),   # 06:00-12:00
        'afternoon': (12, 18),  # 12:00-18:00
        'evening': (18, 24),  # 18:00-24:00
    }

    # Separate ON and OFF events
    has_event_col = 'event' in on_off_df.columns
    on_events = on_off_df[on_off_df['event'] == 'on'] if has_event_col else on_off_df
    off_events = on_off_df[on_off_df['event'] == 'off'] if has_event_col else pd.DataFrame()

    for period_name, (start_h, end_h) in periods.items():
        # Total count
        count = on_off_df[(on_off_df['hour'] >= start_h) & (on_off_df['hour'] < end_h)].shape[0]
        distribution['by_period'][period_name] = count

        # ON events count
        on_count = on_events[(on_events['hour'] >= start_h) & (on_events['hour'] < end_h)].shape[0] if len(on_events) > 0 else 0
        distribution['by_period_on'][period_name] = on_count

        # OFF events count
        off_count = off_events[(off_events['hour'] >= start_h) & (off_events['hour'] < end_h)].shape[0] if len(off_events) > 0 else 0
        distribution['by_period_off'][period_name] = off_count

    return distribution




def find_periodic_patterns(experiment_dir: Path, house_id: str,
                           run_number: int = 0) -> Dict[str, Any]:
    """
    Find patterns that occur at regular intervals (not just daily).

    Looks for patterns like:
    - Every X hours
    - Every X days
    - Weekly patterns

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number

    Returns:
        Dictionary with periodic pattern analysis
    """
    from metrics.patterns import _get_house_dir, _load_monthly_files
    house_dir = _get_house_dir(experiment_dir, house_id, run_number)

    on_off_df = _load_monthly_files(house_dir, "on_off", "on_off_*.pkl")
    if on_off_df is None:
        return {'error': 'No data'}

    # Parse timestamps
    on_off_df['start'] = pd.to_datetime(on_off_df['start'], format='%d/%m/%Y %H:%M', errors='coerce')

    results = {
        'hourly_patterns': [],
        'daily_patterns': [],
        'weekly_patterns': [],
    }

    # Group by phase and magnitude
    on_off_df['magnitude_group'] = (on_off_df['magnitude'].abs() / 100).round() * 100

    for (phase, mag_group), group in on_off_df.groupby(['phase', 'magnitude_group']):
        if len(group) < 5:
            continue

        # Sort by time
        group = group.sort_values('start')
        timestamps = group['start'].values

        # Calculate intervals between consecutive events
        intervals = []
        for i in range(1, len(timestamps)):
            delta = (timestamps[i] - timestamps[i-1]) / np.timedelta64(1, 'h')  # hours
            if 0 < delta < 24 * 7:  # Up to 7 days
                intervals.append(delta)

        if len(intervals) < 3:
            continue

        intervals = np.array(intervals)

        # Look for common intervals
        # Check for hourly patterns (every 1-6 hours)
        for target_hours in [1, 2, 3, 4, 6, 8, 12]:
            matches = np.sum(np.abs(intervals - target_hours) < 0.5)  # Within 30 min
            if matches >= 3:
                results['hourly_patterns'].append({
                    'phase': phase,
                    'magnitude': mag_group,
                    'interval_hours': target_hours,
                    'occurrences': int(matches),
                    'confidence': matches / len(intervals),
                })

        # Check for daily patterns (every 24 hours +/- 2 hours)
        daily_matches = np.sum(np.abs(intervals - 24) < 2)
        if daily_matches >= 3:
            results['daily_patterns'].append({
                'phase': phase,
                'magnitude': mag_group,
                'occurrences': int(daily_matches),
                'confidence': daily_matches / len(intervals),
            })

        # Check for weekly patterns
        weekly_matches = np.sum(np.abs(intervals - 24*7) < 12)  # Within 12 hours
        if weekly_matches >= 2:
            results['weekly_patterns'].append({
                'phase': phase,
                'magnitude': mag_group,
                'occurrences': int(weekly_matches),
                'confidence': weekly_matches / len(intervals),
            })

    return results


