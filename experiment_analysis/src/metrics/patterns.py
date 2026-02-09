"""
Event pattern analysis for experiment results.

Analyzes recurring patterns, proximity between events, and daily statistics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict


def _get_house_dir(experiment_dir: Path, house_id: str, run_number: int) -> Path:
    """Get the house directory, supporting both old and new structures."""
    # Try new structure first: experiment_dir/run_N/house_X/
    new_dir = experiment_dir / f"run_{run_number}" / f"house_{house_id}"
    if new_dir.exists():
        return new_dir
    # Fall back to old structure: experiment_dir/house_X/run_N/house_X/
    return experiment_dir / f"house_{house_id}" / f"run_{run_number}" / f"house_{house_id}"


def _load_monthly_files(house_dir: Path, subfolder: str, pattern: str):
    """Load and concatenate monthly files from a subfolder or fallback to direct files."""
    subdir = house_dir / subfolder
    if subdir.exists():
        files = sorted(subdir.glob(pattern))
        if files:
            return pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    # Fallback: try files directly in house_dir
    files = list(house_dir.glob(pattern))
    if files:
        return pd.read_pickle(files[0])
    return None


def calculate_pattern_metrics(experiment_dir: Path, house_id: str,
                              run_number: int = 0) -> Dict[str, Any]:
    """
    Calculate event pattern metrics for a house.

    Supports both old (single file) and new (monthly subfolder) structures.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Iteration number (default 0)

    Returns:
        Dictionary with pattern metrics
    """
    metrics = {
        'house_id': house_id,
        'run_number': run_number,
    }

    house_dir = _get_house_dir(experiment_dir, house_id, run_number)

    if not house_dir.exists():
        return metrics

    # Load on_off events
    on_off_df = _load_monthly_files(house_dir, "on_off", "on_off_*.pkl")
    if on_off_df is None:
        return metrics

    # Parse timestamps
    for col in ['start', 'end']:
        if col in on_off_df.columns:
            on_off_df[col] = pd.to_datetime(on_off_df[col], format='%d/%m/%Y %H:%M', errors='coerce')

    # Load matches
    matches_df = _load_monthly_files(house_dir, "matches", f"matches_{house_id}_*.pkl")
    if matches_df is not None:
        for col in ['on_start', 'on_end', 'off_start', 'off_end']:
            if col in matches_df.columns:
                matches_df[col] = pd.to_datetime(matches_df[col], format='%d/%m/%Y %H:%M', errors='coerce')

    # Calculate daily statistics
    metrics['daily_stats'] = _calculate_daily_stats(on_off_df, matches_df)

    # NOTE: recurring_events and proximity_stats removed - not displayed in HTML
    # This saves significant computation time

    # Calculate recurring MATCHES (ON+OFF pairs that repeat regularly)
    if matches_df is not None and len(matches_df) > 0:
        metrics['recurring_matches'] = _find_recurring_matches(matches_df)
    else:
        metrics['recurring_matches'] = {'patterns': [], 'total_recurring': 0}

    # Calculate time-of-day distribution
    metrics['time_distribution'] = _calculate_time_distribution(on_off_df)

    return metrics


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


def get_recurring_patterns_summary(experiment_dir: Path, house_id: str,
                                    run_number: int = 0,
                                    min_occurrences: int = 5) -> str:
    """
    Generate a formatted summary of recurring patterns for a house.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number
        min_occurrences: Minimum occurrences to consider recurring

    Returns:
        Formatted summary string
    """
    metrics = calculate_pattern_metrics(experiment_dir, house_id, run_number)

    lines = []
    lines.append(f"Recurring Patterns for House {house_id}")
    lines.append("=" * 60)

    recurring = metrics.get('recurring_events', {})
    patterns = recurring.get('recurring_patterns', [])

    if not patterns:
        lines.append("No recurring patterns found.")
        return '\n'.join(lines)

    # Filter to significant patterns
    significant_patterns = [p for p in patterns if p.get('occurrences', 0) >= min_occurrences]

    if not significant_patterns:
        lines.append(f"No patterns with {min_occurrences}+ occurrences found.")
        return '\n'.join(lines)

    lines.append(f"Found {len(significant_patterns)} recurring patterns (min {min_occurrences} occurrences):")
    lines.append(f"\n{'Phase':<6} {'Type':<5} {'Power':<8} {'Time':<12} {'Range':<17} {'Days':<6}")
    lines.append("-" * 60)

    # Sort by occurrences
    for p in sorted(significant_patterns, key=lambda x: x.get('occurrences', 0), reverse=True)[:20]:
        phase = p.get('phase', '?')
        event_type = p.get('event_type', '?')[:4]
        magnitude = f"{p.get('magnitude', 0):.0f}W"
        avg_time = p.get('avg_time', '?')
        time_range = p.get('time_range', '?')
        occurrences = p.get('occurrences', 0)
        unique_days = p.get('unique_days', 0)

        lines.append(f"{phase:<6} {event_type:<5} {magnitude:<8} {avg_time:<12} {time_range:<17} {unique_days:<6}")

    # Summary statistics
    total_recurring = recurring.get('total_recurring_events', 0)
    pct_recurring = recurring.get('recurring_event_percentage', 0)

    lines.append("-" * 60)
    lines.append(f"Total events in patterns: {total_recurring} ({pct_recurring:.1f}% of all events)")

    # Group by phase
    phase_patterns = {}
    for p in significant_patterns:
        phase = p.get('phase', '?')
        if phase not in phase_patterns:
            phase_patterns[phase] = 0
        phase_patterns[phase] += 1

    lines.append(f"\nPatterns by phase: " + ", ".join(f"{k}: {v}" for k, v in sorted(phase_patterns.items())))

    return '\n'.join(lines)


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


def _group_activations_into_sessions(activations: List[Dict],
                                       gap_threshold_minutes: int = 60) -> List[Dict]:
    """
    Group individual activations into sessions based on time gaps.

    AC compressors cycle multiple times during a cooling session (4-5 min ON/OFF).
    These cycles should be grouped into a single "session" - from first event
    until last event before a gap > threshold.

    Args:
        activations: List of activation dicts with 'date', 'on_time', 'off_time',
                    'duration_minutes', 'magnitude', etc.
        gap_threshold_minutes: Gap between activations to start a new session (default 60 min)

    Returns:
        List of session dicts with aggregated data
    """
    if not activations:
        return []

    # Sort activations by datetime
    def get_datetime(act):
        try:
            date_str = act.get('date', '')
            time_str = act.get('on_time', '00:00')
            return pd.to_datetime(f"{date_str} {time_str}")
        except:
            return pd.NaT

    sorted_acts = sorted(activations, key=lambda x: get_datetime(x))

    sessions = []
    current_session = {
        'activations': [sorted_acts[0]],
        'start_datetime': get_datetime(sorted_acts[0]),
    }

    gap_threshold = pd.Timedelta(minutes=gap_threshold_minutes)

    for i in range(1, len(sorted_acts)):
        act = sorted_acts[i]
        act_datetime = get_datetime(act)

        # Get the end time of the last activation in current session
        last_act = current_session['activations'][-1]
        last_end_time = last_act.get('off_time', last_act.get('on_time', '00:00'))
        last_date = last_act.get('date', '')

        try:
            last_datetime = pd.to_datetime(f"{last_date} {last_end_time}")
        except:
            last_datetime = get_datetime(last_act)

        # Check if this activation starts more than gap_threshold after last ended
        if pd.notna(act_datetime) and pd.notna(last_datetime):
            time_gap = act_datetime - last_datetime

            if time_gap > gap_threshold:
                # Start a new session - save current one first
                sessions.append(_summarize_session(current_session))
                current_session = {
                    'activations': [act],
                    'start_datetime': act_datetime,
                }
            else:
                # Continue current session
                current_session['activations'].append(act)
        else:
            # Can't determine gap - add to current session
            current_session['activations'].append(act)

    # Don't forget the last session
    sessions.append(_summarize_session(current_session))

    return sessions


def _summarize_session(session: Dict) -> Dict:
    """
    Summarize a session of activations into a single record.

    Args:
        session: Dict with 'activations' list and 'start_datetime'

    Returns:
        Session summary dict
    """
    activations = session['activations']
    if not activations:
        return {}

    first_act = activations[0]
    last_act = activations[-1]

    # Calculate total duration from first ON to last OFF
    first_date = first_act.get('date', '')
    first_time = first_act.get('on_time', '00:00')
    last_date = last_act.get('date', '')
    last_off_time = last_act.get('off_time', last_act.get('on_time', '00:00'))

    try:
        start_dt = pd.to_datetime(f"{first_date} {first_time}")
        end_dt = pd.to_datetime(f"{last_date} {last_off_time}")
        total_duration = (end_dt - start_dt).total_seconds() / 60
    except:
        # Fallback: sum individual durations
        total_duration = sum(a.get('duration_minutes', 0) for a in activations)

    # Get magnitude (use max or average)
    magnitudes = []
    for act in activations:
        if 'magnitude' in act:
            magnitudes.append(act['magnitude'])
        elif 'total_magnitude' in act:
            magnitudes.append(act['total_magnitude'])

    avg_magnitude = int(sum(magnitudes) / len(magnitudes)) if magnitudes else 0
    max_magnitude = max(magnitudes) if magnitudes else 0

    # Preserve phase info
    phase = first_act.get('phase', '')
    phases = first_act.get('phases', [phase] if phase else [])
    phase_magnitudes = first_act.get('phase_magnitudes', {})

    return {
        'date': first_date,
        'on_time': first_time,
        'off_time': last_off_time,
        'duration_minutes': int(round(total_duration)),
        'magnitude': avg_magnitude,
        'max_magnitude': max_magnitude,
        'total_magnitude': sum(magnitudes) if magnitudes else 0,
        'cycle_count': len(activations),  # Number of compressor cycles in this session
        'phase': phase,
        'phases': phases,
        'phase_magnitudes': phase_magnitudes,
        '_raw_activations': activations,  # Keep raw data for validation
    }


def _is_valid_ac_session(session: Dict, raw_activations: List[Dict],
                          min_cycles: int = 2,
                          min_session_duration: int = 30,
                          max_magnitude_std_pct: float = 0.20) -> bool:
    """
    Validate if a session looks like real AC usage (not just any high-power device).

    Args:
        session: Session summary dict with cycle_count, duration_minutes, etc.
        raw_activations: List of individual activations in this session
        min_cycles: Minimum compressor cycles required (default 2)
        min_session_duration: Minimum total session duration in minutes (default 30)
        max_magnitude_std_pct: Maximum magnitude std as percentage of mean (default 20%)

    Returns:
        True if session looks like AC, False otherwise
    """
    # Check minimum cycles
    cycle_count = session.get('cycle_count', 1)
    if cycle_count < min_cycles:
        return False

    # Check minimum session duration
    duration = session.get('duration_minutes', 0)
    if duration < min_session_duration:
        return False

    # Check magnitude consistency (std < 20% of mean)
    if raw_activations and len(raw_activations) >= 2:
        magnitudes = [a.get('magnitude', 0) for a in raw_activations]
        if magnitudes:
            mean_mag = np.mean(magnitudes)
            std_mag = np.std(magnitudes)
            if mean_mag > 0 and (std_mag / mean_mag) > max_magnitude_std_pct:
                return False

    return True


def detect_ac_patterns(experiment_dir: Path, house_id: str,
                       run_number: int = 0,
                       time_tolerance_minutes: int = 10,
                       damaged_phases: List[str] = None) -> Dict[str, Any]:
    """
    Detect AC patterns - both central AC (multi-phase) and regular AC (single phase).

    Central AC: Events that occur on all 3 phases (or 2 if one is damaged) at similar times.
    Regular AC: High-power events only on w1 phase.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number to analyze
        time_tolerance_minutes: Max time difference between phases to be considered synchronized
        damaged_phases: List of damaged phase names (e.g., ['w3'])

    Returns:
        Dictionary with central_ac and regular_ac activation lists
    """
    result = {
        'central_ac': {
            'activations': [],
            'total_count': 0,
            'phases_used': [],
        },
        'regular_ac': {
            'activations': [],
            'total_count': 0,
        },
        'has_central_ac': False,
        'has_regular_ac': False,
    }

    house_dir = _get_house_dir(experiment_dir, house_id, run_number)
    if not house_dir.exists():
        return result

    # Load matches
    matches_df = _load_monthly_files(house_dir, "matches", f"matches_{house_id}_*.pkl")
    if matches_df is None or len(matches_df) == 0:
        return result

    # Parse timestamps
    for col in ['on_start', 'on_end', 'off_start', 'off_end']:
        if col in matches_df.columns:
            matches_df[col] = pd.to_datetime(matches_df[col], format='%d/%m/%Y %H:%M', errors='coerce')

    # Determine which phases to check
    all_phases = ['w1', 'w2', 'w3']
    damaged_phases = damaged_phases or []
    active_phases = [p for p in all_phases if p not in damaged_phases]

    # Need at least 2 phases for central AC
    if len(active_phases) < 2:
        return result

    tolerance = pd.Timedelta(minutes=time_tolerance_minutes)

    # Group matches by phase
    phase_matches = {}
    for phase in active_phases:
        phase_df = matches_df[matches_df['phase'] == phase].copy()
        if not phase_df.empty:
            phase_matches[phase] = phase_df

    # Find central AC activations (synchronized across phases)
    # Optimized: use merge_asof instead of iterrows
    central_activations = []
    used_match_ids = set()

    # Use w1 as reference if available, otherwise use first available phase
    reference_phase = 'w1' if 'w1' in phase_matches else active_phases[0]
    if reference_phase not in phase_matches:
        return result

    ref_matches = phase_matches[reference_phase].copy()
    ref_matches = ref_matches.sort_values('on_start').reset_index(drop=True)

    # Use merge_asof to find synchronized events on other phases
    merged = ref_matches[['on_start', 'on_event_id', 'on_magnitude', 'off_end', 'off_start', 'duration']].copy()
    merged = merged.rename(columns={
        'on_event_id': f'id_{reference_phase}',
        'on_magnitude': f'mag_{reference_phase}',
        'off_end': f'off_end_{reference_phase}',
        'off_start': f'off_start_{reference_phase}',
        'duration': f'dur_{reference_phase}'
    })

    other_phases_in_merge = []
    for other_phase in active_phases:
        if other_phase == reference_phase or other_phase not in phase_matches:
            continue

        other_df = phase_matches[other_phase][['on_start', 'on_event_id', 'on_magnitude', 'off_end', 'off_start', 'duration']].copy()
        other_df = other_df.sort_values('on_start').reset_index(drop=True)
        other_df = other_df.rename(columns={
            'on_start': f'on_start_{other_phase}',
            'on_event_id': f'id_{other_phase}',
            'on_magnitude': f'mag_{other_phase}',
            'off_end': f'off_end_{other_phase}',
            'off_start': f'off_start_{other_phase}',
            'duration': f'dur_{other_phase}'
        })

        # Merge with tolerance - ensure both DataFrames have reset index
        merged = merged.sort_values('on_start').reset_index(drop=True)
        other_df = other_df.sort_values(f'on_start_{other_phase}').reset_index(drop=True)

        merged = pd.merge_asof(
            merged,
            other_df,
            left_on='on_start',
            right_on=f'on_start_{other_phase}',
            tolerance=tolerance,
            direction='nearest'
        )
        other_phases_in_merge.append(other_phase)

    # Filter rows where ALL phases have matches (central AC)
    if other_phases_in_merge:
        all_matched_mask = merged[f'id_{reference_phase}'].notna()
        for other_phase in other_phases_in_merge:
            all_matched_mask &= merged[f'id_{other_phase}'].notna()

        central_rows = merged[all_matched_mask]

        for _, row in central_rows.iterrows():
            ref_id = row[f'id_{reference_phase}']
            if ref_id in used_match_ids:
                continue

            # Check if any other phase ID is already used
            skip = False
            for other_phase in other_phases_in_merge:
                if row[f'id_{other_phase}'] in used_match_ids:
                    skip = True
                    break
            if skip:
                continue

            # Mark all IDs as used
            used_match_ids.add(ref_id)
            for other_phase in other_phases_in_merge:
                used_match_ids.add(row[f'id_{other_phase}'])

            # Calculate totals
            total_magnitude = abs(row.get(f'mag_{reference_phase}', 0) or 0)
            durations = [row.get(f'dur_{reference_phase}', 0) or 0]
            off_times = []

            ref_off = row.get(f'off_end_{reference_phase}') or row.get(f'off_start_{reference_phase}')
            if pd.notna(ref_off):
                off_times.append(ref_off)

            phase_magnitudes = {reference_phase: int(abs(row.get(f'mag_{reference_phase}', 0) or 0))}

            for other_phase in other_phases_in_merge:
                mag = row.get(f'mag_{other_phase}', 0) or 0
                total_magnitude += abs(mag)
                phase_magnitudes[other_phase] = int(abs(mag))

                dur = row.get(f'dur_{other_phase}', 0)
                if dur:
                    durations.append(dur)

                off = row.get(f'off_end_{other_phase}') or row.get(f'off_start_{other_phase}')
                if pd.notna(off):
                    off_times.append(off)

            avg_duration = np.mean(durations) if durations else 0
            off_end = max(off_times) if off_times else ref_off
            ref_on_start = row['on_start']

            activation = {
                'date': ref_on_start.strftime('%Y-%m-%d'),
                'on_time': ref_on_start.strftime('%H:%M'),
                'off_time': off_end.strftime('%H:%M') if pd.notna(off_end) else '',
                'duration_minutes': int(round(avg_duration)),
                'total_magnitude': int(total_magnitude),
                'phases': [reference_phase] + other_phases_in_merge,
                'phase_magnitudes': phase_magnitudes,
            }
            central_activations.append(activation)

    # Sort by date and time
    central_activations.sort(key=lambda x: (x['date'], x['on_time']))

    # Group into sessions (compressor cycles within 1 hour = same session)
    central_sessions = _group_activations_into_sessions(central_activations, gap_threshold_minutes=60)

    result['central_ac']['activations'] = central_sessions  # Sessions, not individual cycles
    result['central_ac']['raw_activations'] = central_activations  # Keep raw data for reference
    result['central_ac']['total_count'] = len(central_sessions)  # Session count
    result['central_ac']['total_cycles'] = len(central_activations)  # Individual cycle count
    result['central_ac']['phases_used'] = active_phases
    result['has_central_ac'] = len(central_sessions) >= 3  # Need at least 3 sessions

    # Find regular AC on ALL phases (not just w1), not part of central AC
    # AC detection criteria:
    # - Individual cycle: 800W+, duration 3-30 minutes
    # - Session: 2+ cycles, 30+ min total, consistent magnitude (std < 20%)
    regular_ac_by_phase = {}

    # AC cycle duration bounds (minutes)
    MIN_CYCLE_DURATION = 3
    MAX_CYCLE_DURATION = 30

    for phase in active_phases:
        if phase not in phase_matches:
            continue

        phase_df = phase_matches[phase].copy()

        # Vectorized filtering instead of iterrows
        filtered = phase_df[
            (~phase_df['on_event_id'].isin(used_match_ids)) &
            (phase_df['on_magnitude'].abs() >= 800) &
            (phase_df['duration'] >= MIN_CYCLE_DURATION) &
            (phase_df['duration'] <= MAX_CYCLE_DURATION)
        ].copy()

        if filtered.empty:
            continue

        # Vectorized: prepare off_time column
        filtered['off_time_val'] = filtered['off_end'].fillna(filtered['off_start'])

        # Build activations list (still need a loop for dict creation, but much smaller dataset now)
        phase_activations = []
        for _, row in filtered.iterrows():
            on_start = row['on_start']
            off_end = row['off_time_val']
            activation = {
                'date': on_start.strftime('%Y-%m-%d'),
                'on_time': on_start.strftime('%H:%M'),
                'off_time': off_end.strftime('%H:%M') if pd.notna(off_end) else '',
                'duration_minutes': int(round(row['duration'])),
                'magnitude': int(abs(row['on_magnitude'])),
                'phase': phase,
            }
            phase_activations.append(activation)

        if phase_activations:
            # Sort by date and time
            phase_activations.sort(key=lambda x: (x['date'], x['on_time']))

            # Group into sessions (compressor cycles within 1 hour = same session)
            phase_sessions = _group_activations_into_sessions(phase_activations, gap_threshold_minutes=60)

            # Filter sessions using AC validation criteria
            valid_sessions = []
            valid_raw_activations = []
            for session in phase_sessions:
                raw_acts = session.get('_raw_activations', [])
                if _is_valid_ac_session(session, raw_acts):
                    # Remove internal field before storing
                    session_copy = {k: v for k, v in session.items() if not k.startswith('_')}
                    valid_sessions.append(session_copy)
                    valid_raw_activations.extend(raw_acts)

            if valid_sessions:
                regular_ac_by_phase[phase] = {
                    'activations': valid_sessions,  # Sessions, not individual cycles
                    'raw_activations': valid_raw_activations,  # Keep raw data for reference
                    'total_count': len(valid_sessions),  # Session count
                    'total_cycles': len(valid_raw_activations),  # Individual cycle count
                }

    # Keep backwards compatibility with old format (combine all into regular_ac)
    all_regular_activations = []
    all_raw_activations = []
    for phase_data in regular_ac_by_phase.values():
        all_regular_activations.extend(phase_data['activations'])
        all_raw_activations.extend(phase_data.get('raw_activations', []))
    all_regular_activations.sort(key=lambda x: (x['date'], x['on_time']))
    all_raw_activations.sort(key=lambda x: (x['date'], x['on_time']))

    result['regular_ac']['activations'] = all_regular_activations  # Sessions
    result['regular_ac']['raw_activations'] = all_raw_activations  # Individual cycles
    result['regular_ac']['total_count'] = len(all_regular_activations)  # Session count
    result['regular_ac']['total_cycles'] = len(all_raw_activations)  # Individual cycle count
    result['has_regular_ac'] = len(all_regular_activations) >= 3  # Need at least 3 sessions

    # New format: grouped by phase
    result['regular_ac_by_phase'] = regular_ac_by_phase

    return result


def analyze_device_usage_patterns(ac_detection: Dict[str, Any],
                                   boiler_detection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze device usage by season (winter/summer) and time of day (day/night).

    Args:
        ac_detection: Result from detect_ac_patterns
        boiler_detection: Result from detect_boiler_patterns

    Returns:
        Dictionary with seasonal and time-of-day breakdown for each device type
    """
    result = {
        'central_ac': {'seasonal': {}, 'time_of_day': {}},
        'regular_ac': {'seasonal': {}, 'time_of_day': {}},
        'boiler': {'seasonal': {}, 'time_of_day': {}},
    }

    # Define seasons (Northern Hemisphere, Israel)
    # Winter: Dec, Jan, Feb (months 12, 1, 2)
    # Summer: Jun, Jul, Aug (months 6, 7, 8)
    # Spring/Fall: Mar, Apr, May, Sep, Oct, Nov
    def get_season(date_str):
        """Get season from date string (YYYY-MM-DD)."""
        try:
            month = int(date_str.split('-')[1])
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [6, 7, 8]:
                return 'summer'
            elif month in [3, 4, 5]:
                return 'spring'
            else:
                return 'fall'
        except:
            return 'unknown'

    def get_time_of_day(time_str):
        """Get time of day from time string (HH:MM)."""
        try:
            hour = int(time_str.split(':')[0])
            if 6 <= hour < 18:
                return 'day'
            else:
                return 'night'
        except:
            return 'unknown'

    def analyze_activations(activations):
        """Analyze a list of activations by season and time of day."""
        seasonal = {'winter': 0, 'spring': 0, 'summer': 0, 'fall': 0}
        time_of_day = {'day': 0, 'night': 0}

        for act in activations:
            date = act.get('date', '')
            time = act.get('on_time', '')

            season = get_season(date)
            tod = get_time_of_day(time)

            if season in seasonal:
                seasonal[season] += 1
            if tod in time_of_day:
                time_of_day[tod] += 1

        return {'seasonal': seasonal, 'time_of_day': time_of_day}

    # Analyze Central AC
    if ac_detection:
        central_activations = ac_detection.get('central_ac', {}).get('activations', [])
        if central_activations:
            analysis = analyze_activations(central_activations)
            result['central_ac'] = analysis

        # Analyze Regular AC (all phases combined)
        regular_activations = ac_detection.get('regular_ac', {}).get('activations', [])
        if regular_activations:
            analysis = analyze_activations(regular_activations)
            result['regular_ac'] = analysis

    # Analyze Boiler
    if boiler_detection:
        boiler_activations = boiler_detection.get('boiler', {}).get('activations', [])
        if boiler_activations:
            analysis = analyze_activations(boiler_activations)
            result['boiler'] = analysis

    return result


def detect_boiler_patterns(experiment_dir: Path, house_id: str,
                           run_number: int = 0,
                           min_duration_minutes: int = 25,
                           min_magnitude: int = 1500,
                           isolation_window_minutes: int = 30) -> Dict[str, Any]:
    """
    Detect water heater (boiler) patterns.

    Boiler characteristics:
    - Long duration (typically 25-60+ minutes)
    - High power (typically 1500-3000W)
    - Isolated: no medium-duration events before or after within a time window

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number to analyze
        min_duration_minutes: Minimum match duration to consider (default 25 min)
        min_magnitude: Minimum magnitude in watts (default 1500W)
        isolation_window_minutes: Window to check for nearby events (default 30 min)

    Returns:
        Dictionary with boiler detection results
    """
    result = {
        'boiler': {
            'activations': [],
            'total_count': 0,
            'avg_duration': 0,
            'avg_magnitude': 0,
        },
        'has_boiler': False,
    }

    house_dir = _get_house_dir(experiment_dir, house_id, run_number)
    if not house_dir.exists():
        return result

    # Load matches
    matches_df = _load_monthly_files(house_dir, "matches", f"matches_{house_id}_*.pkl")
    if matches_df is None or len(matches_df) == 0:
        return result

    # Parse timestamps
    for col in ['on_start', 'on_end', 'off_start', 'off_end']:
        if col in matches_df.columns:
            matches_df[col] = pd.to_datetime(matches_df[col], format='%d/%m/%Y %H:%M', errors='coerce')

    # Step 1: Find all long-duration, high-power matches (potential boilers)
    if 'duration' not in matches_df.columns or 'on_magnitude' not in matches_df.columns:
        return result

    long_matches = matches_df[
        (matches_df['duration'] >= min_duration_minutes) &
        (matches_df['on_magnitude'].abs() >= min_magnitude)
    ].copy()

    if long_matches.empty:
        return result

    # Step 2: Find medium-duration matches (these should NOT be near boiler events)
    # Medium = 3-24 minutes (as defined in segmentation)
    medium_matches = matches_df[
        (matches_df['duration'] > 2) &
        (matches_df['duration'] <= 24)
    ].copy()

    isolation_window = pd.Timedelta(minutes=isolation_window_minutes)

    # Optimized: use merge_asof to find nearby medium events
    # Process each phase separately
    boiler_activations = []

    for phase in long_matches['phase'].unique():
        phase_long = long_matches[long_matches['phase'] == phase].copy()
        phase_long = phase_long.sort_values('on_start').reset_index(drop=True)

        if medium_matches.empty:
            # No medium events - all long events are isolated
            isolated = phase_long
        else:
            phase_medium = medium_matches[medium_matches['phase'] == phase].copy()

            if phase_medium.empty:
                # No medium events on this phase - all isolated
                isolated = phase_long
            else:
                phase_medium = phase_medium.sort_values('on_start').reset_index(drop=True)

                # Use merge_asof to find nearest medium event
                # Ensure both DataFrames are sorted and have reset index (merge_asof requires this)
                left_df = phase_long[['on_start', 'off_end', 'off_start', 'on_magnitude', 'duration', 'phase']].copy()
                left_df = left_df.sort_values('on_start').reset_index(drop=True)
                right_df = phase_medium[['on_start']].rename(columns={'on_start': 'nearby_medium_start'}).copy()
                right_df = right_df.sort_values('nearby_medium_start').reset_index(drop=True)

                merged = pd.merge_asof(
                    left_df,
                    right_df,
                    left_on='on_start',
                    right_on='nearby_medium_start',
                    tolerance=isolation_window,
                    direction='nearest'
                )

                # Isolated = no nearby medium event found
                isolated = merged[merged['nearby_medium_start'].isna()].copy()

        # Build activation dicts
        for _, row in isolated.iterrows():
            on_start = row['on_start']
            if pd.isna(on_start):
                continue
            off_end = row.get('off_end') if pd.notna(row.get('off_end')) else row.get('off_start')

            activation = {
                'date': on_start.strftime('%Y-%m-%d'),
                'on_time': on_start.strftime('%H:%M'),
                'off_time': off_end.strftime('%H:%M') if pd.notna(off_end) else '',
                'duration_minutes': int(round(row['duration'])),
                'magnitude': int(abs(row['on_magnitude'])),
                'phase': phase,
            }
            boiler_activations.append(activation)

    # Sort by date and time
    boiler_activations.sort(key=lambda x: (x['date'], x['on_time']))

    # ===== NEW: Detect multi-phase simultaneous events (might be EV charging, not boiler) =====
    # Boiler should only use ONE phase. If multiple phases are active at the same time,
    # it's likely something else (EV charging, heavy industrial equipment, etc.)

    sync_tolerance = pd.Timedelta(minutes=5)  # Events within 5 min are "simultaneous"

    # Convert to DataFrame for easier processing
    if boiler_activations:
        boiler_df = pd.DataFrame(boiler_activations)
        boiler_df['on_datetime'] = pd.to_datetime(boiler_df['date'] + ' ' + boiler_df['on_time'])

        # Find events that have other-phase events at the same time
        multi_phase_events = []
        single_phase_events = []

        for i, row in boiler_df.iterrows():
            on_time = row['on_datetime']
            phase = row['phase']

            # Find other-phase events at similar time
            other_phase_events = boiler_df[
                (boiler_df['phase'] != phase) &
                (abs(boiler_df['on_datetime'] - on_time) <= sync_tolerance)
            ]

            if len(other_phase_events) > 0:
                # This event has simultaneous activity on other phases
                event = row.to_dict()
                event['other_phases_active'] = other_phase_events['phase'].unique().tolist()
                event['num_phases_active'] = len(other_phase_events['phase'].unique()) + 1
                event.pop('on_datetime', None)  # Remove helper column
                multi_phase_events.append(event)
            else:
                # Single-phase event - likely real boiler
                event = row.to_dict()
                event['other_phases_active'] = []
                event['num_phases_active'] = 1
                event.pop('on_datetime', None)
                single_phase_events.append(event)

        # Count events per phase for single-phase events
        phase_counts = {}
        for evt in single_phase_events:
            p = evt['phase']
            phase_counts[p] = phase_counts.get(p, 0) + 1

        # Find dominant phase (most boiler events)
        dominant_phase = max(phase_counts, key=phase_counts.get) if phase_counts else None
        result['boiler']['dominant_phase'] = dominant_phase
        result['boiler']['phase_distribution'] = phase_counts

        # Final boiler activations = single-phase events only
        result['boiler']['activations'] = single_phase_events
        result['boiler']['total_count'] = len(single_phase_events)

        if single_phase_events:
            result['boiler']['avg_duration'] = sum(a['duration_minutes'] for a in single_phase_events) / len(single_phase_events)
            result['boiler']['avg_magnitude'] = sum(a['magnitude'] for a in single_phase_events) / len(single_phase_events)

        # Multi-phase events table (might be EV charging or other heavy load)
        result['suspicious_multi_phase'] = {
            'activations': multi_phase_events,
            'total_count': len(multi_phase_events),
            'description': 'Events with boiler-like characteristics but multiple phases active simultaneously - may be EV charging or other high-power device',
        }

        if multi_phase_events:
            # Check if these are truly synchronized (3 phases = likely EV or central device)
            three_phase_count = sum(1 for e in multi_phase_events if e['num_phases_active'] == 3)
            two_phase_count = sum(1 for e in multi_phase_events if e['num_phases_active'] == 2)

            result['suspicious_multi_phase']['three_phase_count'] = three_phase_count
            result['suspicious_multi_phase']['two_phase_count'] = two_phase_count

            if three_phase_count > len(multi_phase_events) * 0.5:
                result['suspicious_multi_phase']['likely_device'] = 'EV_charging_or_central_device'
            else:
                result['suspicious_multi_phase']['likely_device'] = 'unknown'

    else:
        result['boiler']['activations'] = []
        result['boiler']['total_count'] = 0
        result['boiler']['dominant_phase'] = None
        result['boiler']['phase_distribution'] = {}
        result['suspicious_multi_phase'] = {'activations': [], 'total_count': 0}

    result['has_boiler'] = result['boiler']['total_count'] >= 3  # Need at least 3 single-phase activations

    return result
