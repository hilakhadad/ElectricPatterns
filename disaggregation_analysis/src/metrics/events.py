"""
Event metrics for experiment results.

Analyzes detected events characteristics.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List


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


def calculate_event_metrics(experiment_dir: Path, house_id: str,
                            run_number: int = 0,
                            preloaded: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calculate event-level metrics for a house run.

    Supports both old (single file) and new (monthly subfolder) structures.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number to analyze
        preloaded: Optional dict with pre-loaded DataFrames ('on_off') to avoid redundant file I/O

    Returns:
        Dictionary with event metrics
    """
    metrics = {
        'house_id': house_id,
        'run_number': run_number,
    }

    if preloaded:
        on_off_df = preloaded.get('on_off')
    else:
        house_dir = _get_house_dir(experiment_dir, house_id, run_number)
        on_off_df = _load_monthly_files(house_dir, "on_off", "on_off_*.pkl")

    if on_off_df is None:
        metrics['error'] = 'No on_off file found'
        return metrics
    on_off_df['start'] = pd.to_datetime(on_off_df['start'], format='mixed', dayfirst=True)
    on_off_df['end'] = pd.to_datetime(on_off_df['end'], format='mixed', dayfirst=True)

    # Event type breakdown
    on_events = on_off_df[on_off_df['event'] == 'on']
    off_events = on_off_df[on_off_df['event'] == 'off']

    # Phase distribution
    if 'phase' in on_off_df.columns:
        phase_dist = on_off_df['phase'].value_counts().to_dict()
        metrics['events_by_phase'] = phase_dist

        for phase in ['w1', 'w2', 'w3']:
            metrics[f'events_{phase}'] = phase_dist.get(phase, 0)

    # Magnitude distribution
    if 'magnitude' in on_off_df.columns:
        magnitudes = on_off_df['magnitude'].abs()

        metrics['avg_magnitude'] = magnitudes.mean()
        metrics['median_magnitude'] = magnitudes.median()
        metrics['std_magnitude'] = magnitudes.std()
        metrics['min_magnitude'] = magnitudes.min()
        metrics['max_magnitude'] = magnitudes.max()

        # Magnitude ranges
        metrics['events_under_500W'] = (magnitudes < 500).sum()
        metrics['events_500_1000W'] = ((magnitudes >= 500) & (magnitudes < 1000)).sum()
        metrics['events_1000_2000W'] = ((magnitudes >= 1000) & (magnitudes < 2000)).sum()
        metrics['events_2000_5000W'] = ((magnitudes >= 2000) & (magnitudes < 5000)).sum()
        metrics['events_over_5000W'] = (magnitudes >= 5000).sum()

        # Magnitude percentiles
        metrics['magnitude_p25'] = magnitudes.quantile(0.25)
        metrics['magnitude_p75'] = magnitudes.quantile(0.75)
        metrics['magnitude_p90'] = magnitudes.quantile(0.90)
        metrics['magnitude_p99'] = magnitudes.quantile(0.99)

    # Duration distribution (for ON events)
    if 'duration' in on_events.columns:
        durations = on_events['duration']

        metrics['avg_event_duration'] = durations.mean()
        metrics['median_event_duration'] = durations.median()
        metrics['max_event_duration'] = durations.max()

        # Duration ranges (in minutes)
        metrics['events_under_5min'] = (durations < 5).sum()
        metrics['events_5_30min'] = ((durations >= 5) & (durations < 30)).sum()
        metrics['events_30_60min'] = ((durations >= 30) & (durations < 60)).sum()
        metrics['events_1_6hr'] = ((durations >= 60) & (durations < 360)).sum()
        metrics['events_over_6hr'] = (durations >= 360).sum()

    # Time of day distribution
    if 'start' in on_off_df.columns:
        try:
            hours = pd.to_datetime(on_off_df['start']).dt.hour
            metrics['events_night'] = ((hours >= 22) | (hours < 6)).sum()  # 22:00-06:00
            metrics['events_morning'] = ((hours >= 6) & (hours < 12)).sum()  # 06:00-12:00
            metrics['events_afternoon'] = ((hours >= 12) & (hours < 18)).sum()  # 12:00-18:00
            metrics['events_evening'] = ((hours >= 18) & (hours < 22)).sum()  # 18:00-22:00

            # Peak event hour
            hour_counts = hours.value_counts()
            if not hour_counts.empty:
                metrics['peak_event_hour'] = hour_counts.idxmax()
                metrics['events_at_peak_hour'] = hour_counts.max()
        except:
            pass

    # Matched vs unmatched breakdown
    if 'matched' in on_off_df.columns:
        matched = on_off_df[on_off_df['matched'] == 1]
        unmatched = on_off_df[on_off_df['matched'] == 0]

        if 'magnitude' in on_off_df.columns:
            metrics['matched_avg_magnitude'] = matched['magnitude'].abs().mean() if len(matched) > 0 else 0
            metrics['unmatched_avg_magnitude'] = unmatched['magnitude'].abs().mean() if len(unmatched) > 0 else 0

            # Are unmatched events typically smaller or larger?
            if len(matched) > 0 and len(unmatched) > 0:
                metrics['unmatched_vs_matched_magnitude'] = (
                    unmatched['magnitude'].abs().mean() / matched['magnitude'].abs().mean()
                )

    return metrics


def analyze_unmatched_events(experiment_dir: Path, house_id: str,
                             run_number: int = 0) -> Dict[str, Any]:
    """
    Detailed analysis of unmatched events.

    Supports both old (single file) and new (monthly subfolder) structures.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number to analyze

    Returns:
        Dictionary with unmatched event analysis
    """
    metrics = {
        'house_id': house_id,
        'run_number': run_number,
    }

    house_dir = _get_house_dir(experiment_dir, house_id, run_number)

    # Load unmatched files
    un_on = _load_monthly_files(house_dir, "unmatched_on", f"unmatched_on_{house_id}_*.pkl")
    un_off = _load_monthly_files(house_dir, "unmatched_off", f"unmatched_off_{house_id}_*.pkl")

    if un_on is not None:
        metrics['unmatched_on_count'] = len(un_on)

        if 'magnitude' in un_on.columns and len(un_on) > 0:
            mags = un_on['magnitude'].abs()
            metrics['unmatched_on_avg_magnitude'] = mags.mean()
            metrics['unmatched_on_max_magnitude'] = mags.max()
            metrics['unmatched_on_total_power'] = mags.sum()

            # Large unmatched ON events (potential issues)
            metrics['unmatched_on_over_2000W'] = (mags > 2000).sum()

        if 'phase' in un_on.columns:
            metrics['unmatched_on_by_phase'] = un_on['phase'].value_counts().to_dict()

    if un_off is not None:
        metrics['unmatched_off_count'] = len(un_off)

        if 'magnitude' in un_off.columns and len(un_off) > 0:
            mags = un_off['magnitude'].abs()
            metrics['unmatched_off_avg_magnitude'] = mags.mean()
            metrics['unmatched_off_max_magnitude'] = mags.max()
            metrics['unmatched_off_total_power'] = mags.sum()

        if 'phase' in un_off.columns:
            metrics['unmatched_off_by_phase'] = un_off['phase'].value_counts().to_dict()

    # Imbalance analysis
    un_on_count = metrics.get('unmatched_on_count', 0)
    un_off_count = metrics.get('unmatched_off_count', 0)

    if un_on_count > 0 or un_off_count > 0:
        metrics['unmatched_imbalance'] = un_on_count - un_off_count
        metrics['unmatched_imbalance_ratio'] = (
            un_on_count / un_off_count if un_off_count > 0 else float('inf')
        )

    return metrics
