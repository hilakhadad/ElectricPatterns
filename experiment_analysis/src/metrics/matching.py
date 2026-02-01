"""
Matching metrics for experiment results.

Analyzes ON/OFF event matching performance.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional


def _get_house_dir(experiment_dir: Path, house_id: str, run_number: int) -> Path:
    """Get the house directory, supporting both old and new structures."""
    # Try new structure first: experiment_dir/run_N/house_X/
    new_dir = experiment_dir / f"run_{run_number}" / f"house_{house_id}"
    if new_dir.exists():
        return new_dir
    # Fall back to old structure: experiment_dir/house_X/run_N/house_X/
    return experiment_dir / f"house_{house_id}" / f"run_{run_number}" / f"house_{house_id}"


def _load_monthly_files(house_dir: Path, subfolder: str, pattern: str) -> Optional[pd.DataFrame]:
    """Load and concatenate monthly files from a subfolder or fallback to direct files."""
    subdir = house_dir / subfolder
    if subdir.exists():
        files = sorted(subdir.glob(pattern))
        if files:
            return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Fallback: try files directly in house_dir
    files = list(house_dir.glob(pattern))
    if files:
        return pd.read_csv(files[0])
    return None


def calculate_matching_metrics(experiment_dir: Path, house_id: str,
                                run_number: int = 0) -> Dict[str, Any]:
    """
    Calculate matching metrics for a house run.

    Supports both old (single file) and new (monthly subfolder) structures.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number to analyze

    Returns:
        Dictionary with matching metrics
    """
    metrics = {
        'house_id': house_id,
        'run_number': run_number,
    }

    house_dir = _get_house_dir(experiment_dir, house_id, run_number)

    # Load on_off log
    on_off_df = _load_monthly_files(house_dir, "on_off", "on_off_*.csv")
    if on_off_df is None:
        metrics['error'] = 'No on_off file found'
        return metrics

    # Total events
    on_events = on_off_df[on_off_df['event'] == 'on']
    off_events = on_off_df[on_off_df['event'] == 'off']

    metrics['total_on_events'] = len(on_events)
    metrics['total_off_events'] = len(off_events)
    metrics['total_events'] = len(on_off_df)

    # Matched events
    if 'matched' in on_off_df.columns:
        matched_on = on_events[on_events['matched'] == 1]
        matched_off = off_events[off_events['matched'] == 1]
        unmatched_on = on_events[on_events['matched'] == 0]
        unmatched_off = off_events[off_events['matched'] == 0]

        metrics['matched_on_events'] = len(matched_on)
        metrics['matched_off_events'] = len(matched_off)
        metrics['unmatched_on_events'] = len(unmatched_on)
        metrics['unmatched_off_events'] = len(unmatched_off)

        # Matching rate
        if len(on_events) > 0:
            metrics['on_matching_rate'] = len(matched_on) / len(on_events)
        else:
            metrics['on_matching_rate'] = 0

        if len(off_events) > 0:
            metrics['off_matching_rate'] = len(matched_off) / len(off_events)
        else:
            metrics['off_matching_rate'] = 0

        metrics['overall_matching_rate'] = (
            (len(matched_on) + len(matched_off)) / len(on_off_df)
            if len(on_off_df) > 0 else 0
        )

    # Load matches file for tag analysis
    matches_df = _load_monthly_files(house_dir, "matches", f"matches_{house_id}_*.csv")
    if matches_df is not None:
        metrics['total_matches'] = len(matches_df)

        # Tag breakdown
        if 'tag' in matches_df.columns:
            tag_counts = matches_df['tag'].value_counts().to_dict()
            metrics['tag_breakdown'] = tag_counts  # Used by HTML report
            metrics['matches_by_tag'] = tag_counts  # Legacy key

            for tag in ['NON-M', 'NOISY', 'PARTIAL']:
                metrics[f'matches_{tag.lower().replace("-", "_")}'] = tag_counts.get(tag, 0)

        # Phase breakdown
        if 'phase' in matches_df.columns:
            phase_counts = matches_df['phase'].value_counts().to_dict()
            metrics['phase_breakdown'] = phase_counts  # Used by HTML report
            metrics['matches_by_phase'] = phase_counts  # Legacy key

        # Duration statistics
        if 'duration' in matches_df.columns:
            durations = matches_df['duration']
            metrics['avg_match_duration'] = durations.mean()
            metrics['median_match_duration'] = durations.median()
            metrics['max_match_duration'] = durations.max()
            metrics['min_match_duration'] = durations.min()

            # Duration breakdown matching segmentation categories:
            # short: <= 2 min, medium: 3-24 min, long: >= 25 min
            metrics['duration_breakdown'] = {
                'short': int((durations <= 2).sum()),
                'medium': int(((durations > 2) & (durations <= 24)).sum()),
                'long': int((durations > 24).sum()),
            }

        # Magnitude statistics
        if 'on_magnitude' in matches_df.columns:
            magnitudes = matches_df['on_magnitude'].abs()
            metrics['avg_match_magnitude'] = magnitudes.mean()
            metrics['total_matched_power'] = magnitudes.sum()

            # Magnitude breakdown for histogram
            # Bins aligned with segmentation threshold (~1300W minimum)
            metrics['magnitude_breakdown'] = {
                '1300-1800': int(((magnitudes >= 1300) & (magnitudes < 1800)).sum()),
                '1800-2500': int(((magnitudes >= 1800) & (magnitudes < 2500)).sum()),
                '2500-3500': int(((magnitudes >= 2500) & (magnitudes < 3500)).sum()),
                '3500-5000': int(((magnitudes >= 3500) & (magnitudes < 5000)).sum()),
                '5000+': int((magnitudes >= 5000).sum()),
            }

    # Unmatched analysis
    unmatched_on_df = _load_monthly_files(house_dir, "unmatched_on", f"unmatched_on_{house_id}_*.csv")
    unmatched_off_df = _load_monthly_files(house_dir, "unmatched_off", f"unmatched_off_{house_id}_*.csv")

    if unmatched_on_df is not None:
        metrics['unmatched_on_total_power'] = unmatched_on_df['magnitude'].abs().sum() if 'magnitude' in unmatched_on_df.columns else 0
        metrics['unmatched_on_avg_magnitude'] = unmatched_on_df['magnitude'].abs().mean() if 'magnitude' in unmatched_on_df.columns else 0

    if unmatched_off_df is not None:
        metrics['unmatched_off_total_power'] = unmatched_off_df['magnitude'].abs().sum() if 'magnitude' in unmatched_off_df.columns else 0

    # Remainder events (from partial matching)
    remainder_files = list(house_dir.glob("remainder_*.csv"))
    if remainder_files:
        remainder_df = pd.read_csv(remainder_files[0])
        metrics['remainder_events'] = len(remainder_df)
        metrics['remainder_total_power'] = remainder_df['magnitude'].abs().sum() if 'magnitude' in remainder_df.columns else 0

    return metrics


def calculate_matching_quality(matches_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate quality metrics for matches.

    Args:
        matches_df: DataFrame with match data

    Returns:
        Dictionary with quality metrics
    """
    metrics = {}

    if matches_df.empty:
        return metrics

    # Magnitude difference between ON and OFF
    if 'on_magnitude' in matches_df.columns and 'off_magnitude' in matches_df.columns:
        mag_diff = (matches_df['on_magnitude'].abs() - matches_df['off_magnitude'].abs()).abs()
        metrics['avg_magnitude_diff'] = mag_diff.mean()
        metrics['max_magnitude_diff'] = mag_diff.max()
        metrics['pct_large_mag_diff'] = (mag_diff > 350).sum() / len(matches_df) * 100

    # Duration distribution
    if 'duration' in matches_df.columns:
        durations = matches_df['duration']
        metrics['pct_short_duration'] = (durations < 5).sum() / len(matches_df) * 100  # < 5 min
        metrics['pct_medium_duration'] = ((durations >= 5) & (durations < 60)).sum() / len(matches_df) * 100
        metrics['pct_long_duration'] = ((durations >= 60) & (durations < 360)).sum() / len(matches_df) * 100
        metrics['pct_very_long_duration'] = (durations >= 360).sum() / len(matches_df) * 100  # > 6 hours

    return metrics
