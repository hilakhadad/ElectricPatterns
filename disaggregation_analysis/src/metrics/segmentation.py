"""
Segmentation metrics for experiment results.

Analyzes power segmentation quality and coverage.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def _get_house_dir(experiment_dir: Path, house_id: str, run_number: int) -> Path:
    """Get the house directory, supporting both old and new structures."""
    # Try new structure first: experiment_dir/run_N/house_X/
    new_dir = experiment_dir / f"run_{run_number}" / f"house_{house_id}"
    if new_dir.exists():
        return new_dir
    # Fall back to old structure: experiment_dir/house_X/run_N/house_X/
    return experiment_dir / f"house_{house_id}" / f"run_{run_number}" / f"house_{house_id}"


def calculate_segmentation_metrics(experiment_dir: Path, house_id: str,
                                    run_number: int = 0) -> Dict[str, Any]:
    """
    Calculate segmentation metrics for a house run.

    Supports both old (single file) and new (monthly subfolder) structures.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number to analyze

    Returns:
        Dictionary with segmentation metrics
    """
    logger.debug("calculate_segmentation_metrics: house_id=%s, run_number=%d", house_id, run_number)
    metrics = {
        'house_id': house_id,
        'run_number': run_number,
    }

    house_dir = _get_house_dir(experiment_dir, house_id, run_number)

    # Load segmented data - use summarized file instead (much smaller)
    # Check new structure: summarized/ subfolder with monthly files
    summarized_dir = house_dir / "summarized"
    if summarized_dir.exists():
        summarized_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.pkl"))
    else:
        summarized_files = list(house_dir.glob("summarized_*.pkl"))

    segmented_files = list(house_dir.glob("segmented_*.pkl"))

    # Prefer summarized file (much smaller ~80MB vs 1.2GB)
    if summarized_files:
        data_files = summarized_files
    elif segmented_files:
        data_files = segmented_files
    else:
        logger.warning("calculate_segmentation_metrics: No segmented/summarized data for house %s run %d",
                       house_id, run_number)
        metrics['error'] = 'No segmented/summarized file found'
        return metrics

    # Read first file to get column names
    sample_df = pd.read_pickle(data_files[0])
    columns = list(sample_df.columns)

    # Calculate sums
    chunk_sums = {col: 0 for col in columns if col != 'timestamp'}
    negative_counts = {col: 0 for col in columns if 'remaining' in col.lower()}
    total_rows = 0

    # Process all files (handles monthly structure)
    for data_file in data_files:
        df = pd.read_pickle(data_file)
        total_rows += len(df)
        for col in chunk_sums.keys():
            if col in df.columns:
                chunk_sums[col] += df[col].sum()
                if col in negative_counts:
                    negative_counts[col] += (df[col] < 0).sum()

    # Create a mock dataframe-like structure for compatibility
    class MockDF:
        def __init__(self, sums, cols, rows):
            self.sums = sums
            self.columns = cols
            self._len = rows
        def __len__(self):
            return self._len

    seg_df = MockDF(chunk_sums, columns, total_rows)

    # Find phase columns and event power columns
    phases = ['w1', 'w2', 'w3']
    if 'w1' not in seg_df.columns:
        # Check for original_ prefix
        if 'original_w1' in seg_df.columns:
            phases = ['w1', 'w2', 'w3']
        else:
            phases = ['1', '2', '3']

    event_power_cols = [c for c in seg_df.columns if c.startswith('event_power_')]
    remaining_cols = [c for c in seg_df.columns if 'remaining' in c.lower()]
    original_cols = [c for c in seg_df.columns if c.startswith('original_')]

    # Total power metrics (use original_ columns if available, else phase columns)
    total_original = 0
    for phase in phases:
        orig_col = f'original_{phase}'
        if orig_col in chunk_sums:
            total_power = chunk_sums[orig_col]
            metrics[f'{phase}_total_power'] = total_power
            total_original += total_power
        elif phase in chunk_sums:
            total_power = chunk_sums[phase]
            metrics[f'{phase}_total_power'] = total_power
            total_original += total_power

    # Segmented power metrics - look for short/medium/long duration columns
    total_segmented = 0
    duration_cols = [c for c in seg_df.columns if 'duration' in c.lower()]

    for col in duration_cols:
        if col in chunk_sums:
            col_sum = chunk_sums[col]
            total_segmented += col_sum
            metrics[f'{col}_sum'] = col_sum

    # Remaining power
    total_remaining = 0
    for col in remaining_cols:
        if col in chunk_sums:
            col_sum = chunk_sums[col]
            total_remaining += col_sum
            metrics[f'{col}_sum'] = col_sum

    metrics['total_segmented_power'] = total_segmented
    metrics['total_remaining_power'] = total_remaining
    metrics['total_power'] = total_original

    # Segmentation ratio
    if total_original > 0:
        metrics['segmentation_ratio'] = total_segmented / total_original
        metrics['remaining_ratio'] = total_remaining / total_original
    else:
        metrics['segmentation_ratio'] = 0
        metrics['remaining_ratio'] = 1

    # Negative values (already counted during chunked reading)
    metrics['negative_value_columns'] = {k: v for k, v in negative_counts.items() if v > 0}
    metrics['negative_value_count'] = sum(negative_counts.values())
    metrics['has_negative_values'] = metrics['negative_value_count'] > 0

    # Per-phase segmentation
    for phase in phases:
        phase_duration_cols = [c for c in duration_cols if phase in c]
        phase_remaining_col = f'remaining_{phase}'

        orig_col = f'original_{phase}'
        if orig_col in chunk_sums:
            phase_total = chunk_sums[orig_col]
        elif phase in chunk_sums:
            phase_total = chunk_sums[phase]
        else:
            phase_total = 0

        phase_segmented = sum(chunk_sums.get(c, 0) for c in phase_duration_cols)

        metrics[f'{phase}_segmented_power'] = phase_segmented
        if phase_total > 0:
            metrics[f'{phase}_segmentation_ratio'] = phase_segmented / phase_total
        else:
            metrics[f'{phase}_segmentation_ratio'] = 0

        if phase_remaining_col in negative_counts:
            metrics[f'{phase}_negative_remaining'] = negative_counts[phase_remaining_col]

    return metrics


def calculate_segmentation_quality(seg_df: pd.DataFrame,
                                   phases: List[str] = None) -> Dict[str, Any]:
    """
    Calculate detailed quality metrics for segmentation.

    Args:
        seg_df: DataFrame with segmented data
        phases: List of phase column names

    Returns:
        Dictionary with quality metrics
    """
    logger.debug("calculate_segmentation_quality: %d rows in seg_df", len(seg_df))
    if phases is None:
        phases = ['w1', 'w2', 'w3'] if 'w1' in seg_df.columns else ['1', '2', '3']

    metrics = {}

    if seg_df.empty:
        logger.warning("calculate_segmentation_quality: empty segmentation DataFrame")
        return metrics

    remaining_cols = [c for c in seg_df.columns if c.startswith('remaining_power_')]

    for col in remaining_cols:
        phase = col.replace('remaining_power_', '')
        remaining = seg_df[col]

        # Negative value analysis
        negative_mask = remaining < 0
        if negative_mask.any():
            metrics[f'{phase}_negative_count'] = negative_mask.sum()
            metrics[f'{phase}_negative_min'] = remaining[negative_mask].min()
            metrics[f'{phase}_negative_sum'] = remaining[negative_mask].sum()

        # Remaining power distribution
        metrics[f'{phase}_remaining_mean'] = remaining.mean()
        metrics[f'{phase}_remaining_std'] = remaining.std()
        metrics[f'{phase}_remaining_max'] = remaining.max()

        # Percentage of time at zero remaining
        metrics[f'{phase}_zero_remaining_pct'] = (remaining == 0).sum() / len(remaining) * 100

    return metrics


def calculate_threshold_explanation_metrics(experiment_dir: Path, house_id: str,
                                            run_number: int = 0,
                                            threshold: float = 1300) -> Dict[str, Any]:
    """
    Calculate how many high-power minutes are segregated.

    For each phase, counts:
    1. Minutes where original power > threshold (high power)
    2. Minutes where original > threshold BUT remaining < threshold (segregated)

    A minute is "segregated" when the segregation successfully attributed enough
    power to events that the remaining power dropped below the threshold.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number to analyze (0 = compare original vs run_0 output)
        threshold: Power threshold in watts (default 1300W, matching pipeline config)

    Returns:
        Dictionary with threshold explanation metrics per phase and total
    """
    logger.debug("calculate_threshold_explanation_metrics: house_id=%s, run_number=%d, threshold=%.0f",
                 house_id, run_number, threshold)
    metrics = {
        'threshold': threshold,
        'house_id': house_id,
        'run_number': run_number,
    }

    house_dir = _get_house_dir(experiment_dir, house_id, run_number)

    # Load summarized data
    summarized_dir = house_dir / "summarized"
    if summarized_dir.exists():
        summarized_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.pkl"))
    else:
        summarized_files = list(house_dir.glob("summarized_*.pkl"))

    if not summarized_files:
        logger.warning("calculate_threshold_explanation_metrics: No summarized data for house %s run %d",
                       house_id, run_number)
        metrics['error'] = 'No summarized file found'
        return metrics

    phases = ['w1', 'w2', 'w3']

    # Initialize counters per phase
    phase_stats = {p: {'above_th': 0, 'explained': 0} for p in phases}
    total_minutes = 0

    # Process all files
    for data_file in summarized_files:
        df = pd.read_pickle(data_file)
        total_minutes += len(df)

        for phase in phases:
            # Original power column
            orig_col = f'original_{phase}'
            if orig_col not in df.columns:
                orig_col = phase
            if orig_col not in df.columns:
                continue

            # Remaining power column for this phase
            remaining_col = f'remaining_{phase}'
            if remaining_col not in df.columns:
                continue

            # Minutes above threshold in original
            above_th_mask = df[orig_col] > threshold
            above_th_count = above_th_mask.sum()
            phase_stats[phase]['above_th'] += above_th_count

            # Minutes segregated: original > TH but remaining < TH
            # This means the segregation successfully attributed the high power
            if above_th_count > 0:
                explained_mask = above_th_mask & (df[remaining_col] < threshold)
                explained_count = explained_mask.sum()
                phase_stats[phase]['explained'] += explained_count

    # Calculate per-phase metrics
    total_above_th = 0
    total_explained = 0

    for phase in phases:
        above_th = phase_stats[phase]['above_th']
        explained = phase_stats[phase]['explained']

        metrics[f'{phase}_minutes_above_th'] = above_th
        metrics[f'{phase}_minutes_explained'] = explained
        if above_th > 0:
            metrics[f'{phase}_explanation_rate'] = explained / above_th
        else:
            metrics[f'{phase}_explanation_rate'] = 0

        total_above_th += above_th
        total_explained += explained

    # Total metrics
    metrics['total_minutes'] = total_minutes
    metrics['total_minutes_above_th'] = total_above_th
    metrics['total_minutes_explained'] = total_explained
    if total_above_th > 0:
        metrics['total_explanation_rate'] = total_explained / total_above_th
    else:
        metrics['total_explanation_rate'] = 0

    return metrics


def calculate_threshold_explanation_all_iterations(experiment_dir: Path, house_id: str,
                                                    max_iterations: int = 10,
                                                    threshold: float = 1300) -> List[Dict[str, Any]]:
    """
    Calculate threshold explanation metrics for all iterations.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        max_iterations: Maximum number of iterations to check
        threshold: Power threshold in watts (default 1300W, matching pipeline config)

    Returns:
        List of dictionaries with metrics per iteration
    """
    logger.debug("calculate_threshold_explanation_all_iterations: house_id=%s, max_iterations=%d, threshold=%.0f",
                 house_id, max_iterations, threshold)
    results = []

    for run_number in range(max_iterations):
        house_dir = _get_house_dir(experiment_dir, house_id, run_number)

        # Check if this iteration exists
        summarized_dir = house_dir / "summarized"
        if summarized_dir.exists():
            summarized_files = list(summarized_dir.glob(f"summarized_{house_id}_*.pkl"))
        else:
            summarized_files = list(house_dir.glob("summarized_*.pkl"))

        if not summarized_files:
            break  # No more iterations

        metrics = calculate_threshold_explanation_metrics(
            experiment_dir, house_id, run_number, threshold
        )
        metrics['iteration'] = run_number
        results.append(metrics)

    return results
