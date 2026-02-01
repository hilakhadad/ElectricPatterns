"""
Segmentation metrics for experiment results.

Analyzes power segmentation quality and coverage.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


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
    metrics = {
        'house_id': house_id,
        'run_number': run_number,
    }

    house_dir = _get_house_dir(experiment_dir, house_id, run_number)

    # Load segmented data - use summarized file instead (much smaller)
    # Check new structure: summarized/ subfolder with monthly files
    summarized_dir = house_dir / "summarized"
    if summarized_dir.exists():
        summarized_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.csv"))
    else:
        summarized_files = list(house_dir.glob("summarized_*.csv"))

    segmented_files = list(house_dir.glob("segmented_*.csv"))

    # Prefer summarized file (much smaller ~80MB vs 1.2GB)
    if summarized_files:
        data_files = summarized_files
    elif segmented_files:
        data_files = segmented_files
    else:
        metrics['error'] = 'No segmented/summarized file found'
        return metrics

    # Read only first few rows to get column names (from first file)
    sample_df = pd.read_csv(data_files[0], nrows=5)
    columns = list(sample_df.columns)

    # Calculate sums using chunked reading to avoid memory issues
    chunk_sums = {col: 0 for col in columns if col != 'timestamp'}
    negative_counts = {col: 0 for col in columns if 'remaining' in col.lower()}
    total_rows = 0

    # Process all files (handles monthly structure)
    for data_file in data_files:
        for chunk in pd.read_csv(data_file, chunksize=100000):
            total_rows += len(chunk)
            for col in chunk_sums.keys():
                if col in chunk.columns:
                    chunk_sums[col] += chunk[col].sum()
                    if col in negative_counts:
                        negative_counts[col] += (chunk[col] < 0).sum()

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
    if phases is None:
        phases = ['w1', 'w2', 'w3'] if 'w1' in seg_df.columns else ['1', '2', '3']

    metrics = {}

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
