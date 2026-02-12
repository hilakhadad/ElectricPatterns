"""
Iteration metrics for experiment results.

Analyzes progress across multiple iterations.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


def _find_run_dir(experiment_dir: Path, run_number: int) -> Optional[Path]:
    """
    Find the run directory, supporting both standard and dynamic threshold naming.

    Supports:
    - Standard: run_0/, run_1/, ...
    - Dynamic threshold: run_0_th2000/, run_1_th1500/, ...

    Returns:
        Path to the run directory, or None if not found.
    """
    # Try standard naming first
    standard = experiment_dir / f"run_{run_number}"
    if standard.exists():
        return standard

    # Try dynamic threshold naming (run_N_thXXXX)
    for d in sorted(experiment_dir.glob(f"run_{run_number}_th*")):
        if d.is_dir():
            return d

    return None


def _get_house_dir(experiment_dir: Path, house_id: str, run_number: int) -> Path:
    """Get the house directory, supporting both old, new, and dynamic threshold structures."""
    # Try to find the run directory (supports run_N and run_N_thXXXX)
    run_dir = _find_run_dir(experiment_dir, run_number)
    if run_dir is not None:
        house_dir = run_dir / f"house_{house_id}"
        if house_dir.exists():
            return house_dir

    # Fall back to old structure: experiment_dir/house_X/run_N/house_X/
    return experiment_dir / f"house_{house_id}" / f"run_{run_number}" / f"house_{house_id}"


def _load_monthly_files(house_dir: Path, subfolder: str, pattern: str) -> Optional[pd.DataFrame]:
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


def calculate_iteration_metrics(experiment_dir: Path, house_id: str,
                                 max_iterations: int = 10,
                                 verbose: bool = False) -> Dict[str, Any]:
    """
    Calculate metrics across all iterations for a house.

    Supports both old (single file) and new (monthly subfolder) structures.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        max_iterations: Maximum number of iterations to check
        verbose: Print debug info

    Returns:
        Dictionary with iteration metrics
    """
    metrics = {
        'house_id': house_id,
    }

    iterations_data = []

    for run_num in range(max_iterations):
        house_dir = _get_house_dir(experiment_dir, house_id, run_num)

        if verbose:
            print(f"  [DEBUG] Checking: {house_dir}")
            print(f"  [DEBUG] Exists: {house_dir.exists()}")

        if not house_dir.exists():
            if verbose and run_num == 0:
                # Try to show what's actually in the directory
                parent = experiment_dir / f"run_{run_num}"
                if parent.exists():
                    print(f"  [DEBUG] Contents of {parent}:")
                    for item in list(parent.iterdir())[:5]:
                        print(f"    - {item.name}")
            break

        iter_data = {'run_number': run_num}

        # Check for key files - support both monthly subfolder and direct files
        on_off_df = _load_monthly_files(house_dir, "on_off", "on_off_*.pkl")
        matches_df = _load_monthly_files(house_dir, "matches", f"matches_{house_id}_*.pkl")
        summarized_df = _load_monthly_files(house_dir, "summarized", f"summarized_{house_id}_*.pkl")

        if on_off_df is not None:
            iter_data['total_events'] = len(on_off_df)

            # Separate ON and OFF event counts
            on_events = on_off_df[on_off_df['event'] == 'on'] if 'event' in on_off_df.columns else pd.DataFrame()
            off_events = on_off_df[on_off_df['event'] == 'off'] if 'event' in on_off_df.columns else pd.DataFrame()
            iter_data['on_events'] = len(on_events)
            iter_data['off_events'] = len(off_events)

            if 'matched' in on_off_df.columns:
                iter_data['matched_events'] = on_off_df['matched'].sum()
                iter_data['unmatched_events'] = len(on_off_df) - on_off_df['matched'].sum()
                iter_data['matching_rate'] = on_off_df['matched'].mean()

                # Separate unmatched counts for ON and OFF
                if len(on_events) > 0 and 'matched' in on_events.columns:
                    iter_data['unmatched_on'] = len(on_events[on_events['matched'] == 0])
                if len(off_events) > 0 and 'matched' in off_events.columns:
                    iter_data['unmatched_off'] = len(off_events[off_events['matched'] == 0])

        if matches_df is not None:
            iter_data['total_matches'] = len(matches_df)

            if 'on_magnitude' in matches_df.columns:
                iter_data['matched_power'] = matches_df['on_magnitude'].abs().sum()

            # Add matched minutes (sum of all match durations)
            # Fix negative durations (events crossing midnight) by adding 1440 min
            if 'duration' in matches_df.columns:
                durations = matches_df['duration'].copy()
                durations = durations.apply(lambda x: x + 1440 if x < 0 else x)
                iter_data['matched_minutes'] = float(durations.sum())
                iter_data['negative_duration_count'] = int((matches_df['duration'] < 0).sum())
            elif 'on_start' in matches_df.columns and 'off_end' in matches_df.columns:
                # Fallback: calculate duration from timestamps (for older files without duration column)
                on_starts = pd.to_datetime(matches_df['on_start'], dayfirst=True)
                off_ends = pd.to_datetime(matches_df['off_end'], dayfirst=True)
                durations = (off_ends - on_starts).dt.total_seconds() / 60
                # Fix negative durations (events crossing midnight)
                durations = durations.apply(lambda x: x + 1440 if x < 0 else x)
                iter_data['matched_minutes'] = float(durations.sum())
                iter_data['negative_duration_count'] = int((durations < 0).sum())

        # Try summarized (new) or segmented (old)
        seg_df = summarized_df
        if seg_df is None:
            segmented_files = list(house_dir.glob("segmented_*.pkl"))
            if segmented_files:
                seg_df = pd.read_pickle(segmented_files[0])

        if seg_df is not None:
            # Find event power columns
            event_cols = [c for c in seg_df.columns if c.startswith('event_power_')]
            if event_cols:
                iter_data['segmented_power'] = sum(seg_df[c].sum() for c in event_cols)

            # Find remaining power columns
            remaining_cols = [c for c in seg_df.columns if c.startswith('remaining_power_') or c.startswith('remaining_')]
            if remaining_cols:
                iter_data['remaining_power'] = sum(seg_df[c].sum() for c in remaining_cols)

                # Check for negative values
                neg_count = sum((seg_df[c] < 0).sum() for c in remaining_cols)
                iter_data['negative_values'] = neg_count

        iterations_data.append(iter_data)

    metrics['iterations_completed'] = len(iterations_data)
    metrics['iterations_data'] = iterations_data

    if iterations_data:
        # First iteration stats
        metrics['first_iter_events'] = iterations_data[0].get('total_events', 0)
        metrics['first_iter_matching_rate'] = iterations_data[0].get('matching_rate', 0)

        # Last iteration stats
        metrics['last_iter_events'] = iterations_data[-1].get('total_events', 0)
        metrics['last_iter_matching_rate'] = iterations_data[-1].get('matching_rate', 0)

        # Progress metrics
        if len(iterations_data) > 1:
            # Events reduction (should decrease)
            first_events = iterations_data[0].get('total_events', 0)
            last_events = iterations_data[-1].get('total_events', 0)
            if first_events > 0:
                metrics['events_reduction_ratio'] = 1 - (last_events / first_events)

            # Cumulative matched power
            total_matched_power = sum(
                d.get('matched_power', 0) for d in iterations_data
            )
            metrics['total_matched_power'] = total_matched_power

            # Cumulative matched minutes
            total_matched_minutes = sum(
                d.get('matched_minutes', 0) for d in iterations_data
            )
            metrics['total_matched_minutes'] = total_matched_minutes

            # Matching rate improvement
            first_rate = iterations_data[0].get('matching_rate', 0)
            last_rate = iterations_data[-1].get('matching_rate', 0)
            metrics['matching_rate_change'] = last_rate - first_rate

        # Check for issues
        total_negatives = sum(d.get('negative_values', 0) for d in iterations_data)
        metrics['total_negative_values'] = total_negatives
        metrics['has_negative_values'] = total_negatives > 0

    return metrics


def analyze_iteration_progression(iterations_data: List[Dict]) -> Dict[str, Any]:
    """
    Analyze how metrics progress across iterations.

    Args:
        iterations_data: List of per-iteration metrics

    Returns:
        Dictionary with progression analysis
    """
    if not iterations_data or len(iterations_data) < 2:
        return {}

    analysis = {}

    # Extract time series
    events = [d.get('total_events', 0) for d in iterations_data]
    matches = [d.get('total_matches', 0) for d in iterations_data]
    rates = [d.get('matching_rate', 0) for d in iterations_data]

    # Diminishing returns analysis
    if len(matches) >= 2:
        match_diffs = [matches[i] - matches[i-1] if i > 0 else matches[0]
                       for i in range(len(matches))]
        analysis['matches_per_iteration'] = match_diffs

        # When did we stop getting significant matches?
        for i, diff in enumerate(match_diffs):
            if diff < 5 and i > 0:  # Less than 5 new matches
                analysis['diminishing_returns_at'] = i
                break

    # Convergence check
    if len(events) >= 2:
        # Did we run out of events to match?
        analysis['converged'] = events[-1] == 0 or (
            len(events) >= 2 and events[-1] == events[-2]
        )

    return analysis


def get_iteration_summary(experiment_dir: Path, house_id: str) -> str:
    """
    Get a text summary of iteration progress.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier

    Returns:
        Human-readable summary string
    """
    metrics = calculate_iteration_metrics(experiment_dir, house_id)

    lines = []
    lines.append(f"House {house_id}: {metrics.get('iterations_completed', 0)} iterations")

    if 'first_iter_events' in metrics:
        lines.append(f"  First iteration: {metrics['first_iter_events']} events, "
                    f"{metrics.get('first_iter_matching_rate', 0):.1%} matched")

    if 'last_iter_events' in metrics:
        lines.append(f"  Last iteration: {metrics['last_iter_events']} events, "
                    f"{metrics.get('last_iter_matching_rate', 0):.1%} matched")

    if 'events_reduction_ratio' in metrics:
        lines.append(f"  Events reduction: {metrics['events_reduction_ratio']:.1%}")

    if 'total_matched_power' in metrics:
        lines.append(f"  Total matched power: {metrics['total_matched_power']/1000:.1f} kW")

    if metrics.get('has_negative_values'):
        lines.append(f"  WARNING: {metrics['total_negative_values']} negative values detected")

    return '\n'.join(lines)
