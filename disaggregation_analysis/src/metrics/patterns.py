"""
Event pattern analysis for experiment results.

Analyzes recurring patterns, proximity between events, and daily statistics.

This module is a facade -- actual implementations are split across:
  - pattern_detection.py  -- recurring pattern detection, clustering, time distribution
  - pattern_ac.py         -- AC-specific detection and validation
  - pattern_boiler.py     -- boiler-specific detection and validation
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


# Import from sub-modules (after _get_house_dir and _load_monthly_files are defined,
# since sub-modules import them from this module)
from metrics.pattern_detection import (
    _calculate_daily_stats,
    _find_recurring_events,
    _find_recurring_matches,
    _cluster_times_with_dates,
    _cluster_times_with_duration,
    _cluster_times,
    _minutes_to_time_str,
    _calculate_proximity_stats,
    _find_event_clusters,
    _calculate_time_distribution,
    find_periodic_patterns,
)
from metrics.pattern_ac import (
    _group_activations_into_sessions,
    _summarize_session,
    _is_valid_ac_session,
    detect_ac_patterns,
    analyze_device_usage_patterns,
)
from metrics.pattern_boiler import (
    _filter_ac_from_boiler_candidates,
    detect_boiler_patterns,
)


def calculate_pattern_metrics(experiment_dir: Path, house_id: str,
                              run_number: int = 0,
                              preloaded: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Calculate event pattern metrics for a house.

    Supports both old (single file) and new (monthly subfolder) structures.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Iteration number (default 0)
        preloaded: Optional dict with pre-loaded DataFrames ('on_off', 'matches')

    Returns:
        Dictionary with pattern metrics
    """
    metrics = {
        'house_id': house_id,
        'run_number': run_number,
    }

    if preloaded:
        on_off_df = preloaded.get('on_off')
        matches_df = preloaded.get('matches')
    else:
        house_dir = _get_house_dir(experiment_dir, house_id, run_number)
        if not house_dir.exists():
            return metrics
        on_off_df = _load_monthly_files(house_dir, "on_off", "on_off_*.pkl")
        matches_df = _load_monthly_files(house_dir, "matches", f"matches_{house_id}_*.pkl")

    if on_off_df is None:
        return metrics

    # Parse timestamps (skip if already datetime)
    for col in ['start', 'end']:
        if col in on_off_df.columns and not pd.api.types.is_datetime64_any_dtype(on_off_df[col]):
            on_off_df[col] = pd.to_datetime(on_off_df[col], format='mixed', dayfirst=True, errors='coerce')

    if matches_df is not None:
        for col in ['on_start', 'on_end', 'off_start', 'off_end']:
            if col in matches_df.columns and not pd.api.types.is_datetime64_any_dtype(matches_df[col]):
                matches_df[col] = pd.to_datetime(matches_df[col], format='mixed', dayfirst=True, errors='coerce')

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

    lines_out = []
    lines_out.append(f"Recurring Patterns for House {house_id}")
    lines_out.append("=" * 60)

    recurring = metrics.get('recurring_events', {})
    patterns = recurring.get('recurring_patterns', [])

    if not patterns:
        lines_out.append("No recurring patterns found.")
        return '\n'.join(lines_out)

    significant_patterns = [p for p in patterns if p.get('occurrences', 0) >= min_occurrences]

    if not significant_patterns:
        lines_out.append(f"No patterns with {min_occurrences}+ occurrences found.")
        return '\n'.join(lines_out)

    lines_out.append(f"Found {len(significant_patterns)} recurring patterns (min {min_occurrences} occurrences):")
    lines_out.append(f"\n{'Phase':<6} {'Type':<5} {'Power':<8} {'Time':<12} {'Range':<17} {'Days':<6}")
    lines_out.append("-" * 60)

    for p in sorted(significant_patterns, key=lambda x: x.get('occurrences', 0), reverse=True)[:20]:
        phase = p.get('phase', '?')
        event_type = p.get('event_type', '?')[:4]
        magnitude = f"{p.get('magnitude', 0):.0f}W"
        avg_time = p.get('avg_time', '?')
        time_range = p.get('time_range', '?')
        unique_days = p.get('unique_days', 0)
        lines_out.append(f"{phase:<6} {event_type:<5} {magnitude:<8} {avg_time:<12} {time_range:<17} {unique_days:<6}")

    total_recurring = recurring.get('total_recurring_events', 0)
    pct_recurring = recurring.get('recurring_event_percentage', 0)
    lines_out.append("-" * 60)
    lines_out.append(f"Total events in patterns: {total_recurring} ({pct_recurring:.1f}% of all events)")

    phase_patterns = {}
    for p in significant_patterns:
        phase = p.get('phase', '?')
        if phase not in phase_patterns:
            phase_patterns[phase] = 0
        phase_patterns[phase] += 1
    lines_out.append(f"\nPatterns by phase: " + ", ".join(f"{k}: {v}" for k, v in sorted(phase_patterns.items())))

    return '\n'.join(lines_out)
