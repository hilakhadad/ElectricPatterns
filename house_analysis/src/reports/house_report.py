"""
Per-house analysis and report generation.

Analyzes a single house and generates a detailed report.
"""
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from metrics import (
    calculate_coverage_metrics,
    calculate_power_statistics,
    calculate_temporal_patterns,
    calculate_data_quality_metrics
)
from metrics.temporal import calculate_flat_segments


def analyze_single_house(data: pd.DataFrame, house_id: str,
                         phase_cols: list = None) -> Dict[str, Any]:
    """
    Run complete analysis on a single house.

    Args:
        data: DataFrame with timestamp and phase power columns
        house_id: House identifier
        phase_cols: List of phase column names

    Returns:
        Dictionary with all analysis results
    """
    results = {
        'house_id': house_id,
        'analysis_timestamp': datetime.now().isoformat(),
    }

    # Coverage metrics
    coverage = calculate_coverage_metrics(data, phase_cols)
    results['coverage'] = coverage

    # Power statistics
    power_stats = calculate_power_statistics(data, phase_cols)
    results['power_statistics'] = power_stats

    # Temporal patterns
    temporal = calculate_temporal_patterns(data, phase_cols)
    results['temporal_patterns'] = temporal

    # Data quality
    quality = calculate_data_quality_metrics(data, phase_cols)
    results['data_quality'] = quality

    # Flat segments
    flat = calculate_flat_segments(data, phase_cols)
    results['flat_segments'] = flat

    # Summary flags for easy filtering
    results['flags'] = _generate_flags(results)

    return results


def _generate_flags(analysis: Dict[str, Any]) -> Dict[str, bool]:
    """
    Generate boolean flags for easy filtering of houses.

    Args:
        analysis: Complete analysis results

    Returns:
        Dictionary of boolean flags
    """
    flags = {}

    coverage = analysis.get('coverage', {})
    quality = analysis.get('data_quality', {})
    power = analysis.get('power_statistics', {})
    temporal = analysis.get('temporal_patterns', {})

    flat = analysis.get('flat_segments', {})

    # Coverage flags
    flags['low_coverage'] = coverage.get('coverage_ratio', 1) < 0.8
    flags['short_duration'] = coverage.get('days_span', 365) < 30
    flags['has_large_gaps'] = coverage.get('max_gap_minutes', 0) > 60
    flags['many_gaps'] = coverage.get('pct_gaps_over_2min', 0) > 5  # More than 5% gaps over 2min

    # Quality flags
    flags['has_negative_values'] = any(
        quality.get(f'{c}_negative_count', 0) > 0
        for c in ['w1', 'w2', 'w3', '1', '2', '3']
    )
    flags['many_outliers'] = any(
        quality.get(f'{c}_outliers_3sd_pct', 0) > 1
        for c in ['w1', 'w2', 'w3', '1', '2', '3']
    )
    flags['many_large_jumps'] = any(
        quality.get(f'{c}_jumps_over_2000W', 0) > 500
        for c in ['w1', 'w2', 'w3', '1', '2', '3']
    )
    flags['low_quality_score'] = quality.get('quality_score', 100) < 70

    # Power flags
    flags['unbalanced_phases'] = power.get('phase_balance_ratio', 1) > 3
    flags['single_active_phase'] = power.get('active_phases', 3) == 1
    flags['very_high_power'] = power.get('total_max', 0) > 20000  # Over 20kW max

    # Flat segments flags (potential metering issues)
    flags['many_flat_segments'] = any(
        flat.get(f'phase_{c}_flat_pct', 0) > 30  # More than 30% flat
        for c in ['w1', 'w2', 'w3', '1', '2', '3']
        if f'phase_{c}_flat_pct' in flat
    )

    # Temporal flags
    flags['unusual_night_ratio'] = any(
        temporal.get(f'phase_{c}_night_day_ratio', 1) > 3  # Night 3x higher than day
        for c in ['w1', 'w2', 'w3', '1', '2', '3']
        if f'phase_{c}_night_day_ratio' in temporal
    )

    return flags


def generate_house_report(analysis: Dict[str, Any], output_dir: str,
                          format: str = 'json') -> str:
    """
    Generate and save a report for a single house.

    Args:
        analysis: Analysis results from analyze_single_house
        output_dir: Directory to save the report
        format: Output format ('json' or 'csv')

    Returns:
        Path to the saved report
    """
    os.makedirs(output_dir, exist_ok=True)

    house_id = analysis.get('house_id', 'unknown')

    if format == 'json':
        output_path = os.path.join(output_dir, f'analysis_{house_id}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False, default=str)

    elif format == 'csv':
        # Flatten the nested dict for CSV
        flat_data = _flatten_dict(analysis)
        df = pd.DataFrame([flat_data])
        output_path = os.path.join(output_dir, f'analysis_{house_id}.csv')
        df.to_csv(output_path, index=False)

    else:
        raise ValueError(f"Unknown format: {format}")

    return output_path


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_house_data(file_path: str) -> pd.DataFrame:
    """
    Load house data from CSV file.

    Args:
        file_path: Path to the CSV file

    Returns:
        DataFrame with parsed timestamps
    """
    data = pd.read_csv(file_path)

    # Parse timestamp if present
    if 'timestamp' in data.columns:
        data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Rename columns if needed
    if '1' in data.columns and 'w1' not in data.columns:
        data = data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'})

    return data
