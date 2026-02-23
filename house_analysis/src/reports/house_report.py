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
    calculate_data_quality_metrics,
    calculate_wave_behavior_metrics
)
from metrics.temporal import calculate_flat_segments, calculate_temporal_patterns_by_period


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

    # Temporal patterns by year/month (for yearly breakdown reports)
    temporal_by_period = calculate_temporal_patterns_by_period(data, phase_cols)
    results['temporal_by_period'] = temporal_by_period

    # Data quality (pass coverage metrics for scoring)
    coverage_ratio = coverage.get('coverage_ratio', 1.0)
    days_span = coverage.get('days_span', 0)
    max_gap_minutes = coverage.get('max_gap_minutes', 0)
    pct_gaps_over_2min = coverage.get('pct_gaps_over_2min', 0)
    avg_nan_pct = coverage.get('avg_nan_pct', 0)
    anomaly_count = coverage.get('anomaly_count', 0)
    quality = calculate_data_quality_metrics(
        data, phase_cols,
        coverage_ratio=coverage_ratio,
        days_span=days_span,
        max_gap_minutes=max_gap_minutes,
        pct_gaps_over_2min=pct_gaps_over_2min,
        avg_nan_pct=avg_nan_pct,
        anomaly_count=anomaly_count,
    )
    results['data_quality'] = quality

    # Flat segments
    flat = calculate_flat_segments(data, phase_cols)
    results['flat_segments'] = flat

    # Wave behavior pre-analysis
    wave_behavior = calculate_wave_behavior_metrics(data, phase_cols)
    results['wave_behavior'] = wave_behavior

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
    flags['many_nan_values'] = coverage.get('avg_nan_pct', 0) > 2  # More than 2% NaN across phases
    flags['has_duplicate_timestamps'] = coverage.get('has_duplicate_timestamps', False)

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
    # Raised threshold from 30% to 70% - with 10W tolerance, 30% is too sensitive
    flags['many_flat_segments'] = any(
        flat.get(f'phase_{c}_flat_pct', 0) > 70  # More than 70% flat (likely meter issue)
        for c in ['w1', 'w2', 'w3', '1', '2', '3']
        if f'phase_{c}_flat_pct' in flat
    )

    # Wave behavior flags
    wave = analysis.get('wave_behavior', {})
    flags['has_wave_behavior'] = wave.get('has_wave_behavior', False)
    flags['wave_dominant'] = wave.get('wave_classification') == 'wave_dominant'

    # Temporal flags
    flags['unusual_night_ratio'] = any(
        temporal.get(f'phase_{c}_night_day_ratio', 1) > 3  # Night 3x higher than day
        for c in ['w1', 'w2', 'w3', '1', '2', '3']
        if f'phase_{c}_night_day_ratio' in temporal
    )

    # Faulty/defective meter flags
    flags['has_dead_phase'] = quality.get('has_dead_phase', False)
    flags['has_faulty_nan_phase'] = quality.get('has_faulty_nan_phase', False)
    flags['quality_label'] = quality.get('quality_label')  # faulty_dead_phase / faulty_high_nan / faulty_both / None

    # Quality scoring component flags (from quality.py quality_flags)
    quality_flags = quality.get('quality_flags', [])
    flags['low_sharp_entry'] = 'low_sharp_entry' in quality_flags
    flags['low_device_signature'] = 'low_device_signature' in quality_flags
    flags['low_power_profile'] = 'low_power_profile' in quality_flags
    flags['low_variability'] = 'low_variability' in quality_flags
    flags['low_data_volume'] = 'low_data_volume' in quality_flags
    flags['low_data_integrity'] = 'low_data_integrity' in quality_flags

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
    Load house data from pkl file.

    Args:
        file_path: Path to the pkl file

    Returns:
        DataFrame with parsed timestamps
    """
    data = pd.read_pickle(file_path)

    # Ensure timestamp is datetime if present
    if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Rename columns if needed
    if '1' in data.columns and 'w1' not in data.columns:
        data = data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'})

    return data
