"""
Aggregate report generation for all houses.

Combines per-house analyses into summary statistics and comparative reports.
"""
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


def aggregate_all_houses(house_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate analysis results from multiple houses.

    Args:
        house_analyses: List of analysis results from analyze_single_house

    Returns:
        Dictionary with aggregated statistics
    """
    if not house_analyses:
        return {}

    report = {
        'generated_at': datetime.now().isoformat(),
        'total_houses': len(house_analyses),
    }

    # Count flags
    flag_counts = {}
    for analysis in house_analyses:
        flags = analysis.get('flags', {})
        for flag_name, flag_value in flags.items():
            if flag_name not in flag_counts:
                flag_counts[flag_name] = 0
            if flag_value:
                flag_counts[flag_name] += 1

    report['flag_counts'] = flag_counts
    report['flag_percentages'] = {
        k: v / len(house_analyses) * 100 for k, v in flag_counts.items()
    }

    # Aggregate coverage statistics
    coverage_stats = _aggregate_section(house_analyses, 'coverage', [
        'total_rows', 'days_span', 'coverage_ratio', 'months_count',
        'median_step_seconds', 'max_gap_minutes',
        'pct_gaps_over_2min', 'pct_gaps_over_10min', 'pct_gaps_over_60min'
    ])
    report['coverage_summary'] = coverage_stats

    # Aggregate power statistics
    power_keys = []
    for prefix in ['phase_w1', 'phase_w2', 'phase_w3', 'phase_1', 'phase_2', 'phase_3']:
        power_keys.extend([
            f'{prefix}_mean', f'{prefix}_std', f'{prefix}_max',
            f'{prefix}_cv', f'{prefix}_share_0_100', f'{prefix}_share_2000_plus'
        ])
    power_keys.extend(['total_mean', 'total_max', 'phase_balance_ratio', 'active_phases'])
    power_stats = _aggregate_section(house_analyses, 'power_statistics', power_keys)
    report['power_summary'] = power_stats

    # Aggregate quality scores
    quality_keys = ['quality_score']
    for c in ['w1', 'w2', 'w3', '1', '2', '3']:
        quality_keys.extend([
            f'{c}_negative_count', f'{c}_zero_pct', f'{c}_outliers_3sd_pct'
        ])
    quality_stats = _aggregate_section(house_analyses, 'data_quality', quality_keys)
    report['quality_summary'] = quality_stats

    # Aggregate temporal patterns
    temporal_keys = []
    for prefix in ['phase_w1', 'phase_w2', 'phase_w3', 'phase_1', 'phase_2', 'phase_3']:
        temporal_keys.extend([
            f'{prefix}_night_day_ratio', f'{prefix}_weekend_weekday_ratio',
            f'{prefix}_peak_hour', f'{prefix}_hourly_cv'
        ])
    temporal_keys.extend(['total_night_day_ratio', 'total_peak_hour'])
    temporal_stats = _aggregate_section(house_analyses, 'temporal_patterns', temporal_keys)
    report['temporal_summary'] = temporal_stats

    # Houses by quality tier
    quality_tiers = {'excellent': [], 'good': [], 'fair': [], 'poor': []}
    for analysis in house_analyses:
        house_id = analysis.get('house_id', 'unknown')
        score = analysis.get('data_quality', {}).get('quality_score', 0)
        if score >= 90:
            quality_tiers['excellent'].append(house_id)
        elif score >= 75:
            quality_tiers['good'].append(house_id)
        elif score >= 50:
            quality_tiers['fair'].append(house_id)
        else:
            quality_tiers['poor'].append(house_id)

    report['quality_tiers'] = {
        tier: {'count': len(houses), 'houses': houses}
        for tier, houses in quality_tiers.items()
    }

    # Houses with specific issues
    report['issues'] = {
        'low_coverage': [a['house_id'] for a in house_analyses
                         if a.get('flags', {}).get('low_coverage', False)],
        'low_quality': [a['house_id'] for a in house_analyses
                        if a.get('flags', {}).get('low_quality_score', False)],
        'unbalanced_phases': [a['house_id'] for a in house_analyses
                              if a.get('flags', {}).get('unbalanced_phases', False)],
        'negative_values': [a['house_id'] for a in house_analyses
                            if a.get('flags', {}).get('has_negative_values', False)],
    }

    # Wave behavior summary
    from collections import Counter
    wave_classifications = []
    wave_scores = []
    wave_dominant_houses = []
    has_waves_houses = []

    for analysis in house_analyses:
        wave = analysis.get('wave_behavior', {})
        cls = wave.get('wave_classification', 'no_waves')
        wave_classifications.append(cls)
        wave_scores.append(wave.get('max_wave_score', 0.0))
        house_id = analysis.get('house_id', 'unknown')
        if cls == 'wave_dominant':
            wave_dominant_houses.append(house_id)
        elif cls == 'has_waves':
            has_waves_houses.append(house_id)

    cls_counts = Counter(wave_classifications)
    report['wave_summary'] = {
        'n_wave_dominant': cls_counts.get('wave_dominant', 0),
        'n_has_waves': cls_counts.get('has_waves', 0),
        'n_no_waves': cls_counts.get('no_waves', 0),
        'avg_max_wave_score': sum(wave_scores) / len(wave_scores) if wave_scores else 0.0,
        'wave_dominant_houses': wave_dominant_houses,
        'has_waves_houses': has_waves_houses,
    }

    return report


def _aggregate_section(analyses: List[Dict], section_key: str,
                       metric_keys: List[str]) -> Dict[str, Any]:
    """
    Calculate aggregate statistics for a section of metrics.

    Args:
        analyses: List of analysis results
        section_key: Key for the section (e.g., 'coverage', 'power_statistics')
        metric_keys: List of metric names to aggregate

    Returns:
        Dictionary with min, max, mean, median for each metric
    """
    result = {}

    for key in metric_keys:
        values = []
        for analysis in analyses:
            section = analysis.get(section_key, {})
            if key in section:
                val = section[key]
                if isinstance(val, (int, float)) and not pd.isna(val):
                    values.append(val)

        if values:
            result[key] = {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'median': sorted(values)[len(values) // 2],
                'count': len(values)
            }

    return result


def generate_summary_report(aggregate: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Generate and save the aggregate summary report.

    Args:
        aggregate: Aggregated analysis from aggregate_all_houses
        output_dir: Directory to save the reports

    Returns:
        Dictionary with paths to generated files
    """
    os.makedirs(output_dir, exist_ok=True)

    # All aggregate data is already embedded in the HTML report.
    # No separate JSON/CSV files needed.
    return {}


def create_comparison_table(house_analyses: List[Dict[str, Any]],
                            output_path: str) -> pd.DataFrame:
    """
    Create a comparison table with one row per house.

    Args:
        house_analyses: List of analysis results
        output_path: Path to save the CSV

    Returns:
        DataFrame with comparison data
    """
    rows = []

    for analysis in house_analyses:
        row = {'house_id': analysis.get('house_id', 'unknown')}

        # Coverage
        coverage = analysis.get('coverage', {})
        row['days_span'] = coverage.get('days_span', 0)
        row['coverage_ratio'] = coverage.get('coverage_ratio', 0)
        row['months_count'] = coverage.get('months_count', 0)
        row['max_gap_minutes'] = coverage.get('max_gap_minutes', 0)

        # Quality
        quality = analysis.get('data_quality', {})
        row['quality_score'] = quality.get('quality_score', 0)

        # Power
        power = analysis.get('power_statistics', {})
        row['total_mean'] = power.get('total_mean', 0)
        row['total_max'] = power.get('total_max', 0)
        row['phase_balance_ratio'] = power.get('phase_balance_ratio', 0)
        row['active_phases'] = power.get('active_phases', 0)

        # Temporal
        temporal = analysis.get('temporal_patterns', {})
        row['total_night_day_ratio'] = temporal.get('total_night_day_ratio', 0)
        row['total_peak_hour'] = temporal.get('total_peak_hour', 0)

        # Flags (as boolean columns)
        flags = analysis.get('flags', {})
        for flag_name, flag_value in flags.items():
            row[f'flag_{flag_name}'] = flag_value

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    return df
