"""
Classification quality metrics for device identification evaluation.

Computes internal consistency metrics WITHOUT ground truth:
  A. Temporal consistency — does the device appear consistently across months?
  B. Magnitude stability — are magnitudes consistent for each device type?
  C. Duration plausibility — do durations match physical expectations?
  D. Seasonal coherence — do activations follow expected seasonal patterns?
  E. Energy conservation — no double-counting or impossible values?

Input: device_activations_{house_id}.json (unified JSON from pipeline)
Output: per-house quality dict with metrics, flags, and overall score.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEVICE_TYPES = ['boiler', 'central_ac', 'regular_ac', 'unclassified']
CLASSIFIED_TYPES = ['boiler', 'central_ac', 'regular_ac']

# Israeli seasons: warm = May-October, cool = November-April
WARM_MONTHS = {5, 6, 7, 8, 9, 10}
COOL_MONTHS = {1, 2, 3, 4, 11, 12}


# ============================================================================
# Main entry point
# ============================================================================

def calculate_classification_quality(
    experiment_dir: Path,
    house_id: str,
) -> Dict[str, Any]:
    """
    Calculate all classification quality metrics for a house.

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID string

    Returns:
        Dict with metrics A-E, flags, overall quality score and tier.
    """
    activations = _load_activations(experiment_dir, house_id)
    if activations is None:
        return _empty_quality(house_id)

    matched = [a for a in activations if a.get('match_type') == 'matched']
    if not matched:
        return _empty_quality(house_id)

    # Group activations by device type
    by_type = defaultdict(list)
    for a in matched:
        by_type[a.get('device_type', 'unclassified')].append(a)

    # Extract month coverage from data
    months_with_data = _get_months_with_data(matched)

    # Compute each metric
    temporal = _metric_temporal_consistency(by_type, months_with_data)
    magnitude = _metric_magnitude_stability(by_type)
    duration = _metric_duration_plausibility(by_type)
    seasonal = _metric_seasonal_coherence(by_type, months_with_data)
    energy = _metric_energy_conservation(matched, by_type)

    # Collect all flags
    all_flags = []
    for metric_name, metric_result in [
        ('temporal_consistency', temporal),
        ('magnitude_stability', magnitude),
        ('duration_plausibility', duration),
        ('seasonal_coherence', seasonal),
        ('energy_conservation', energy),
    ]:
        for dtype, dtype_data in metric_result.items():
            if isinstance(dtype_data, dict) and dtype_data.get('flag'):
                all_flags.append({
                    'metric': metric_name,
                    'device_type': dtype,
                    'flag': dtype_data['flag'],
                    'detail': dtype_data.get('flag_detail', ''),
                })

    # Overall quality score
    quality_score = _compute_overall_quality(
        temporal, magnitude, duration, seasonal, energy
    )
    quality_tier = _score_to_tier(quality_score)

    return {
        'house_id': house_id,
        'total_activations': len(matched),
        'data_months': len(months_with_data),
        'metrics': {
            'temporal_consistency': temporal,
            'magnitude_stability': magnitude,
            'duration_plausibility': duration,
            'seasonal_coherence': seasonal,
            'energy_conservation': energy,
        },
        'flags': all_flags,
        'overall_quality_score': round(quality_score, 3),
        'quality_tier': quality_tier,
    }


# ============================================================================
# Metric A: Temporal Consistency
# ============================================================================

def _metric_temporal_consistency(
    by_type: Dict[str, List],
    months_with_data: set,
) -> Dict[str, Any]:
    """
    For each device type, compute fraction of months with at least one activation.
    """
    total_months = len(months_with_data)
    if total_months == 0:
        return {}

    result = {}
    for dtype in CLASSIFIED_TYPES:
        activations = by_type.get(dtype, [])
        if not activations:
            continue

        months_with_device = set()
        for a in activations:
            month_key = _extract_month_key(a)
            if month_key:
                months_with_device.add(month_key)

        value = len(months_with_device) / total_months
        flag, flag_detail = _check_temporal_flags(dtype, value, len(activations), total_months)

        result[dtype] = {
            'value': round(value, 3),
            'months_with': len(months_with_device),
            'months_total': total_months,
            'flag': flag,
            'flag_detail': flag_detail,
        }

    return result


def _check_temporal_flags(dtype, value, count, total_months):
    """Check for temporal consistency anomalies."""
    if dtype == 'boiler' and value < 0.50:
        return 'TEMPORAL_ANOMALY', f'boiler consistency={value:.2f}, expected >=0.50'
    if dtype in ('regular_ac', 'central_ac') and value > 0.90:
        return 'TEMPORAL_ANOMALY', f'AC consistency={value:.2f}, expected <0.90 (seasonal)'
    if value < 0.20 and count > 5:
        return 'TEMPORAL_SPARSE', f'{count} activations in only {int(value * total_months)} months'
    return None, None


# ============================================================================
# Metric B: Magnitude Stability
# ============================================================================

def _metric_magnitude_stability(
    by_type: Dict[str, List],
) -> Dict[str, Any]:
    """
    For each device type, compute CV of magnitudes and phase switching rate.
    """
    result = {}
    for dtype in CLASSIFIED_TYPES:
        activations = by_type.get(dtype, [])
        if not activations:
            continue

        magnitudes = [abs(a['magnitude']) for a in activations if 'magnitude' in a]
        if len(magnitudes) < 2:
            result[dtype] = {
                'mean': round(magnitudes[0], 1) if magnitudes else 0,
                'std': 0, 'cv': 0,
                'phase_distribution': _phase_distribution(activations),
                'phase_switching_rate': 0,
                'flag': None, 'flag_detail': None,
            }
            continue

        mean_mag = np.mean(magnitudes)
        std_mag = np.std(magnitudes)
        cv = std_mag / mean_mag if mean_mag > 0 else 0

        phase_dist = _phase_distribution(activations)
        total_count = sum(phase_dist.values())
        max_phase_count = max(phase_dist.values()) if phase_dist else 0
        phase_switching = 1 - (max_phase_count / total_count) if total_count > 0 else 0

        flag, flag_detail = _check_magnitude_flags(dtype, cv, phase_switching, mean_mag)

        result[dtype] = {
            'mean': round(mean_mag, 1),
            'std': round(std_mag, 1),
            'cv': round(cv, 3),
            'phase_distribution': phase_dist,
            'phase_switching_rate': round(phase_switching, 3),
            'flag': flag, 'flag_detail': flag_detail,
        }

    return result


def _check_magnitude_flags(dtype, cv, phase_switching, mean_mag):
    """Check for magnitude stability anomalies."""
    if dtype == 'boiler':
        if cv > 0.30:
            return 'MAGNITUDE_UNSTABLE', f'boiler CV={cv:.2f}, expected <0.30'
        if phase_switching > 0.10:
            return 'PHASE_SWITCHING', f'boiler phase_switching={phase_switching:.2f}, expected <0.10'
        if mean_mag < 1500 or mean_mag > 5000:
            return 'MAGNITUDE_SUSPICIOUS', f'boiler mean={mean_mag:.0f}W, expected 1500-5000W'
    if dtype == 'regular_ac' and cv > 0.40:
        return 'MAGNITUDE_UNSTABLE', f'regular_ac CV={cv:.2f}, expected <0.40'
    return None, None


# ============================================================================
# Metric C: Duration Distribution Plausibility
# ============================================================================

def _metric_duration_plausibility(
    by_type: Dict[str, List],
) -> Dict[str, Any]:
    """
    For each device type, check if duration distributions match physical expectations.
    """
    result = {}
    for dtype in CLASSIFIED_TYPES:
        activations = by_type.get(dtype, [])
        if not activations:
            continue

        durations = [a['duration'] for a in activations if 'duration' in a and a['duration'] > 0]
        if not durations:
            continue

        durations_arr = np.array(durations)
        median_dur = float(np.median(durations_arr))
        q25 = float(np.percentile(durations_arr, 25))
        q75 = float(np.percentile(durations_arr, 75))
        frac_below_25 = float(np.mean(durations_arr < 25))
        frac_above_180 = float(np.mean(durations_arr > 180))

        flag, flag_detail = _check_duration_flags(dtype, median_dur, frac_above_180, frac_below_25, durations_arr)

        entry = {
            'count': len(durations),
            'median': round(median_dur, 1),
            'q25': round(q25, 1),
            'q75': round(q75, 1),
            'frac_below_25min': round(frac_below_25, 3),
            'frac_above_180min': round(frac_above_180, 3),
            'flag': flag, 'flag_detail': flag_detail,
        }

        # AC-specific: count initial runs vs cycling events
        if dtype in ('regular_ac', 'central_ac'):
            cycling = int(np.sum((durations_arr >= 3) & (durations_arr <= 30)))
            initial = int(np.sum(durations_arr > 30))
            entry['cycling_events'] = cycling
            entry['initial_runs'] = initial

        result[dtype] = entry

    return result


def _check_duration_flags(dtype, median_dur, frac_above_180, frac_below_25, durations):
    """Check for duration distribution anomalies."""
    if dtype == 'boiler':
        if median_dur > 180:
            return 'DURATION_ANOMALY', f'boiler median={median_dur:.0f}min, expected <180'
        if frac_below_25 > 0.05 and len(durations) > 5:
            return 'DURATION_TOO_SHORT', f'boiler {frac_below_25:.0%} activations <25min'
    if dtype in ('regular_ac', 'central_ac'):
        if frac_above_180 > 0.10:
            return 'DURATION_ANOMALY', f'AC {frac_above_180:.0%} activations >180min'
        # Check for missing cycling mode
        cycling = np.sum((durations >= 3) & (durations <= 30))
        if cycling == 0 and len(durations) > 5:
            return 'DURATION_MISSING_MODE', f'AC has no cycling events (3-30min), {len(durations)} total'
    return None, None


# ============================================================================
# Metric D: Seasonal Coherence
# ============================================================================

def _metric_seasonal_coherence(
    by_type: Dict[str, List],
    months_with_data: set,
) -> Dict[str, Any]:
    """
    Compare warm-month vs cool-month activation counts per device type.
    """
    # Check we have data in both seasons
    has_warm = any(m[0] in WARM_MONTHS for m in months_with_data)
    has_cool = any(m[0] in COOL_MONTHS for m in months_with_data)
    if not has_warm or not has_cool:
        return {}

    result = {}
    for dtype in CLASSIFIED_TYPES:
        activations = by_type.get(dtype, [])
        if not activations:
            continue

        warm_count = 0
        cool_count = 0
        for a in activations:
            month = _extract_month(a)
            if month is None:
                continue
            if month in WARM_MONTHS:
                warm_count += 1
            else:
                cool_count += 1

        # Seasonal ratio with epsilon to avoid division by zero
        ratio = warm_count / max(cool_count, 1)

        flag, flag_detail = _check_seasonal_flags(dtype, ratio, len(months_with_data))

        result[dtype] = {
            'warm_count': warm_count,
            'cool_count': cool_count,
            'ratio': round(ratio, 2),
            'flag': flag, 'flag_detail': flag_detail,
        }

    return result


def _check_seasonal_flags(dtype, ratio, total_months):
    """Check for seasonal coherence anomalies."""
    if dtype in ('regular_ac', 'central_ac'):
        if ratio < 1.0:
            return 'SEASONAL_INVERTED', f'AC ratio={ratio:.2f}, expected >1.0 (more in summer)'
        if 0.8 <= ratio <= 1.2 and total_months >= 6:
            return 'SEASONAL_FLAT', f'AC ratio={ratio:.2f}, expected >2.0 for clear seasonality'
    if dtype == 'boiler' and ratio > 3.0:
        return 'SEASONAL_INVERTED', f'boiler ratio={ratio:.2f}, expected <3.0 (year-round use)'
    return None, None


# ============================================================================
# Metric E: Energy Conservation
# ============================================================================

def _metric_energy_conservation(
    matched: List[Dict],
    by_type: Dict[str, List],
) -> Dict[str, Any]:
    """
    Check for energy conservation: no double-counting, no impossible values.
    """
    # Cross-iteration overlap check
    overlaps = _check_cross_iteration_overlaps(matched)

    # Total classified energy (magnitude * duration in watt-minutes)
    total_energy = 0
    classified_energy = 0
    for a in matched:
        energy = abs(a.get('magnitude', 0)) * a.get('duration', 0)
        total_energy += energy
        if a.get('device_type', 'unclassified') != 'unclassified':
            classified_energy += energy

    classified_ratio = classified_energy / total_energy if total_energy > 0 else 0

    flag = None
    flag_detail = None
    if overlaps > 0:
        flag = 'ENERGY_OVERLAP'
        flag_detail = f'{overlaps} cross-iteration overlaps detected'

    return {
        'classified_energy_ratio': round(classified_ratio, 3),
        'total_energy_wattmin': round(total_energy, 0),
        'classified_energy_wattmin': round(classified_energy, 0),
        'cross_iteration_overlaps': overlaps,
        'flag': flag, 'flag_detail': flag_detail,
    }


def _check_cross_iteration_overlaps(matched: List[Dict]) -> int:
    """Count activations from different iterations that overlap on the same phase."""
    # Group by phase
    by_phase = defaultdict(list)
    for a in matched:
        phase = a.get('phase', '')
        by_phase[phase].append(a)

    overlap_count = 0
    for phase, phase_acts in by_phase.items():
        # Sort by start time
        sorted_acts = sorted(phase_acts, key=lambda x: x.get('on_start', ''))
        for i in range(len(sorted_acts) - 1):
            a1 = sorted_acts[i]
            a2 = sorted_acts[i + 1]
            # Check if different iterations overlap
            if a1.get('iteration') != a2.get('iteration'):
                if a1.get('off_end', '') > a2.get('on_start', ''):
                    overlap_count += 1

    return overlap_count


# ============================================================================
# Overall Quality Score
# ============================================================================

def _compute_overall_quality(temporal, magnitude, duration, seasonal, energy):
    """
    Combine individual metrics into a single quality score (0.0-1.0).

    Weights: temporal=0.20, magnitude=0.25, duration=0.15,
             seasonal=0.15, energy=0.10, classification=0.15
    """
    weights = {
        'temporal': 0.25,
        'magnitude': 0.30,
        'duration': 0.20,
        'seasonal': 0.15,
        'energy': 0.10,
    }

    scores = {
        'temporal': _metric_score(temporal),
        'magnitude': _metric_score(magnitude),
        'duration': _metric_score(duration),
        'seasonal': _metric_score(seasonal),
        'energy': 1.0 if not energy.get('flag') else 0.5,
    }

    total = sum(weights[k] * scores[k] for k in weights)
    return min(1.0, max(0.0, total))


def _metric_score(metric_dict: Dict) -> float:
    """Convert a per-device-type metric dict to a single score (0-1)."""
    if not metric_dict:
        return 0.5  # No data = neutral

    flags = []
    for dtype, data in metric_dict.items():
        if isinstance(data, dict) and data.get('flag'):
            flags.append(data['flag'])

    if not flags:
        return 1.0  # No flags = perfect
    # Count severity
    critical_flags = [f for f in flags if 'ANOMALY' in f or 'SWITCHING' in f or 'INVERTED' in f]
    if critical_flags:
        return 0.0
    return 0.5  # Warning-level flags


def _score_to_tier(score: float) -> str:
    """Convert quality score to tier name."""
    if score >= 0.85:
        return 'excellent'
    if score >= 0.65:
        return 'good'
    if score >= 0.45:
        return 'fair'
    return 'poor'


# ============================================================================
# Helpers
# ============================================================================

def _load_activations(experiment_dir: Path, house_id: str) -> Optional[List[Dict]]:
    """Load device activations list from JSON."""
    experiment_dir = Path(experiment_dir)
    json_path = experiment_dir / "device_activations" / f"device_activations_{house_id}.json"
    if not json_path.exists():
        json_path = experiment_dir / f"device_activations_{house_id}.json"

    if not json_path.exists():
        logger.warning(f"Device activations JSON not found for house {house_id}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('activations', [])
    except Exception as e:
        logger.warning(f"Failed to load device activations: {e}")
        return None


def _get_months_with_data(activations: List[Dict]) -> set:
    """Get set of (month, year) tuples that have activations."""
    months = set()
    for a in activations:
        key = _extract_month_key(a)
        if key:
            months.add(key)
    return months


def _extract_month_key(activation: Dict):
    """Extract (month, year) tuple from activation timestamps."""
    ts = activation.get('on_start', '')
    if not ts:
        return None
    try:
        # Format: "2024-01-15 08:30:00"
        parts = ts.split('-')
        year = int(parts[0])
        month = int(parts[1])
        return (month, year)
    except (ValueError, IndexError):
        return None


def _extract_month(activation: Dict) -> Optional[int]:
    """Extract month number (1-12) from activation."""
    key = _extract_month_key(activation)
    return key[0] if key else None


def _phase_distribution(activations: List[Dict]) -> Dict[str, int]:
    """Count activations per phase."""
    dist = defaultdict(int)
    for a in activations:
        phase = a.get('phase', 'unknown')
        dist[phase] += 1
    return dict(dist)


def _empty_quality(house_id: str) -> Dict[str, Any]:
    """Return empty quality structure."""
    return {
        'house_id': house_id,
        'total_activations': 0,
        'data_months': 0,
        'metrics': {
            'temporal_consistency': {},
            'magnitude_stability': {},
            'duration_plausibility': {},
            'seasonal_coherence': {},
            'energy_conservation': {},
        },
        'flags': [],
        'overall_quality_score': 0,
        'quality_tier': 'poor',
    }
