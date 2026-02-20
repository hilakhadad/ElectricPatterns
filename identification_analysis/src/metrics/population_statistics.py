"""
Cross-house population statistics and z-score analysis.

Computes population-level metrics across all houses to identify outliers
and provide context for per-house classification quality.

Uses robust statistics (median + MAD) to handle outlier houses.

Input: Per-house quality and confidence results (from classification_quality.py
       and confidence_scoring.py).
Output: Population statistics with per-house z-scores.
"""
import logging
from collections import defaultdict
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEVICE_TYPES = ['boiler', 'central_ac', 'regular_ac']

# Z-score thresholds
Z_EXTREME = 3.0
Z_WARNING = 2.0


# ============================================================================
# Main entry point
# ============================================================================

def compute_population_statistics(
    all_quality: List[Dict[str, Any]],
    all_confidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute population-level statistics from all houses' quality and confidence data.

    Args:
        all_quality: List of per-house quality dicts (from calculate_classification_quality)
        all_confidence: List of per-house confidence dicts (from calculate_confidence_scores)

    Returns:
        Dict with population stats, per-house z-scores, and outlier list.
    """
    if not all_quality:
        return _empty_population()

    # Build per-house feature vectors
    house_features = _extract_house_features(all_quality, all_confidence)
    if not house_features:
        return _empty_population()

    # Compute population distributions per device type
    per_device = _compute_device_distributions(house_features)

    # Compute z-scores for each house
    house_z_scores = _compute_house_z_scores(house_features, per_device)

    # Identify outlier houses
    outliers = _identify_outliers(house_z_scores)

    # Compute classification rate analysis (Metric H)
    classification_rates = _compute_classification_rates(all_quality, all_confidence)

    # Quality score distribution
    quality_scores = [q.get('overall_quality', 0) for q in all_quality
                      if q.get('overall_quality') is not None]
    quality_distribution = _compute_distribution(quality_scores) if quality_scores else {}

    # Confidence distribution
    conf_means = [c.get('confidence_summary', {}).get('mean', 0)
                  for c in all_confidence if c.get('total_activations', 0) > 0]
    confidence_distribution = _compute_distribution(conf_means) if conf_means else {}

    return {
        'houses_analyzed': len(house_features),
        'per_device_type': per_device,
        'house_z_scores': house_z_scores,
        'outlier_houses': outliers,
        'classification_rates': classification_rates,
        'quality_distribution': quality_distribution,
        'confidence_distribution': confidence_distribution,
    }


# ============================================================================
# Feature extraction
# ============================================================================

def _extract_house_features(
    all_quality: List[Dict[str, Any]],
    all_confidence: List[Dict[str, Any]],
) -> Dict[str, Dict]:
    """Extract per-house, per-device-type feature vectors."""
    # Index confidence by house_id
    conf_by_house = {c['house_id']: c for c in all_confidence if 'house_id' in c}

    features = {}
    for quality in all_quality:
        house_id = quality.get('house_id')
        if not house_id:
            continue

        temporal = quality.get('temporal_consistency', {})
        magnitude = quality.get('magnitude_stability', {})
        duration = quality.get('duration_plausibility', {})
        seasonal = quality.get('seasonal_coherence', {})

        conf = conf_by_house.get(house_id, {})
        conf_summary = conf.get('confidence_summary', {})

        house_feat = {
            'overall_quality': quality.get('overall_quality', 0),
            'quality_tier': quality.get('quality_tier', 'unknown'),
            'flags_count': len(quality.get('flags', [])),
            'avg_confidence': conf_summary.get('mean', 0),
            'high_confidence_rate': 0,
            'devices': {},
        }

        total_acts = conf.get('total_activations', 0)
        if total_acts > 0:
            high_count = conf_summary.get('high_count', 0)
            house_feat['high_confidence_rate'] = high_count / total_acts

        for dtype in DEVICE_TYPES:
            t = temporal.get(dtype, {})
            m = magnitude.get(dtype, {})
            d = duration.get(dtype, {})
            s = seasonal.get(dtype, {})

            house_feat['devices'][dtype] = {
                'temporal_consistency': t.get('value', 0),
                'months_with': t.get('months_with', 0),
                'magnitude_cv': m.get('cv', 0),
                'mean_magnitude': m.get('mean', 0),
                'phase_switching_rate': m.get('phase_switching_rate', 0),
                'median_duration': d.get('median', 0),
                'count': d.get('count', 0),
                'seasonal_ratio': s.get('ratio', 0),
            }

        features[house_id] = house_feat

    return features


# ============================================================================
# Population distributions
# ============================================================================

def _compute_device_distributions(
    house_features: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Compute population distributions per device type per metric."""
    per_device = {}

    for dtype in DEVICE_TYPES:
        metrics = defaultdict(list)

        for house_id, feat in house_features.items():
            dev = feat.get('devices', {}).get(dtype, {})
            count = dev.get('count', 0)
            if count == 0:
                continue

            metrics['count_per_month'].append(dev['months_with'])
            metrics['mean_magnitude'].append(dev['mean_magnitude'])
            metrics['magnitude_cv'].append(dev['magnitude_cv'])
            metrics['median_duration'].append(dev['median_duration'])
            metrics['temporal_consistency'].append(dev['temporal_consistency'])
            metrics['seasonal_ratio'].append(dev['seasonal_ratio'])

        device_dist = {}
        for metric_name, values in metrics.items():
            device_dist[metric_name] = _compute_distribution(values)

        device_dist['houses_with_device'] = sum(
            1 for feat in house_features.values()
            if feat.get('devices', {}).get(dtype, {}).get('count', 0) > 0
        )
        per_device[dtype] = device_dist

    return per_device


def _compute_distribution(values: list) -> Dict[str, float]:
    """Compute robust distribution statistics."""
    if not values:
        return {'median': 0, 'mad': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'n': 0}

    arr = np.array(values, dtype=float)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median))) or 1.0

    return {
        'median': round(median, 4),
        'mad': round(mad, 4),
        'mean': round(float(np.mean(arr)), 4),
        'std': round(float(np.std(arr)), 4),
        'min': round(float(np.min(arr)), 4),
        'max': round(float(np.max(arr)), 4),
        'n': len(arr),
    }


# ============================================================================
# Z-score computation
# ============================================================================

def _compute_house_z_scores(
    house_features: Dict[str, Dict],
    per_device: Dict[str, Dict],
) -> Dict[str, Dict]:
    """Compute per-house z-scores relative to population."""
    z_scores = {}

    z_metrics = ['count_per_month', 'mean_magnitude', 'magnitude_cv',
                 'median_duration', 'temporal_consistency']

    # Map from z_metric name to feature name
    metric_to_feature = {
        'count_per_month': 'months_with',
        'mean_magnitude': 'mean_magnitude',
        'magnitude_cv': 'magnitude_cv',
        'median_duration': 'median_duration',
        'temporal_consistency': 'temporal_consistency',
    }

    for house_id, feat in house_features.items():
        house_z = {}

        for dtype in DEVICE_TYPES:
            dev = feat.get('devices', {}).get(dtype, {})
            if dev.get('count', 0) == 0:
                continue

            dist = per_device.get(dtype, {})
            dtype_z = {}

            for z_metric in z_metrics:
                metric_dist = dist.get(z_metric, {})
                median = metric_dist.get('median', 0)
                mad = metric_dist.get('mad', 1)

                feature_name = metric_to_feature[z_metric]
                value = dev.get(feature_name, 0)

                z = (value - median) / mad if mad > 0 else 0
                dtype_z[z_metric] = round(z, 2)

            dtype_z['max_abs_z'] = round(max(abs(v) for v in dtype_z.values()) if dtype_z else 0, 2)
            house_z[dtype] = dtype_z

        z_scores[house_id] = house_z

    return z_scores


def _identify_outliers(house_z_scores: Dict[str, Dict]) -> List[Dict]:
    """Identify houses with extreme z-scores."""
    outliers = []

    for house_id, device_z in house_z_scores.items():
        extreme_flags = []
        warning_count = 0

        for dtype, z_dict in device_z.items():
            for metric, z_val in z_dict.items():
                if metric == 'max_abs_z':
                    continue
                if abs(z_val) > Z_EXTREME:
                    extreme_flags.append({
                        'device_type': dtype,
                        'metric': metric,
                        'z_score': z_val,
                        'severity': 'extreme',
                    })
                elif abs(z_val) > Z_WARNING:
                    warning_count += 1

        if extreme_flags or warning_count >= 3:
            outliers.append({
                'house_id': house_id,
                'extreme_flags': extreme_flags,
                'warning_count': warning_count,
                'reason': 'extreme_z' if extreme_flags else 'multi_warning',
            })

    return sorted(outliers, key=lambda x: len(x['extreme_flags']), reverse=True)


# ============================================================================
# Classification rate analysis (Metric H)
# ============================================================================

def _compute_classification_rates(
    all_quality: List[Dict[str, Any]],
    all_confidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute population-level classification rate statistics."""
    conf_by_house = {c['house_id']: c for c in all_confidence if 'house_id' in c}

    raw_rates = []
    weighted_rates = []
    high_conf_rates = []
    per_type_stats = defaultdict(lambda: {'counts': [], 'avg_confidences': []})

    for quality in all_quality:
        house_id = quality.get('house_id')
        if not house_id:
            continue

        # Raw classification rate from quality flags
        temporal = quality.get('temporal_consistency', {})
        total_months = 0
        classified_months = 0
        for dtype in DEVICE_TYPES:
            t = temporal.get(dtype, {})
            months = t.get('months_with', 0)
            classified_months += months

        conf = conf_by_house.get(house_id, {})
        total_acts = conf.get('total_activations', 0)
        if total_acts == 0:
            continue

        conf_summary = conf.get('confidence_summary', {})
        high_count = conf_summary.get('high_count', 0)
        mean_conf = conf_summary.get('mean', 0)

        high_conf_rate = high_count / total_acts
        high_conf_rates.append(high_conf_rate)

        # Collect per-activation data
        activations = conf.get('activations', [])
        total_confidence_sum = sum(a.get('confidence', 0) for a in activations)
        weighted_rate = total_confidence_sum / total_acts if total_acts > 0 else 0
        weighted_rates.append(weighted_rate)

    return {
        'high_conf_rates': _compute_distribution(high_conf_rates),
        'weighted_rates': _compute_distribution(weighted_rates),
        'houses_analyzed': len(weighted_rates),
    }


# ============================================================================
# Helpers
# ============================================================================

def _empty_population() -> Dict[str, Any]:
    """Return empty population statistics structure."""
    return {
        'houses_analyzed': 0,
        'per_device_type': {},
        'house_z_scores': {},
        'outlier_houses': [],
        'classification_rates': {},
        'quality_distribution': {},
        'confidence_distribution': {},
    }
