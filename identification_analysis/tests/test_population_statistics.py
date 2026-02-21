"""
Tests for population statistics (cross-house z-scores).

Tests the population-level analysis using synthetic per-house data,
without requiring actual experiment output files.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from metrics.population_statistics import (
    compute_population_statistics,
    _compute_distribution,
    _extract_house_features,
    _compute_house_z_scores,
    _compute_device_distributions,
    _identify_outliers,
)


# ============================================================================
# Fixtures: synthetic quality and confidence data
# ============================================================================

def _make_quality(house_id, boiler_temporal=0.9, boiler_cv=0.10,
                  boiler_median_dur=40, overall_quality=0.8):
    """Create a synthetic quality dict matching classification_quality.py output."""
    return {
        'house_id': house_id,
        'overall_quality_score': overall_quality,
        'quality_tier': 'good',
        'total_activations': 30,
        'data_months': 12,
        'metrics': {
            'temporal_consistency': {
                'boiler': {'value': boiler_temporal, 'months_with': int(boiler_temporal * 12), 'flag': None},
            },
            'magnitude_stability': {
                'boiler': {'mean': 2100, 'std': 2100 * boiler_cv, 'cv': boiler_cv,
                           'phase_switching_rate': 0.0, 'flag': None},
            },
            'duration_plausibility': {
                'boiler': {'count': 30, 'median': boiler_median_dur, 'q25': 30, 'q75': 55, 'flag': None},
            },
            'seasonal_coherence': {
                'boiler': {'warm_count': 15, 'cool_count': 15, 'ratio': 1.0, 'flag': None},
            },
            'energy_conservation': {'flag': None},
        },
        'flags': [],
    }


def _make_confidence(house_id, total=50, mean_conf=0.7, high_count=30):
    """Create a synthetic confidence dict for one house."""
    activations = [{'confidence': mean_conf, 'confidence_tier': 'high'}] * total
    return {
        'house_id': house_id,
        'total_activations': total,
        'confidence_summary': {
            'mean': mean_conf,
            'median': mean_conf,
            'high_count': high_count,
            'medium_count': total - high_count,
            'low_count': 0,
        },
        'activations': activations,
    }


def _make_population(n_houses=10):
    """Create a population of synthetic houses with slight variations."""
    qualities = []
    confidences = []
    for i in range(n_houses):
        hid = str(100 + i)
        qualities.append(_make_quality(
            hid,
            boiler_temporal=0.85 + i * 0.01,
            boiler_cv=0.08 + i * 0.005,
            boiler_median_dur=38 + i,
            overall_quality=0.75 + i * 0.02,
        ))
        confidences.append(_make_confidence(hid, mean_conf=0.65 + i * 0.02))
    return qualities, confidences


# ============================================================================
# Tests: Distribution computation
# ============================================================================

class TestDistribution:
    def test_basic_distribution(self):
        result = _compute_distribution([1, 2, 3, 4, 5])
        assert result['median'] == 3.0
        assert result['n'] == 5
        assert result['min'] == 1.0
        assert result['max'] == 5.0

    def test_empty_distribution(self):
        result = _compute_distribution([])
        assert result['n'] == 0
        assert result['median'] == 0

    def test_single_value(self):
        result = _compute_distribution([42.0])
        assert result['median'] == 42.0
        assert result['n'] == 1
        assert result['mad'] == 1.0  # MAD defaults to 1.0 when computed as 0


# ============================================================================
# Tests: Feature extraction
# ============================================================================

class TestFeatureExtraction:
    def test_extract_features(self):
        qualities, confidences = _make_population(3)
        features = _extract_house_features(qualities, confidences)

        assert len(features) == 3
        assert '100' in features
        assert 'boiler' in features['100']['devices']
        assert features['100']['devices']['boiler']['count'] == 30

    def test_empty_input(self):
        features = _extract_house_features([], [])
        assert features == {}


# ============================================================================
# Tests: Population statistics
# ============================================================================

class TestPopulationStatistics:
    def test_basic_population(self):
        qualities, confidences = _make_population(10)
        result = compute_population_statistics(qualities, confidences)

        assert result['houses_analyzed'] == 10
        assert 'boiler' in result['per_device_type']
        assert 'house_z_scores' in result
        assert len(result['house_z_scores']) == 10

    def test_empty_population(self):
        result = compute_population_statistics([], [])
        assert result['houses_analyzed'] == 0

    def test_quality_distribution(self):
        qualities, confidences = _make_population(10)
        result = compute_population_statistics(qualities, confidences)

        assert 'quality_distribution' in result
        assert result['quality_distribution']['n'] == 10

    def test_confidence_distribution(self):
        qualities, confidences = _make_population(10)
        result = compute_population_statistics(qualities, confidences)

        assert 'confidence_distribution' in result
        assert result['confidence_distribution']['n'] == 10


# ============================================================================
# Tests: Z-scores and outliers
# ============================================================================

class TestZScores:
    def test_z_scores_near_zero_for_normal_house(self):
        """A house near the population median should have low z-scores."""
        qualities, confidences = _make_population(10)
        features = _extract_house_features(qualities, confidences)
        per_device = _compute_device_distributions(features)
        z_scores = _compute_house_z_scores(features, per_device)

        # Middle house (index ~5) should have z-scores near zero
        mid_house = '105'
        if mid_house in z_scores and 'boiler' in z_scores[mid_house]:
            for metric, z in z_scores[mid_house]['boiler'].items():
                if metric != 'max_abs_z':
                    assert abs(z) < 2.0, f"z-score for {metric} too high: {z}"

    def test_outlier_detection(self):
        """An extreme house should be flagged as outlier."""
        qualities, confidences = _make_population(10)

        # Add an extreme outlier
        qualities.append(_make_quality(
            '999', boiler_temporal=0.1, boiler_cv=0.80,
            boiler_median_dur=200, overall_quality=0.2,
        ))
        confidences.append(_make_confidence('999', mean_conf=0.2, high_count=2))

        features = _extract_house_features(qualities, confidences)
        per_device = _compute_device_distributions(features)
        z_scores = _compute_house_z_scores(features, per_device)
        outliers = _identify_outliers(z_scores)

        outlier_ids = [o['house_id'] for o in outliers]
        assert '999' in outlier_ids

    def test_no_outliers_in_homogeneous_population(self):
        """A homogeneous population should have no outliers."""
        qualities = [_make_quality(str(i), boiler_temporal=0.9, boiler_cv=0.10)
                     for i in range(20)]
        confidences = [_make_confidence(str(i)) for i in range(20)]

        result = compute_population_statistics(qualities, confidences)
        assert len(result['outlier_houses']) == 0


# ============================================================================
# Tests: Classification rates
# ============================================================================

class TestClassificationRates:
    def test_rates_computed(self):
        qualities, confidences = _make_population(5)
        result = compute_population_statistics(qualities, confidences)

        rates = result['classification_rates']
        assert 'high_conf_rates' in rates
        assert 'weighted_rates' in rates
        assert rates['houses_analyzed'] == 5
