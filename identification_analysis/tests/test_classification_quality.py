"""
Tests for classification quality metrics and confidence scoring.

Tests the core computation logic using synthetic session/activation data,
without requiring actual experiment output files.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from metrics.classification_quality import (
    _metric_temporal_consistency,
    _metric_magnitude_stability,
    _metric_duration_plausibility,
    _metric_seasonal_coherence,
    _metric_energy_conservation,
    _compute_overall_quality,
    _score_to_tier,
)
from metrics.confidence_scoring import (
    _score_tag_quality,
    _score_iteration,
    _score_seasonal_fit,
    _score_phase_consistency,
    _compute_type_statistics,
    _compute_dominant_phases,
    _confidence_tier,
)


# ============================================================================
# Fixtures: synthetic activations (for confidence scoring tests)
# ============================================================================

def _make_activation(device_type='boiler', phase='w1', magnitude=2100,
                     duration=45, on_start='2024-06-15 08:30:00',
                     tag='EXACT-EXTENDED', iteration=0, match_type='matched'):
    return {
        'device_type': device_type,
        'phase': phase,
        'magnitude': magnitude,
        'duration': duration,
        'on_start': on_start,
        'on_end': '2024-06-15 08:31:00',
        'off_start': '2024-06-15 09:15:00',
        'off_end': '2024-06-15 09:16:00',
        'tag': tag,
        'iteration': iteration,
        'threshold': 2000,
        'match_type': match_type,
    }


# ============================================================================
# Fixtures: synthetic sessions (for classification quality tests)
# ============================================================================

def _make_session(device_type='boiler', phases=None, avg_magnitude=2100,
                  duration_minutes=45, start='2024-06-15 08:30:00',
                  end='2024-06-15 09:15:00', confidence=0.85):
    if phases is None:
        phases = ['w1']
    return {
        'device_type': device_type,
        'phases': phases,
        'avg_cycle_magnitude_w': avg_magnitude,
        'duration_minutes': duration_minutes,
        'start': start,
        'end': end,
        'confidence': confidence,
    }


def _boiler_sessions_12_months():
    """12 months of boiler sessions, one per month."""
    sessions = []
    for m in range(1, 13):
        sessions.append(_make_session(
            device_type='boiler', phases=['w1'], avg_magnitude=2100 + m * 10,
            duration_minutes=40 + m, start=f'2024-{m:02d}-15 08:30:00',
            end=f'2024-{m:02d}-15 09:15:00',
        ))
    return sessions


def _ac_sessions_summer():
    """AC sessions only in warm months (May-Oct)."""
    sessions = []
    for m in [5, 6, 7, 8, 9, 10]:
        for day in [5, 15, 25]:
            sessions.append(_make_session(
                device_type='regular_ac', phases=['w2'], avg_magnitude=1200,
                duration_minutes=120, start=f'2024-{m:02d}-{day:02d} 14:00:00',
                end=f'2024-{m:02d}-{day:02d} 16:00:00',
            ))
    return sessions


# ============================================================================
# Metric A: Temporal Consistency
# ============================================================================

class TestTemporalConsistency:
    def test_boiler_12_months(self):
        by_type = {'boiler': _boiler_sessions_12_months()}
        months = {(m, 2024) for m in range(1, 13)}
        result = _metric_temporal_consistency(by_type, months)

        assert 'boiler' in result
        assert result['boiler']['value'] == 1.0
        assert result['boiler']['months_with'] == 12
        assert result['boiler']['flag'] is None

    def test_ac_seasonal(self):
        by_type = {'regular_ac': _ac_sessions_summer()}
        months = {(m, 2024) for m in range(1, 13)}
        result = _metric_temporal_consistency(by_type, months)

        assert 'regular_ac' in result
        assert result['regular_ac']['value'] == 0.5  # 6 out of 12 months
        assert result['regular_ac']['flag'] is None

    def test_boiler_low_consistency_flags(self):
        # Boiler only in 4 months
        sessions = [_make_session(start=f'2024-{m:02d}-15 08:00:00') for m in [1, 2, 3, 4]]
        by_type = {'boiler': sessions}
        months = {(m, 2024) for m in range(1, 13)}
        result = _metric_temporal_consistency(by_type, months)

        assert result['boiler']['value'] == pytest.approx(4 / 12, abs=0.01)
        assert result['boiler']['flag'] == 'TEMPORAL_ANOMALY'


# ============================================================================
# Metric B: Magnitude Stability
# ============================================================================

class TestMagnitudeStability:
    def test_stable_boiler(self):
        sessions = [_make_session(avg_magnitude=2100 + i * 5) for i in range(20)]
        by_type = {'boiler': sessions}
        result = _metric_magnitude_stability(by_type)

        assert 'boiler' in result
        assert result['boiler']['cv'] < 0.05  # Very stable
        assert result['boiler']['flag'] is None

    def test_unstable_boiler_flags(self):
        # Magnitudes vary from 1500 to 4000
        sessions = [_make_session(avg_magnitude=m) for m in [1500, 2000, 3000, 4000, 1800]]
        by_type = {'boiler': sessions}
        result = _metric_magnitude_stability(by_type)

        assert result['boiler']['cv'] > 0.30
        assert result['boiler']['flag'] == 'MAGNITUDE_UNSTABLE'

    def test_phase_switching_detected(self):
        sessions = []
        for i in range(10):
            sessions.append(_make_session(phases=['w1'], avg_magnitude=2100))
        # Add 5 on different phase
        for i in range(5):
            sessions.append(_make_session(phases=['w2'], avg_magnitude=2100))
        by_type = {'boiler': sessions}
        result = _metric_magnitude_stability(by_type)

        # 15 total, 10 on w1 → switching rate = 1 - 10/15 = 0.333
        assert result['boiler']['phase_switching_rate'] > 0.10
        assert result['boiler']['flag'] == 'PHASE_SWITCHING'


# ============================================================================
# Metric C: Duration Plausibility
# ============================================================================

class TestDurationPlausibility:
    def test_normal_boiler_durations(self):
        sessions = [_make_session(duration_minutes=d) for d in [35, 40, 45, 50, 55, 42, 38]]
        by_type = {'boiler': sessions}
        result = _metric_duration_plausibility(by_type)

        assert 'boiler' in result
        assert result['boiler']['flag'] is None
        assert 30 < result['boiler']['median'] < 60

    def test_boiler_too_long_flags(self):
        sessions = [_make_session(duration_minutes=d) for d in [250, 260, 280, 300, 270]]
        by_type = {'boiler': sessions}
        result = _metric_duration_plausibility(by_type)

        assert result['boiler']['flag'] == 'DURATION_ANOMALY'


# ============================================================================
# Metric D: Seasonal Coherence
# ============================================================================

class TestSeasonalCoherence:
    def test_ac_summer_bias(self):
        by_type = {'regular_ac': _ac_sessions_summer()}
        months = {(m, 2024) for m in range(1, 13)}
        result = _metric_seasonal_coherence(by_type, months)

        assert 'regular_ac' in result
        assert result['regular_ac']['warm_count'] > 0
        assert result['regular_ac']['cool_count'] == 0
        assert result['regular_ac']['flag'] is None  # ratio is warm/max(cool,1) > 2

    def test_ac_inverted_flags(self):
        # More AC in winter than summer
        winter_sessions = [_make_session(device_type='regular_ac',
                                         start=f'2024-{m:02d}-15 14:00:00')
                           for m in [1, 2, 3, 12] for _ in range(5)]
        summer_sessions = [_make_session(device_type='regular_ac',
                                         start=f'2024-{m:02d}-15 14:00:00')
                           for m in [7, 8]]
        by_type = {'regular_ac': winter_sessions + summer_sessions}
        months = {(m, 2024) for m in range(1, 13)}
        result = _metric_seasonal_coherence(by_type, months)

        assert result['regular_ac']['ratio'] < 1.0
        assert result['regular_ac']['flag'] == 'SEASONAL_INVERTED'


# ============================================================================
# Metric E: Energy Conservation
# ============================================================================

class TestEnergyConservation:
    def test_no_overlaps(self):
        sessions = [
            _make_session(start='2024-01-15 08:00:00', end='2024-01-15 09:00:00'),
            _make_session(start='2024-01-15 10:00:00', end='2024-01-15 11:00:00'),
        ]
        by_type = {'boiler': sessions}
        result = _metric_energy_conservation(sessions, by_type)

        assert result['session_overlaps'] == 0
        assert result['flag'] is None


# ============================================================================
# Overall Quality Score
# ============================================================================

class TestOverallQuality:
    def test_perfect_score(self):
        temporal = {'boiler': {'value': 0.9, 'flag': None}}
        magnitude = {'boiler': {'cv': 0.1, 'flag': None}}
        duration = {'boiler': {'median': 40, 'flag': None}}
        seasonal = {'boiler': {'ratio': 1.0, 'flag': None}}
        energy = {'flag': None}

        score = _compute_overall_quality(temporal, magnitude, duration, seasonal, energy)
        assert score == 1.0
        assert _score_to_tier(score) == 'excellent'

    def test_flagged_score_reduced(self):
        temporal = {'boiler': {'value': 0.3, 'flag': 'TEMPORAL_ANOMALY'}}
        magnitude = {'boiler': {'cv': 0.5, 'flag': 'MAGNITUDE_UNSTABLE'}}
        duration = {'boiler': {'median': 40, 'flag': None}}
        seasonal = {}
        energy = {'flag': None}

        score = _compute_overall_quality(temporal, magnitude, duration, seasonal, energy)
        assert score < 0.65  # Should not be 'good'


# ============================================================================
# Confidence Scoring
# ============================================================================

class TestTagQuality:
    def test_exact_highest(self):
        assert _score_tag_quality('EXACT-EXTENDED') == 1.0

    def test_loose_lowest(self):
        assert _score_tag_quality('LOOSE-SPIKE') == 0.4

    def test_noisy_reduces(self):
        assert _score_tag_quality('NOISY-EXACT-MEDIUM') < 1.0
        assert _score_tag_quality('NOISY-EXACT-MEDIUM') == pytest.approx(0.7)

    def test_partial_reduces_more(self):
        assert _score_tag_quality('PARTIAL-MEDIUM') < _score_tag_quality('NOISY-CLOSE-MEDIUM')

    def test_corrected_penalty(self):
        assert _score_tag_quality('EXACT-MEDIUM-CORRECTED') < _score_tag_quality('EXACT-MEDIUM')


class TestIterationScore:
    def test_iter0_best(self):
        assert _score_iteration(0) == 1.0

    def test_iter3_lowest(self):
        assert _score_iteration(3) == 0.4


class TestSeasonalFit:
    def test_ac_in_summer(self):
        assert _score_seasonal_fit({'on_start': '2024-07-15 14:00:00'}, 'regular_ac') == 1.0

    def test_ac_in_winter(self):
        assert _score_seasonal_fit({'on_start': '2024-01-15 14:00:00'}, 'regular_ac') == 0.3


class TestConfidenceTier:
    def test_high(self):
        assert _confidence_tier(0.80) == 'high'

    def test_medium(self):
        assert _confidence_tier(0.55) == 'medium'

    def test_low(self):
        assert _confidence_tier(0.20) == 'low'


class TestPopulationStats:
    def test_compute_type_statistics(self):
        acts = [_make_activation(magnitude=m) for m in [2000, 2100, 2200]]
        stats = _compute_type_statistics(acts)
        assert 'boiler' in stats
        assert stats['boiler']['mag_median'] == 2100.0

    def test_compute_dominant_phases(self):
        acts = [_make_activation(phase='w1')] * 5 + [_make_activation(phase='w2')] * 2
        phases = _compute_dominant_phases(acts)
        assert phases['boiler'] == 'w1'
