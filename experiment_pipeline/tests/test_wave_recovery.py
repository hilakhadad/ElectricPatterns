"""
Tests for the wave recovery module.

Wave patterns are modeled after house 221's 3-phase central AC:
- Sharp rise (~2000-3000W) followed by gradual monotonic decay over 10-30 minutes
- Compressor cycles that M1 rectangle matching misses because ON >> OFF magnitude
- Cross-phase synchronised patterns (W1 strong, W2/W3 weaker)
"""
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from disaggregation.wave_recovery.detection.wave_detector import (
    WavePattern,
    detect_wave_patterns,
    _find_wave_end,
)
from disaggregation.wave_recovery.matching.phase_matcher import find_cross_phase_waves
from disaggregation.wave_recovery.segmentation.wave_segmentor import extract_wave_power
from disaggregation.wave_recovery.segmentation.validator import validate_wave_extraction


# ============================================================================
# Helpers
# ============================================================================

@dataclass
class FakeConfig:
    """Minimal config for wave recovery tests."""
    wave_min_rise_watts: int = 500
    wave_min_duration_minutes: int = 3
    wave_max_duration_minutes: int = 45
    wave_monotonic_tolerance: float = 0.15
    wave_min_decay_fraction: float = 0.3


def _make_timestamps(n, start='2021-07-15 18:00'):
    """Create n consecutive 1-minute timestamps."""
    return pd.date_range(start, periods=n, freq='min')


def _make_wave_signal(baseline=200, peak=2500, duration=20, n_before=5, n_after=5):
    """
    Create a synthetic wave signal: flat baseline → sharp rise → exponential decay → baseline.

    Modeled after house 221 AC compressor: starts at ~2500W, decays over ~20 minutes.
    """
    n_total = n_before + 1 + duration + n_after
    ts = _make_timestamps(n_total)

    values = np.full(n_total, baseline, dtype=float)

    # Sharp rise at position n_before
    rise_idx = n_before

    # Decay: exponential from peak to baseline over `duration` minutes
    t = np.arange(duration + 1)  # 0, 1, ..., duration
    decay_rate = -np.log(0.05) / duration  # Decays to ~5% of (peak-baseline) at end
    wave = baseline + (peak - baseline) * np.exp(-decay_rate * t)

    # Insert wave into signal
    values[rise_idx:rise_idx + duration + 1] = wave

    # Make sure the last few points are truly back at baseline
    for i in range(rise_idx + duration + 1, n_total):
        values[i] = baseline

    return pd.Series(values, index=ts)


# ============================================================================
# Detection tests
# ============================================================================

class TestDetectWavePatterns:
    """Test wave detection on synthetic signals."""

    def test_detect_simple_wave(self):
        """A clear wave (sharp rise + exponential decay) should be detected."""
        signal = _make_wave_signal(baseline=200, peak=2500, duration=20)
        config = FakeConfig()

        waves = detect_wave_patterns(signal, 'w1', config)

        assert len(waves) == 1
        w = waves[0]
        assert w.phase == 'w1'
        assert w.duration_minutes >= 3
        assert w.peak_power > 2000
        assert w.baseline_power < 500

    def test_detect_no_wave_in_flat(self):
        """Flat baseline should produce zero waves."""
        ts = _make_timestamps(60)
        flat = pd.Series(300.0, index=ts)
        config = FakeConfig()

        waves = detect_wave_patterns(flat, 'w2', config)
        assert len(waves) == 0

    def test_detect_rejects_short_wave(self):
        """A wave shorter than min_duration should be rejected."""
        signal = _make_wave_signal(baseline=200, peak=2500, duration=2)  # Only 2-3 min
        config = FakeConfig(wave_min_duration_minutes=5)  # Require 5 min

        waves = detect_wave_patterns(signal, 'w1', config)
        assert len(waves) == 0

    def test_detect_rejects_small_rise(self):
        """A rise below wave_min_rise_watts should not trigger detection."""
        signal = _make_wave_signal(baseline=200, peak=500, duration=15)  # Only 300W rise
        config = FakeConfig(wave_min_rise_watts=500)

        waves = detect_wave_patterns(signal, 'w1', config)
        assert len(waves) == 0

    def test_detect_rejects_non_monotonic(self):
        """A zigzag signal should not be detected as a wave."""
        ts = _make_timestamps(40)
        values = np.full(40, 200.0)
        # Sharp rise at minute 5
        values[5] = 2500
        # Zigzag from minute 6 onward: oscillates between 2000 and 1000 (no decay trend)
        for i in range(6, 35):
            values[i] = 2000 if i % 2 == 0 else 1000
        values[35:] = 200
        signal = pd.Series(values, index=ts)

        config = FakeConfig(wave_monotonic_tolerance=0.15)
        waves = detect_wave_patterns(signal, 'w1', config)
        # The zigzag has ~50% increases which exceeds 15% tolerance
        assert len(waves) == 0

    def test_detect_two_waves(self):
        """Two separated waves should both be detected."""
        ts = _make_timestamps(80)
        values = np.full(80, 200.0)

        # Wave 1: minutes 5-25
        t1 = np.arange(21)
        rate1 = -np.log(0.05) / 20
        values[5:26] = 200 + 2000 * np.exp(-rate1 * t1)

        # Wave 2: minutes 45-60
        t2 = np.arange(16)
        rate2 = -np.log(0.05) / 15
        values[45:61] = 200 + 1800 * np.exp(-rate2 * t2)

        signal = pd.Series(values, index=ts)
        config = FakeConfig()

        waves = detect_wave_patterns(signal, 'w1', config)
        assert len(waves) == 2
        assert waves[0].start < waves[1].start

    def test_wave_profile_shape(self):
        """Wave profile should start high and decrease."""
        signal = _make_wave_signal(baseline=200, peak=2500, duration=20)
        config = FakeConfig()

        waves = detect_wave_patterns(signal, 'w1', config)
        assert len(waves) == 1

        profile = waves[0].wave_profile
        assert len(profile) >= 3
        # Profile should start > 0 and generally decrease
        assert profile[1] > profile[-1]  # Peak > end


# ============================================================================
# Cross-phase matching tests
# ============================================================================

class TestCrossPhaseMatching:
    """Test cross-phase wave detection."""

    def test_cross_phase_finds_weaker_wave(self):
        """A wave on W1 should help find a weaker wave on W2 in the same window."""
        # W1: strong wave (2500W peak)
        w1_signal = _make_wave_signal(baseline=200, peak=2500, duration=20)

        # W2: weaker wave at same time (800W peak — below normal threshold of 500W
        # but above relaxed threshold of 250W)
        w2_signal = _make_wave_signal(baseline=150, peak=600, duration=18)

        # W3: flat — no wave
        ts = _make_timestamps(len(w1_signal))
        w3_signal = pd.Series(100.0, index=ts)

        config = FakeConfig(wave_min_rise_watts=500)

        # Detect on all phases — W1 should find wave, W2 should not (below 500W)
        detected = {
            'w1': detect_wave_patterns(w1_signal, 'w1', config),
            'w2': detect_wave_patterns(w2_signal, 'w2', config),
            'w3': detect_wave_patterns(w3_signal, 'w3', config),
        }
        assert len(detected['w1']) == 1
        assert len(detected['w2']) == 0  # Below threshold

        # Cross-phase should find W2 wave using relaxed threshold (250W)
        remaining_by_phase = {'w1': w1_signal, 'w2': w2_signal, 'w3': w3_signal}
        extra = find_cross_phase_waves(detected, remaining_by_phase, config)

        assert len(extra['w2']) >= 1  # Found via W1 template

    def test_cross_phase_no_duplicates(self):
        """Already-detected waves should not be duplicated by cross-phase search."""
        signal = _make_wave_signal(baseline=200, peak=2500, duration=20)
        config = FakeConfig()

        detected = {
            'w1': detect_wave_patterns(signal, 'w1', config),
            'w2': [],
            'w3': [],
        }
        assert len(detected['w1']) == 1

        remaining = {'w1': signal, 'w2': signal.copy(), 'w3': signal.copy()}
        extra = find_cross_phase_waves(detected, remaining, config)

        # w2, w3 should get cross-phase waves, w1 should NOT get a duplicate
        assert len(extra['w1']) == 0  # Already detected, no duplicate


# ============================================================================
# Segmentation tests
# ============================================================================

class TestWaveSegmentation:
    """Test wave power extraction."""

    def test_extract_wave_power(self):
        """Extraction should remove wave-shaped power from remaining."""
        signal = _make_wave_signal(baseline=200, peak=2500, duration=20)
        config = FakeConfig()

        waves = detect_wave_patterns(signal, 'w1', config)
        assert len(waves) == 1

        extracted, updated = extract_wave_power(signal, waves[0])

        # Total extracted should be positive
        assert extracted.sum() > 0

        # Updated remaining should be <= original everywhere
        assert (updated <= signal + 0.01).all()

        # Updated remaining should be >= 0 (or at least >= baseline)
        assert (updated >= -0.01).all()

    def test_extract_clipped_to_remaining(self):
        """When remaining is less than wave profile, extraction clips correctly."""
        ts = _make_timestamps(30)
        # Remaining is mostly low (only 100W above baseline)
        values = np.full(30, 300.0)
        values[5] = 2500  # One sharp spike
        for i in range(6, 25):
            values[i] = 300 + max(0, 100 - (i - 5) * 5)  # Very little above baseline
        remaining = pd.Series(values, index=ts)

        wave = WavePattern(
            start=ts[5], peak_time=ts[6], end=ts[20],
            phase='w1', peak_power=2500, baseline_power=300,
            duration_minutes=15,
            wave_profile=np.linspace(2200, 0, 16),  # Large profile
        )

        extracted, updated = extract_wave_power(remaining, wave)

        # Should never go below baseline
        assert (updated >= 299.99).all()


# ============================================================================
# Validator tests
# ============================================================================

class TestWaveValidator:
    """Test extraction validation."""

    def test_validate_clean_extraction(self):
        """A properly extracted wave should validate."""
        signal = _make_wave_signal(baseline=200, peak=2500, duration=20)
        config = FakeConfig()

        waves = detect_wave_patterns(signal, 'w1', config)
        assert len(waves) == 1

        extracted, updated = extract_wave_power(signal, waves[0])
        valid, reason = validate_wave_extraction(signal, updated, extracted, waves[0])

        assert valid, f"Should be valid but got: {reason}"

    def test_validate_rejects_negative_remaining(self):
        """If remaining goes negative, validation should fail."""
        ts = _make_timestamps(10)
        before = pd.Series(100.0, index=ts)
        after = pd.Series(100.0, index=ts)
        after.iloc[3] = -50  # Artificially negative
        extracted = before - after

        wave = WavePattern(
            start=ts[1], peak_time=ts[2], end=ts[7],
            phase='w1', peak_power=500, baseline_power=100,
            duration_minutes=6,
            wave_profile=np.zeros(7),
        )

        valid, reason = validate_wave_extraction(before, after, extracted, wave)
        assert not valid
        assert 'Negative' in reason

    def test_validate_rejects_zero_extraction(self):
        """If nothing is extracted, validation should fail."""
        ts = _make_timestamps(10)
        before = pd.Series(100.0, index=ts)
        after = before.copy()
        extracted = pd.Series(0.0, index=ts)

        wave = WavePattern(
            start=ts[1], peak_time=ts[2], end=ts[7],
            phase='w1', peak_power=500, baseline_power=100,
            duration_minutes=6,
            wave_profile=np.zeros(7),
        )

        valid, reason = validate_wave_extraction(before, after, extracted, wave)
        assert not valid
        # Can be rejected for either "No power" or "zero-gaps" — both mean nothing was extracted
        assert 'No power' in reason or 'zero-gap' in reason


# ============================================================================
# Match record format tests
# ============================================================================

class TestMatchRecordFormat:
    """Wave matches must be compatible with M1 match format."""

    def test_match_record_has_required_columns(self):
        """Generated match records should have all M1 columns."""
        from disaggregation.wave_recovery.pipeline.wave_recovery_step import _wave_to_match_record

        ts = _make_timestamps(30)
        wave = WavePattern(
            start=ts[5], peak_time=ts[6], end=ts[20],
            phase='w1', peak_power=2500, baseline_power=200,
            duration_minutes=15,
            wave_profile=np.linspace(2300, 0, 16),
        )

        record = _wave_to_match_record(wave, extracted_power=15000.0)

        required_columns = [
            'on_event_id', 'off_event_id',
            'on_start', 'on_end', 'off_start', 'off_end',
            'duration', 'on_magnitude', 'off_magnitude',
            'correction', 'tag', 'phase',
        ]
        for col in required_columns:
            assert col in record, f"Missing column: {col}"

        assert record['phase'] == 'w1'
        assert record['tag'].startswith('WAVE-')
        assert record['duration'] == 15.0
        assert record['on_magnitude'] > 0
        assert record['correction'] == 0

    def test_wave_tag_mapping(self):
        """Duration-based tags should follow the convention."""
        from disaggregation.wave_recovery.pipeline.wave_recovery_step import _wave_tag

        assert _wave_tag(1) == 'WAVE-SPIKE'
        assert _wave_tag(2) == 'WAVE-SPIKE'
        assert _wave_tag(4) == 'WAVE-QUICK'
        assert _wave_tag(15) == 'WAVE-MEDIUM'
        assert _wave_tag(30) == 'WAVE-EXTENDED'


# ============================================================================
# M2 integration test
# ============================================================================

class TestM2Integration:
    """Test that session_grouper picks up wave matches from run_post/."""

    def test_load_all_matches_includes_wave_matches(self, tmp_path):
        """load_all_matches should find matches in run_post/."""
        from identification.session_grouper import load_all_matches

        house_id = '221'
        threshold_schedule = [2000, 1500, 1100, 800]

        # Create a minimal run_0 with one rectangle match
        run0_matches = tmp_path / "run_0" / f"house_{house_id}" / "matches"
        run0_matches.mkdir(parents=True)
        rect_match = pd.DataFrame([{
            'on_event_id': 'rect_001', 'off_event_id': 'rect_002',
            'on_start': pd.Timestamp('2021-07-15 18:00'),
            'on_end': pd.Timestamp('2021-07-15 18:01'),
            'off_start': pd.Timestamp('2021-07-15 18:20'),
            'off_end': pd.Timestamp('2021-07-15 18:21'),
            'duration': 20.0, 'on_magnitude': 2000.0,
            'off_magnitude': -1800.0, 'correction': 0,
            'tag': 'EXACT-MEDIUM', 'phase': 'w1',
        }])
        rect_match.to_pickle(run0_matches / f"matches_{house_id}_07_2021.pkl")

        # Create run_post with one wave match
        post_matches = tmp_path / "run_post" / f"house_{house_id}" / "matches"
        post_matches.mkdir(parents=True)
        wave_match = pd.DataFrame([{
            'on_event_id': 'wave_abc', 'off_event_id': 'wave_def',
            'on_start': pd.Timestamp('2021-07-15 19:00'),
            'on_end': pd.Timestamp('2021-07-15 19:01'),
            'off_start': pd.Timestamp('2021-07-15 19:14'),
            'off_end': pd.Timestamp('2021-07-15 19:15'),
            'duration': 15.0, 'on_magnitude': 2300.0,
            'off_magnitude': -1150.0, 'correction': 0,
            'tag': 'WAVE-MEDIUM', 'phase': 'w2',
            'iteration': 4, 'threshold': 0,
        }])
        wave_match.to_pickle(post_matches / f"matches_{house_id}_07_2021.pkl")

        # Load all
        combined = load_all_matches(tmp_path, house_id, threshold_schedule)

        assert len(combined) == 2
        assert (combined['tag'] == 'EXACT-MEDIUM').sum() == 1
        assert (combined['tag'] == 'WAVE-MEDIUM').sum() == 1
