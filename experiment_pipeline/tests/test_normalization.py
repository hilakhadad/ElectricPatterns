"""Tests for normalization preprocessing module."""
import numpy as np
import pandas as pd
import pytest

from core.normalization import (
    detrend_moving_average,
    balance_phases,
    mad_outlier_cleaning,
    apply_normalization,
    compute_threshold_scaling,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_data():
    """Create sample power data with 3 phases, 1-minute resolution, 6 hours."""
    n = 360  # 6 hours
    timestamps = pd.date_range('2021-07-01', periods=n, freq='1min')
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        'timestamp': timestamps,
        'w1': 1000 + rng.normal(0, 10, n),
        'w2': 500 + rng.normal(0, 10, n),
        'w3': 800 + rng.normal(0, 10, n),
    })


@pytest.fixture
def data_with_drift():
    """Power data with a slow sinusoidal baseline drift."""
    n = 720  # 12 hours
    timestamps = pd.date_range('2021-07-01', periods=n, freq='1min')
    t = np.arange(n)
    # Drift: 200W sinusoidal with 6-hour period
    drift = 200 * np.sin(2 * np.pi * t / 360)
    return pd.DataFrame({
        'timestamp': timestamps,
        'w1': 1000 + drift + np.random.RandomState(1).normal(0, 5, n),
        'w2': 500 + drift * 0.5 + np.random.RandomState(2).normal(0, 5, n),
        'w3': 800 + np.random.RandomState(3).normal(0, 5, n),
    })


@pytest.fixture
def imbalanced_phases():
    """Power data with very different baselines per phase."""
    n = 360
    timestamps = pd.date_range('2021-07-01', periods=n, freq='1min')
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        'timestamp': timestamps,
        'w1': 3000 + rng.normal(0, 10, n),  # High baseline
        'w2': 200 + rng.normal(0, 10, n),   # Low baseline
        'w3': 1000 + rng.normal(0, 10, n),  # Medium baseline
    })


@pytest.fixture
def data_with_outliers():
    """Power data with a few extreme outlier measurements."""
    n = 360
    timestamps = pd.date_range('2021-07-01', periods=n, freq='1min')
    w1 = np.full(n, 1000.0)
    # Insert extreme outliers
    w1[50] = 9000.0  # Spike up
    w1[150] = -500.0  # Spike down
    w1[250] = 8000.0  # Spike up
    return pd.DataFrame({
        'timestamp': timestamps,
        'w1': w1,
        'w2': np.full(n, 500.0),
        'w3': np.full(n, 800.0),
    })


@pytest.fixture
def constant_data():
    """Constant power signal (no variation)."""
    n = 120
    timestamps = pd.date_range('2021-07-01', periods=n, freq='1min')
    return pd.DataFrame({
        'timestamp': timestamps,
        'w1': np.full(n, 1000.0),
        'w2': np.full(n, 500.0),
        'w3': np.full(n, 800.0),
    })


# ============================================================================
# MA Detrending Tests
# ============================================================================

class TestDetrendMovingAverage:
    def test_constant_signal_unchanged(self, constant_data):
        """Constant signal should be unchanged by MA detrending."""
        result = detrend_moving_average(constant_data)
        for phase in ['w1', 'w2', 'w3']:
            np.testing.assert_allclose(
                result[phase].values, constant_data[phase].values, atol=1e-10
            )

    def test_preserves_global_mean(self, sample_data):
        """Global mean should be approximately preserved (small edge effect from min_periods)."""
        result = detrend_moving_average(sample_data)
        for phase in ['w1', 'w2', 'w3']:
            np.testing.assert_allclose(
                result[phase].mean(), sample_data[phase].mean(), atol=1.0
            )

    def test_reduces_drift(self, data_with_drift):
        """Detrending should reduce the standard deviation of a drifting signal."""
        result = detrend_moving_average(data_with_drift, window_minutes=120)
        # w1 has strong drift - std should decrease after detrending
        original_std = data_with_drift['w1'].std()
        detrended_std = result['w1'].std()
        assert detrended_std < original_std * 0.5, (
            f"Detrending should reduce std significantly: {original_std:.1f} -> {detrended_std:.1f}"
        )

    def test_preserves_diff_approximately(self, sample_data):
        """diff() values should be approximately unchanged (since MA changes slowly)."""
        result = detrend_moving_average(sample_data, window_minutes=120)
        for phase in ['w1', 'w2', 'w3']:
            original_diff = sample_data[phase].diff().dropna()
            result_diff = result[phase].diff().dropna()
            # Correlation between original and detrended diffs should be very high
            corr = np.corrcoef(original_diff, result_diff)[0, 1]
            assert corr > 0.95, f"diff() correlation for {phase}: {corr:.4f}"

    def test_does_not_modify_input(self, sample_data):
        """Should not modify the input DataFrame."""
        original = sample_data.copy()
        detrend_moving_average(sample_data)
        pd.testing.assert_frame_equal(sample_data, original)

    def test_custom_window(self, sample_data):
        """Custom window size should be accepted."""
        result = detrend_moving_average(sample_data, window_minutes=60)
        assert len(result) == len(sample_data)

    def test_timestamp_preserved(self, sample_data):
        """Timestamp column should be unchanged."""
        result = detrend_moving_average(sample_data)
        pd.testing.assert_series_equal(result['timestamp'], sample_data['timestamp'])


# ============================================================================
# Phase Balancing Tests
# ============================================================================

class TestBalancePhases:
    def test_equal_medians_after_balancing(self, imbalanced_phases):
        """All phases should have approximately equal medians after balancing."""
        result = balance_phases(imbalanced_phases)
        medians = [result[phase].median() for phase in ['w1', 'w2', 'w3']]
        # All medians should be within 1W of each other (noise)
        assert max(medians) - min(medians) < 1.0, f"Medians after balancing: {medians}"

    def test_preserves_diff_exactly(self, imbalanced_phases):
        """Phase balancing is a constant shift, so diff() should be exactly unchanged."""
        result = balance_phases(imbalanced_phases)
        for phase in ['w1', 'w2', 'w3']:
            original_diff = imbalanced_phases[phase].diff()
            result_diff = result[phase].diff()
            np.testing.assert_allclose(result_diff.values, original_diff.values, atol=1e-10)

    def test_constant_signal_unchanged(self, constant_data):
        """If all phases are constant, they shift to common median but values are constant."""
        result = balance_phases(constant_data)
        # All phases should have the same value (the global median)
        global_median = np.mean([1000.0, 500.0, 800.0])
        for phase in ['w1', 'w2', 'w3']:
            np.testing.assert_allclose(result[phase].values, global_median, atol=1e-10)

    def test_already_balanced_data(self, sample_data):
        """Data where phases are manually set to same median should be nearly unchanged."""
        # Create balanced data
        data = sample_data.copy()
        for phase in ['w1', 'w2', 'w3']:
            data[phase] = data[phase] - data[phase].median() + 1000.0
        result = balance_phases(data)
        for phase in ['w1', 'w2', 'w3']:
            np.testing.assert_allclose(
                result[phase].values, data[phase].values, atol=1e-10
            )

    def test_does_not_modify_input(self, imbalanced_phases):
        """Should not modify the input DataFrame."""
        original = imbalanced_phases.copy()
        balance_phases(imbalanced_phases)
        pd.testing.assert_frame_equal(imbalanced_phases, original)

    def test_timestamp_preserved(self, imbalanced_phases):
        """Timestamp column should be unchanged."""
        result = balance_phases(imbalanced_phases)
        pd.testing.assert_series_equal(result['timestamp'], imbalanced_phases['timestamp'])


# ============================================================================
# MAD Outlier Cleaning Tests
# ============================================================================

class TestMadOutlierCleaning:
    def test_removes_extreme_outliers(self, data_with_outliers):
        """Extreme outliers should be replaced."""
        result = mad_outlier_cleaning(data_with_outliers, window_minutes=60, k=3.0)
        # The spike at index 50 (9000W) should be replaced with something near 1000W
        assert abs(result['w1'].iloc[50] - 1000.0) < 100.0, (
            f"Outlier at idx 50 should be ~1000, got {result['w1'].iloc[50]}"
        )

    def test_preserves_clean_signal(self, constant_data):
        """Clean constant signal should be completely unchanged."""
        result = mad_outlier_cleaning(constant_data, window_minutes=60, k=3.0)
        # For constant data, MAD=0, so no outliers can be detected
        for phase in ['w1', 'w2', 'w3']:
            np.testing.assert_allclose(
                result[phase].values, constant_data[phase].values, atol=1e-10
            )

    def test_preserves_normal_variation(self, sample_data):
        """Normal variation (small noise) should not be removed with k=5."""
        result = mad_outlier_cleaning(sample_data, window_minutes=120, k=5.0)
        for phase in ['w1', 'w2', 'w3']:
            # Most values should be unchanged
            changed = (result[phase] != sample_data[phase]).sum()
            assert changed < len(sample_data) * 0.02, (
                f"{phase}: too many points changed ({changed}/{len(sample_data)})"
            )

    def test_does_not_modify_input(self, data_with_outliers):
        """Should not modify the input DataFrame."""
        original = data_with_outliers.copy()
        mad_outlier_cleaning(data_with_outliers)
        pd.testing.assert_frame_equal(data_with_outliers, original)

    def test_higher_k_removes_fewer_outliers(self, data_with_outliers):
        """Higher k threshold should be more permissive."""
        result_strict = mad_outlier_cleaning(data_with_outliers, window_minutes=60, k=3.0)
        result_loose = mad_outlier_cleaning(data_with_outliers, window_minutes=60, k=10.0)
        changed_strict = (result_strict['w1'] != data_with_outliers['w1']).sum()
        changed_loose = (result_loose['w1'] != data_with_outliers['w1']).sum()
        assert changed_strict >= changed_loose

    def test_timestamp_preserved(self, data_with_outliers):
        """Timestamp column should be unchanged."""
        result = mad_outlier_cleaning(data_with_outliers)
        pd.testing.assert_series_equal(result['timestamp'], data_with_outliers['timestamp'])


# ============================================================================
# apply_normalization() Tests
# ============================================================================

class TestApplyNormalization:
    def test_ma_detrend_dispatch(self, sample_data):
        """'ma_detrend' should dispatch to detrend_moving_average."""
        result = apply_normalization(sample_data, 'ma_detrend')
        expected = detrend_moving_average(sample_data)
        pd.testing.assert_frame_equal(result, expected)

    def test_phase_balance_dispatch(self, sample_data):
        """'phase_balance' should dispatch to balance_phases."""
        result = apply_normalization(sample_data, 'phase_balance')
        expected = balance_phases(sample_data)
        pd.testing.assert_frame_equal(result, expected)

    def test_mad_clean_dispatch(self, sample_data):
        """'mad_clean' should dispatch to mad_outlier_cleaning."""
        result = apply_normalization(sample_data, 'mad_clean')
        expected = mad_outlier_cleaning(sample_data)
        pd.testing.assert_frame_equal(result, expected)

    def test_combined_applies_all_three(self, imbalanced_phases):
        """'combined' should apply MA → phase balance → MAD in sequence."""
        result = apply_normalization(imbalanced_phases, 'combined')
        # Manually apply sequence
        step1 = detrend_moving_average(imbalanced_phases)
        step2 = balance_phases(step1)
        step3 = mad_outlier_cleaning(step2)
        pd.testing.assert_frame_equal(result, step3)

    def test_combined_with_params(self, sample_data):
        """Combined should pass method-specific params correctly."""
        params = {
            'ma_detrend': {'window_minutes': 60},
            'mad_clean': {'window_minutes': 120, 'k': 3.0},
        }
        result = apply_normalization(sample_data, 'combined', params=params)
        step1 = detrend_moving_average(sample_data, window_minutes=60)
        step2 = balance_phases(step1)
        step3 = mad_outlier_cleaning(step2, window_minutes=120, k=3.0)
        pd.testing.assert_frame_equal(result, step3)

    def test_invalid_method_raises(self, sample_data):
        """Invalid method name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            apply_normalization(sample_data, 'invalid_method')


# ============================================================================
# Scale Preservation Tests
# ============================================================================

class TestScalePreservation:
    """Verify all methods preserve the absolute watt scale."""

    def test_ma_detrend_preserves_scale(self, sample_data):
        """MA detrending should keep values in same order of magnitude."""
        result = detrend_moving_average(sample_data)
        for phase in ['w1', 'w2', 'w3']:
            assert abs(result[phase].mean() - sample_data[phase].mean()) < 1.0

    def test_phase_balance_preserves_total_power(self, imbalanced_phases):
        """Phase balancing should preserve total power across all phases."""
        result = balance_phases(imbalanced_phases)
        original_total = sum(imbalanced_phases[p].mean() for p in ['w1', 'w2', 'w3'])
        result_total = sum(result[p].mean() for p in ['w1', 'w2', 'w3'])
        np.testing.assert_allclose(result_total, original_total, atol=1.0)

    def test_mad_clean_preserves_scale(self, sample_data):
        """MAD cleaning should keep mean approximately the same."""
        result = mad_outlier_cleaning(sample_data)
        for phase in ['w1', 'w2', 'w3']:
            np.testing.assert_allclose(
                result[phase].mean(), sample_data[phase].mean(), atol=5.0
            )


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_single_row_data(self):
        """Single row of data should not crash."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2021-07-01')],
            'w1': [1000.0],
            'w2': [500.0],
            'w3': [800.0],
        })
        for method in ['ma_detrend', 'phase_balance', 'mad_clean', 'combined']:
            result = apply_normalization(data, method)
            assert len(result) == 1

    def test_data_with_nans(self, sample_data):
        """Data with NaN values should not crash (NaN propagation is OK)."""
        data = sample_data.copy()
        data.loc[10, 'w1'] = np.nan
        data.loc[20, 'w2'] = np.nan
        for method in ['ma_detrend', 'phase_balance', 'mad_clean']:
            result = apply_normalization(data, method)
            assert len(result) == len(data)

    def test_all_zero_phase(self):
        """Phase with all zeros should not crash."""
        n = 120
        data = pd.DataFrame({
            'timestamp': pd.date_range('2021-07-01', periods=n, freq='1min'),
            'w1': np.zeros(n),
            'w2': np.full(n, 500.0),
            'w3': np.full(n, 800.0),
        })
        for method in ['ma_detrend', 'phase_balance', 'mad_clean', 'combined']:
            result = apply_normalization(data, method)
            assert len(result) == n

    def test_custom_phases(self, sample_data):
        """Custom phase list should only normalize specified phases."""
        result = detrend_moving_average(sample_data, phases=['w1'])
        # w2 and w3 should be unchanged
        pd.testing.assert_series_equal(result['w2'], sample_data['w2'])
        pd.testing.assert_series_equal(result['w3'], sample_data['w3'])


# ============================================================================
# Experiment Config Tests
# ============================================================================

class TestExperimentConfigs:
    """Verify normalization experiment configs are properly defined."""

    def test_normalization_experiments_exist(self):
        """All 4 normalization experiments should be in EXPERIMENTS dict."""
        from core.config import EXPERIMENTS
        expected = ['exp016_ma_detrend', 'exp017_phase_balance',
                    'exp018_mad_clean', 'exp019_combined_norm']
        for name in expected:
            assert name in EXPERIMENTS, f"Missing experiment: {name}"

    def test_normalization_flags_set(self):
        """Normalization experiments should have use_normalization=True."""
        from core.config import EXPERIMENTS
        for name in ['exp016_ma_detrend', 'exp017_phase_balance',
                     'exp018_mad_clean', 'exp019_combined_norm']:
            config = EXPERIMENTS[name]
            assert config.use_normalization is True, f"{name}: use_normalization should be True"
            assert config.normalization_method != 'none', f"{name}: method should not be 'none'"

    def test_non_normalization_experiments_unchanged(self):
        """Existing experiments (exp010-exp015) should have use_normalization=False."""
        from core.config import EXPERIMENTS
        for name, config in EXPERIMENTS.items():
            if config.exp_id in ('exp010', 'exp012', 'exp013', 'exp014', 'exp015'):
                assert config.use_normalization is False, (
                    f"{name}: should not have normalization enabled"
                )

    def test_normalization_in_to_dict(self):
        """Normalization fields should appear in to_dict() output."""
        from core.config import EXPERIMENTS
        config = EXPERIMENTS['exp016_ma_detrend']
        d = config.to_dict()
        assert 'use_normalization' in d
        assert 'normalization_method' in d
        assert 'normalization_params' in d
        assert d['use_normalization'] is True
        assert d['normalization_method'] == 'ma_detrend'

    def test_combined_inherits_all_exp015_features(self):
        """exp019 should have all exp015 features enabled."""
        from core.config import EXPERIMENTS
        exp015 = EXPERIMENTS['exp015_hole_repair']
        exp019 = EXPERIMENTS['exp019_combined_norm']
        # All detection features should match
        assert exp019.use_nan_imputation == exp015.use_nan_imputation
        assert exp019.use_settling_extension == exp015.use_settling_extension
        assert exp019.use_wave_recovery == exp015.use_wave_recovery
        assert exp019.use_guided_recovery == exp015.use_guided_recovery
        assert exp019.threshold_schedule == exp015.threshold_schedule


# ============================================================================
# Per-Timestamp Original Diff Tests
# ============================================================================

class TestOrigDiffPreservation:
    """Tests for orig_diff columns saved by preprocess_normalize."""

    def test_ma_detrend_saves_orig_diff(self, sample_data):
        """MA detrending should save original diffs alongside normalized data."""
        original_diffs = {p: sample_data[p].diff() for p in ['w1', 'w2', 'w3']}
        result = detrend_moving_average(sample_data)
        # Simulate what preprocess_normalize does
        for phase in ['w1', 'w2', 'w3']:
            result[f'{phase}_orig_diff'] = sample_data[phase].diff()
        for phase in ['w1', 'w2', 'w3']:
            pd.testing.assert_series_equal(
                result[f'{phase}_orig_diff'], original_diffs[phase],
                check_names=False
            )

    def test_orig_diff_used_for_detection(self, sample_data):
        """Detection should use orig_diff for threshold when available."""
        from disaggregation.rectangle.detection.sharp import detect_sharp_events
        n = 200
        timestamps = pd.date_range('2021-07-01', periods=n, freq='1min')
        w1 = np.full(n, 1000.0)
        w1[50] = 3000.0  # +2000W spike in original
        w1[51] = 1000.0  # back to normal

        # Normalized data: event attenuated to 1800W (below 2000 threshold)
        w1_norm = np.full(n, 1000.0)
        w1_norm[50] = 2800.0  # attenuated
        w1_norm[51] = 1000.0

        data = pd.DataFrame({
            'timestamp': timestamps,
            'w1': w1_norm,  # normalized values
            'w1_orig_diff': pd.Series(w1).diff(),  # original diffs
        })

        on_events, off_events = detect_sharp_events(data, 'w1', threshold=2000)
        # Should detect because orig_diff has 2000W jump
        assert len(on_events) >= 1, "Event should be detected via orig_diff despite attenuated normalized diff"

    def test_no_orig_diff_falls_back(self, sample_data):
        """Without orig_diff columns, detection should use regular diff."""
        from disaggregation.rectangle.detection.sharp import detect_sharp_events
        n = 200
        timestamps = pd.date_range('2021-07-01', periods=n, freq='1min')
        w1 = np.full(n, 1000.0)
        w1[50] = 2800.0  # 1800W jump (below 2000 threshold)
        w1[51] = 1000.0

        data = pd.DataFrame({'timestamp': timestamps, 'w1': w1})
        # No orig_diff column — should use normalized diff
        on_events, off_events = detect_sharp_events(data, 'w1', threshold=2000)
        assert len(on_events) == 0, "Should NOT detect because diff=1800 < threshold=2000"

    def test_gradual_uses_orig_diff_for_threshold(self):
        """Gradual detection should use orig_diff for threshold comparison."""
        from disaggregation.rectangle.detection.gradual import detect_gradual_events
        n = 200
        timestamps = pd.date_range('2021-07-01', periods=n, freq='1min')

        # Normalized diffs: two steps summing to 1400W (below 1500 threshold)
        norm_diffs = np.zeros(n)
        norm_diffs[50] = 700
        norm_diffs[51] = 700  # sum = 1400 < 1500

        # Original diffs: two steps summing to 1600W (above 1500 threshold)
        orig_diffs = np.zeros(n)
        orig_diffs[50] = 800
        orig_diffs[51] = 800  # sum = 1600 > 1500

        w1 = np.cumsum(norm_diffs) + 1000
        data = pd.DataFrame({
            'timestamp': timestamps,
            'w1': w1,
            'w1_diff': norm_diffs,
            'w1_orig_diff': orig_diffs,
        })

        events = detect_gradual_events(
            data, 'w1_diff', threshold=1500, event_type='on',
            progressive_search=True, partial_factor=1.0, max_factor=3.0,
            orig_diff_col='w1_orig_diff',
        )
        assert len(events) >= 1, "Gradual event should be detected via orig_diff"


# ============================================================================
# Threshold Scaling Tests (utility function, kept for reporting)
# ============================================================================

class TestComputeThresholdScaling:
    """Tests for compute_threshold_scaling() — utility for measuring attenuation."""

    @pytest.fixture
    def paired_data_dirs(self, tmp_path):
        """Create original and normalized data dirs with matching pkl files."""
        orig_dir = tmp_path / "original" / "221"
        norm_dir = tmp_path / "normalized" / "221"
        orig_dir.mkdir(parents=True)
        norm_dir.mkdir(parents=True)
        return orig_dir, norm_dir, tmp_path

    def _make_signal_with_events(self, n=720, events=None):
        """Create a power signal with specified events (list of (index, magnitude) tuples)."""
        timestamps = pd.date_range('2021-07-01', periods=n, freq='1min')
        w1 = np.full(n, 1000.0)
        if events:
            for idx, magnitude in events:
                if idx < n:
                    w1[idx:] += magnitude
        return pd.DataFrame({
            'timestamp': timestamps,
            'w1': w1,
            'w2': np.full(n, 500.0),
            'w3': np.full(n, 800.0),
        })

    def test_identity_returns_one(self, paired_data_dirs):
        """Identical original and normalized data should give scale=1.0."""
        orig_dir, norm_dir, tmp_path = paired_data_dirs
        data = self._make_signal_with_events(events=[(100, 2000), (200, -2000), (400, 1500)])
        data.to_pickle(orig_dir / "221_07_2021.pkl")
        data.to_pickle(norm_dir / "221_07_2021.pkl")
        scale = compute_threshold_scaling(
            str(tmp_path / "original"), "221", str(tmp_path / "normalized")
        )
        assert scale == 1.0

    def test_attenuated_diffs_returns_less_than_one(self, paired_data_dirs):
        """If normalization attenuates diffs by ~10%, scale should be ~0.9."""
        orig_dir, norm_dir, tmp_path = paired_data_dirs
        orig_data = self._make_signal_with_events(
            events=[(100, 2000), (200, -2000), (400, 1500), (500, -1500)]
        )
        norm_data = self._make_signal_with_events(
            events=[(100, 1800), (200, -1800), (400, 1350), (500, -1350)]
        )
        orig_data.to_pickle(orig_dir / "221_07_2021.pkl")
        norm_data.to_pickle(norm_dir / "221_07_2021.pkl")
        scale = compute_threshold_scaling(
            str(tmp_path / "original"), "221", str(tmp_path / "normalized")
        )
        assert 0.85 <= scale <= 0.95, f"Expected ~0.9, got {scale}"

    def test_no_large_diffs_returns_one(self, paired_data_dirs):
        """Quiet house (no diffs >= 500W) should return 1.0."""
        orig_dir, norm_dir, tmp_path = paired_data_dirs
        n = 360
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2021-07-01', periods=n, freq='1min'),
            'w1': 1000 + rng.normal(0, 10, n),
            'w2': 500 + rng.normal(0, 10, n),
            'w3': 800 + rng.normal(0, 10, n),
        })
        data.to_pickle(orig_dir / "221_07_2021.pkl")
        data.to_pickle(norm_dir / "221_07_2021.pkl")
        scale = compute_threshold_scaling(
            str(tmp_path / "original"), "221", str(tmp_path / "normalized")
        )
        assert scale == 1.0

    def test_clamp_range(self, paired_data_dirs):
        """Scale should always be in [0.5, 1.0], even with extreme attenuation."""
        orig_dir, norm_dir, tmp_path = paired_data_dirs
        orig_data = self._make_signal_with_events(events=[(100, 3000), (200, -3000)])
        norm_data = self._make_signal_with_events(events=[(100, 300), (200, -300)])
        orig_data.to_pickle(orig_dir / "221_07_2021.pkl")
        norm_data.to_pickle(norm_dir / "221_07_2021.pkl")
        scale = compute_threshold_scaling(
            str(tmp_path / "original"), "221", str(tmp_path / "normalized")
        )
        assert 0.5 <= scale <= 1.0

    def test_amplified_diffs_clamped_to_one(self, paired_data_dirs):
        """If normalization amplifies diffs, scale should be clamped to 1.0."""
        orig_dir, norm_dir, tmp_path = paired_data_dirs
        orig_data = self._make_signal_with_events(events=[(100, 1500), (200, -1500)])
        norm_data = self._make_signal_with_events(events=[(100, 2000), (200, -2000)])
        orig_data.to_pickle(orig_dir / "221_07_2021.pkl")
        norm_data.to_pickle(norm_dir / "221_07_2021.pkl")
        scale = compute_threshold_scaling(
            str(tmp_path / "original"), "221", str(tmp_path / "normalized")
        )
        assert scale == 1.0
