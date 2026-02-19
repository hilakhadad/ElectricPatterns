"""
Unit tests for core/nan_imputation.py — tiered NaN gap filling.

Tests:
  - Short gap (<=5 min) → forward fill
  - Medium gap (6-60 min) → linear interpolation
  - Long gap (>60 min) → stays NaN
  - No NaN → unchanged
  - Mixed gaps in same column
  - Multi-phase imputation
  - Does not modify original DataFrame
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from core.nan_imputation import impute_nan_gaps


def _make_ts(n: int, start='2020-01-01'):
    """Create n timestamps at 1-minute intervals."""
    return pd.date_range(start, periods=n, freq='min')


# ============================================================================
# Basic tier tests
# ============================================================================

class TestShortGapFfill:
    """Gaps <= 5 minutes should be forward-filled."""

    def test_3min_gap_filled(self):
        """A 3-minute NaN gap should be completely filled via ffill."""
        n = 20
        ts = _make_ts(n)
        vals = np.full(n, 500.0)
        vals[5:8] = np.nan  # 3-minute gap at indices 5,6,7

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1'])

        assert result['w1'].isna().sum() == 0
        # ffill should carry forward the value from index 4 (500.0)
        np.testing.assert_array_equal(result['w1'].values[5:8], [500.0, 500.0, 500.0])

    def test_5min_gap_filled(self):
        """Exactly 5-minute gap — boundary of ffill tier."""
        n = 20
        ts = _make_ts(n)
        vals = np.full(n, 300.0)
        vals[3:8] = np.nan  # 5-minute gap

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1'])

        assert result['w1'].isna().sum() == 0
        np.testing.assert_array_equal(result['w1'].values[3:8], [300.0] * 5)

    def test_1min_gap_filled(self):
        """Single NaN minute should be ffilled."""
        n = 10
        ts = _make_ts(n)
        vals = np.full(n, 1000.0)
        vals[4] = np.nan

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1'])

        assert result['w1'].isna().sum() == 0
        assert result['w1'].iloc[4] == 1000.0


class TestMediumGapInterpolation:
    """Gaps 6-60 minutes should be linearly interpolated."""

    def test_10min_gap_interpolated(self):
        """A 10-minute gap between 100 and 200 should interpolate linearly."""
        n = 30
        ts = _make_ts(n)
        vals = np.full(n, 100.0)
        vals[20:] = 200.0
        vals[10:20] = np.nan  # 10-minute gap between 100 and 200

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1'])

        assert result['w1'].isna().sum() == 0
        # Values should increase linearly from ~100 to ~200
        filled = result['w1'].values[10:20]
        assert filled[0] > 99.0
        assert filled[-1] < 201.0
        # Should be monotonically non-decreasing
        assert all(filled[i] <= filled[i + 1] for i in range(len(filled) - 1))

    def test_6min_gap_interpolated_not_ffilled(self):
        """A 6-minute gap should use interpolation, not ffill."""
        n = 20
        ts = _make_ts(n)
        vals = np.full(n, 100.0)
        vals[14:] = 400.0
        vals[8:14] = np.nan  # 6-minute gap between 100 and 400

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1'])

        assert result['w1'].isna().sum() == 0
        # If it were ffill, all values would be 100. Interpolation should give intermediate values.
        filled = result['w1'].values[8:14]
        assert any(v > 100.0 for v in filled), "Interpolation should produce values between 100 and 400"

    def test_60min_gap_boundary(self):
        """Exactly 60-minute gap — boundary of interpolation tier."""
        n = 80
        ts = _make_ts(n)
        vals = np.full(n, 500.0)
        vals[10:70] = np.nan  # exactly 60-minute gap

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1'])

        # 60 minutes should be filled (interp_limit=60 is inclusive)
        assert result['w1'].isna().sum() == 0


class TestLongGapRemains:
    """Gaps > 60 minutes should remain NaN."""

    def test_61min_gap_stays_nan(self):
        """A 61-minute gap should NOT be filled."""
        n = 80
        ts = _make_ts(n)
        vals = np.full(n, 500.0)
        vals[5:66] = np.nan  # 61-minute gap

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1'])

        assert result['w1'].isna().sum() == 61

    def test_120min_gap_stays_nan(self):
        """A 120-minute gap should remain entirely NaN."""
        n = 150
        ts = _make_ts(n)
        vals = np.full(n, 800.0)
        vals[10:130] = np.nan  # 120 minutes

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1'])

        assert result['w1'].isna().sum() == 120


# ============================================================================
# Edge cases
# ============================================================================

class TestNoNaN:
    """Data without NaN should be returned unchanged."""

    def test_no_nan_unchanged(self):
        n = 20
        ts = _make_ts(n)
        vals = np.arange(n, dtype=float) * 100

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1', 'w2', 'w3'])

        pd.testing.assert_frame_equal(result, df)


class TestDoesNotModifyOriginal:
    """impute_nan_gaps must not modify the input DataFrame."""

    def test_original_unchanged(self):
        n = 20
        ts = _make_ts(n)
        vals = np.full(n, 500.0)
        vals[5:8] = np.nan

        df = pd.DataFrame({'timestamp': ts, 'w1': vals.copy(), 'w2': vals.copy(), 'w3': vals.copy()})
        original_nan_count = df['w1'].isna().sum()

        _ = impute_nan_gaps(df, phase_cols=['w1'])

        assert df['w1'].isna().sum() == original_nan_count, "Original DataFrame was modified!"


class TestMixedGaps:
    """Multiple gaps of different sizes in the same column."""

    def test_mixed_short_medium_long(self):
        n = 200
        ts = _make_ts(n)
        vals = np.full(n, 1000.0)

        # Short gap: 3 min (indices 10-12)
        vals[10:13] = np.nan
        # Medium gap: 20 min (indices 50-69)
        vals[50:70] = np.nan
        # Long gap: 80 min (indices 100-179)
        vals[100:180] = np.nan

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1'])

        # Short gap should be filled
        assert result['w1'].iloc[10:13].isna().sum() == 0
        # Medium gap should be filled
        assert result['w1'].iloc[50:70].isna().sum() == 0
        # Long gap should remain NaN
        assert result['w1'].iloc[100:180].isna().sum() == 80


class TestMultiPhase:
    """Imputation should work independently on each phase."""

    def test_different_phases_different_gaps(self):
        n = 30
        ts = _make_ts(n)
        w1 = np.full(n, 500.0)
        w2 = np.full(n, 800.0)
        w3 = np.full(n, 200.0)

        w1[5:8] = np.nan    # 3 min — short
        w2[10:20] = np.nan  # 10 min — medium
        # w3 has no NaN

        df = pd.DataFrame({'timestamp': ts, 'w1': w1, 'w2': w2, 'w3': w3})
        result = impute_nan_gaps(df)

        assert result['w1'].isna().sum() == 0
        assert result['w2'].isna().sum() == 0
        assert result['w3'].isna().sum() == 0


class TestNaNAtStart:
    """NaN at the very start of the series (no preceding value for ffill)."""

    def test_nan_at_start_short(self):
        """NaN at start can't be ffilled — should remain NaN for short gaps."""
        n = 20
        ts = _make_ts(n)
        vals = np.full(n, 500.0)
        vals[0:3] = np.nan  # 3 min at start — no preceding value

        df = pd.DataFrame({'timestamp': ts, 'w1': vals, 'w2': vals.copy(), 'w3': vals.copy()})
        result = impute_nan_gaps(df, phase_cols=['w1'])

        # ffill with no preceding value should leave them as NaN
        # That's acceptable — start-of-data NaN is a data boundary, not a gap
        # Just check that the rest is fine
        assert result['w1'].iloc[3:].isna().sum() == 0
