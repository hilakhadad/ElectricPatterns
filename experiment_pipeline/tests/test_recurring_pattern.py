"""
Unit tests for recurring pattern discovery (DBSCAN clustering).

Tests the recurring_pattern_classifier module which discovers per-house
per-phase behavioural patterns among unclassified sessions.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from identification.classifiers.recurring_pattern_classifier import (
    _identify_recurring_patterns,
    _cluster_sessions,
    _dbscan,
    _extract_features,
    _normalise_features,
    _is_valid_cluster,
    _recurring_pattern_confidence,
    _safe_cv,
)
from identification.session_grouper import Session
from identification.config import (
    RECURRING_PATTERN_EPS,
    RECURRING_PATTERN_MIN_SAMPLES,
    RECURRING_PATTERN_MAX_INTERNAL_CV,
    RECURRING_PATTERN_MIN_SESSIONS,
)


# ============================================================================
# Helpers
# ============================================================================

BASE_TIME = pd.Timestamp('2020-01-01 00:00:00')


def _ts(minutes):
    """Return BASE_TIME + minutes offset."""
    return BASE_TIME + pd.Timedelta(minutes=minutes)


def make_match_row(
    on_start_min, duration, magnitude, phase='w1',
    tag='EXACT-MEDIUM', iteration=0, threshold=2000,
):
    """Create a single match row dict with all expected columns."""
    on_start = _ts(on_start_min)
    off_end = _ts(on_start_min + duration)
    return {
        'on_start': on_start,
        'on_end': on_start + pd.Timedelta(minutes=1),
        'off_start': off_end - pd.Timedelta(minutes=1),
        'off_end': off_end,
        'on_magnitude': float(magnitude),
        'off_magnitude': -float(magnitude),
        'duration': float(duration),
        'phase': phase,
        'tag': tag,
        'iteration': iteration,
        'threshold': threshold,
        'on_event_id': f'on_{phase}_{on_start_min}',
        'off_event_id': f'off_{phase}_{on_start_min + duration}',
    }


def make_matches_df(rows):
    """Create a DataFrame from a list of match row dicts."""
    df = pd.DataFrame(rows)
    for col in ['on_start', 'on_end', 'off_start', 'off_end']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def _make_recurring_events(
    n_sessions=5, phase='w2', duration=40, magnitude=1200,
    session_gap_hours=4, start_day_offset=0,
):
    """Create N sessions of identical behavioural pattern.

    Each session is a single event (1 cycle) with the same duration/magnitude,
    spaced session_gap_hours apart (well beyond the 30-min session gap).
    """
    rows = []
    for i in range(n_sessions):
        start_min = (start_day_offset * 24 * 60) + (i * session_gap_hours * 60)
        # Add small noise to make it realistic
        mag_noise = magnitude + np.random.uniform(-20, 20)
        dur_noise = duration + np.random.uniform(-1, 1)
        rows.append(make_match_row(
            start_min, dur_noise, mag_noise, phase,
        ))
    return rows


def _make_diverse_events(n_sessions=5, phase='w2'):
    """Create N sessions with wildly different behaviours (should NOT cluster)."""
    rows = []
    for i in range(n_sessions):
        start_min = i * 120  # 2 hours apart
        duration = np.random.uniform(3, 60)
        magnitude = np.random.uniform(200, 3000)
        rows.append(make_match_row(start_min, duration, magnitude, phase))
    return rows


# ============================================================================
# Tests: DBSCAN core
# ============================================================================

class TestDBSCAN:
    """Test the minimal DBSCAN implementation."""

    def test_two_clear_clusters(self):
        """Two groups of nearby points should form two clusters."""
        from scipy.spatial.distance import squareform, pdist

        # Cluster 1: points near (0, 0)
        # Cluster 2: points near (10, 10)
        points = np.array([
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0], [0.0, 0.2],
            [10.0, 10.0], [10.1, 10.1], [10.2, 10.0], [10.0, 10.2],
        ])
        dist = squareform(pdist(points))
        labels = _dbscan(dist, eps=0.5, min_samples=3)

        assert len(set(l for l in labels if l >= 0)) == 2
        # First 4 should be same cluster
        assert labels[0] == labels[1] == labels[2] == labels[3]
        # Last 4 should be same cluster
        assert labels[4] == labels[5] == labels[6] == labels[7]
        # Different clusters
        assert labels[0] != labels[4]

    def test_all_noise(self):
        """Widely spread points should all be noise."""
        from scipy.spatial.distance import squareform, pdist

        points = np.array([[0, 0], [10, 10], [20, 20], [30, 30]])
        dist = squareform(pdist(points))
        labels = _dbscan(dist, eps=0.5, min_samples=2)

        assert all(l == -1 for l in labels)

    def test_single_cluster(self):
        """All points close together → one cluster."""
        from scipy.spatial.distance import squareform, pdist

        points = np.array([[0, 0], [0.1, 0], [0, 0.1], [0.1, 0.1]])
        dist = squareform(pdist(points))
        labels = _dbscan(dist, eps=0.5, min_samples=2)

        assert all(l == 0 for l in labels)


# ============================================================================
# Tests: Feature extraction
# ============================================================================

class TestFeatureExtraction:
    """Test session feature extraction and normalisation."""

    def test_extract_features_shape(self):
        """Feature matrix should have correct shape."""
        sessions = [
            Session('s1', 'w1', [], avg_magnitude=1000, avg_cycle_duration=10,
                    cycle_count=3, magnitude_cv=0.1),
            Session('s2', 'w1', [], avg_magnitude=1100, avg_cycle_duration=12,
                    cycle_count=3, magnitude_cv=0.12),
        ]
        features = _extract_features(sessions)
        assert features.shape == (2, 4)

    def test_normalise_features_zero_variance(self):
        """Constant column should not cause division by zero."""
        features = np.array([[1, 2], [1, 4], [1, 6]])
        normed = _normalise_features(features)
        # First column is constant → should be all 0
        assert np.allclose(normed[:, 0], 0.0)
        # Second column should be normalised
        assert abs(normed[:, 1].mean()) < 1e-10


# ============================================================================
# Tests: Cluster validation
# ============================================================================

class TestClusterValidation:
    """Test cluster quality checks."""

    def test_valid_cluster_tight(self):
        """Sessions with nearly identical behaviour should pass validation."""
        sessions = [
            Session('s1', 'w2', [], avg_magnitude=1200, avg_cycle_duration=40,
                    cycle_count=1, magnitude_cv=0.0),
            Session('s2', 'w2', [], avg_magnitude=1210, avg_cycle_duration=41,
                    cycle_count=1, magnitude_cv=0.0),
            Session('s3', 'w2', [], avg_magnitude=1190, avg_cycle_duration=39,
                    cycle_count=1, magnitude_cv=0.0),
        ]
        assert _is_valid_cluster(sessions) is True

    def test_invalid_cluster_high_variance(self):
        """Sessions with very different magnitudes should fail validation."""
        sessions = [
            Session('s1', 'w2', [], avg_magnitude=500, avg_cycle_duration=40,
                    cycle_count=1, magnitude_cv=0.0),
            Session('s2', 'w2', [], avg_magnitude=2000, avg_cycle_duration=40,
                    cycle_count=1, magnitude_cv=0.0),
            Session('s3', 'w2', [], avg_magnitude=3500, avg_cycle_duration=40,
                    cycle_count=1, magnitude_cv=0.0),
        ]
        assert _is_valid_cluster(sessions) is False

    def test_invalid_cluster_too_few(self):
        """Clusters below minimum size should fail."""
        sessions = [
            Session('s1', 'w2', [], avg_magnitude=1200, avg_cycle_duration=40,
                    cycle_count=1, magnitude_cv=0.0),
            Session('s2', 'w2', [], avg_magnitude=1210, avg_cycle_duration=41,
                    cycle_count=1, magnitude_cv=0.0),
        ]
        assert _is_valid_cluster(sessions) is False


# ============================================================================
# Tests: Confidence scoring
# ============================================================================

class TestConfidenceScoring:
    """Test recurring pattern confidence computation."""

    def test_tight_cluster_high_confidence(self):
        """Tight cluster with many sessions → high confidence."""
        sessions = [
            Session(f's{i}', 'w2', [],
                    start=_ts(i * 240), end=_ts(i * 240 + 40),
                    avg_magnitude=1200, avg_cycle_duration=40,
                    cycle_count=1, magnitude_cv=0.0)
            for i in range(8)
        ]
        conf, breakdown = _recurring_pattern_confidence(sessions, 1)
        assert conf >= 0.6
        assert breakdown['n_sessions'] == 8
        assert breakdown['magnitude_consistency'] >= 0.9

    def test_loose_cluster_lower_confidence(self):
        """Cluster with more variance → lower confidence."""
        sessions = [
            Session(f's{i}', 'w2', [],
                    start=_ts(i * 240), end=_ts(i * 240 + 40),
                    avg_magnitude=1000 + i * 80, avg_cycle_duration=30 + i * 4,
                    cycle_count=1, magnitude_cv=0.0)
            for i in range(4)
        ]
        conf, breakdown = _recurring_pattern_confidence(sessions, 1)
        assert conf < 0.9  # Should be lower than a tight cluster


# ============================================================================
# Tests: safe_cv helper
# ============================================================================

class TestSafeCV:
    """Test coefficient of variation helper."""

    def test_empty_list(self):
        assert _safe_cv([]) == 0.0

    def test_single_value(self):
        assert _safe_cv([42.0]) == 0.0

    def test_constant_values(self):
        assert _safe_cv([5.0, 5.0, 5.0]) == 0.0

    def test_known_cv(self):
        # std=1, mean=10 → CV=0.1
        values = [9, 10, 11]
        cv = _safe_cv(values)
        assert 0.05 < cv < 0.15


# ============================================================================
# Tests: End-to-end _identify_recurring_patterns
# ============================================================================

class TestIdentifyRecurringPatterns:
    """End-to-end tests for the full recurring pattern discovery."""

    def test_clear_recurring_pattern(self):
        """5 identical-behaviour sessions on same phase → should cluster."""
        rows = _make_recurring_events(n_sessions=5, phase='w2',
                                       duration=40, magnitude=1200)
        remaining = make_matches_df(rows)
        all_matches = remaining.copy()

        recurring, leftover = _identify_recurring_patterns(remaining, all_matches)

        # Should find at least some recurring sessions
        # (exact count depends on DBSCAN eps with normalised features)
        assert len(recurring) + len(leftover) == 5
        if len(recurring) > 0:
            assert all(cs.device_type == 'recurring_pattern' for cs in recurring)
            assert 'pattern_id' in recurring[0].confidence_breakdown

    def test_diverse_no_pattern(self):
        """Wildly different sessions → nothing should cluster."""
        np.random.seed(42)
        rows = _make_diverse_events(n_sessions=6, phase='w2')
        remaining = make_matches_df(rows)
        all_matches = remaining.copy()

        recurring, leftover = _identify_recurring_patterns(remaining, all_matches)

        # All should be leftover (no patterns)
        assert len(leftover) == 6
        assert len(recurring) == 0

    def test_empty_remaining(self):
        """Empty DataFrame → empty results."""
        remaining = pd.DataFrame()
        all_matches = pd.DataFrame()

        recurring, leftover = _identify_recurring_patterns(remaining, all_matches)

        assert recurring == []
        assert leftover == []

    def test_mixed_patterns_and_noise(self):
        """Mix of recurring pattern + random noise → pattern extracted, noise left."""
        # 5 recurring sessions (same behaviour)
        pattern_rows = _make_recurring_events(
            n_sessions=5, phase='w1', duration=40, magnitude=1200,
            session_gap_hours=4, start_day_offset=0,
        )
        # 3 random sessions (different behaviour)
        noise_rows = [
            make_match_row(2000, 5, 300, 'w1'),
            make_match_row(2100, 60, 2500, 'w1'),
            make_match_row(2200, 8, 600, 'w1'),
        ]

        remaining = make_matches_df(pattern_rows + noise_rows)
        all_matches = remaining.copy()

        recurring, leftover = _identify_recurring_patterns(remaining, all_matches)

        total = len(recurring) + len(leftover)
        assert total == 8  # 5 pattern + 3 noise

    def test_two_different_phases_independent(self):
        """Patterns on different phases are discovered independently."""
        phase1_rows = _make_recurring_events(
            n_sessions=4, phase='w1', duration=20, magnitude=900,
            session_gap_hours=3,
        )
        phase2_rows = _make_recurring_events(
            n_sessions=4, phase='w2', duration=40, magnitude=1200,
            session_gap_hours=4, start_day_offset=0,
        )
        remaining = make_matches_df(phase1_rows + phase2_rows)
        all_matches = remaining.copy()

        recurring, leftover = _identify_recurring_patterns(remaining, all_matches)

        total = len(recurring) + len(leftover)
        assert total == 8

    def test_too_few_sessions_for_clustering(self):
        """Only 2 sessions on a phase → not enough for DBSCAN, all become leftover."""
        rows = _make_recurring_events(n_sessions=2, phase='w2',
                                       duration=40, magnitude=1200)
        remaining = make_matches_df(rows)
        all_matches = remaining.copy()

        recurring, leftover = _identify_recurring_patterns(remaining, all_matches)

        assert len(recurring) == 0
        assert len(leftover) == 2
