"""
Unit tests for cross-house recurring pattern matching.

Tests the cross_house_patterns module which compares recurring_pattern
signatures across houses and assigns global device names.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from metrics.cross_house_patterns import (
    match_patterns_across_houses,
    _extract_pattern_signatures,
    _group_signatures,
    _assign_global_names,
    _signatures_match,
    _compute_match_quality,
    _safe_cv,
    update_house_jsons,
    save_cross_house_summary,
    CROSS_HOUSE_MAGNITUDE_TOLERANCE,
    CROSS_HOUSE_DURATION_TOLERANCE,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_session(pattern_id, magnitude=1200, duration=40, confidence=0.8, phase='w1'):
    """Create a minimal recurring_pattern session dict."""
    return {
        'session_id': f's_{pattern_id}_{magnitude}',
        'device_type': 'recurring_pattern',
        'confidence': confidence,
        'confidence_breakdown': {
            'pattern_id': pattern_id,
            'avg_magnitude_w': float(magnitude),
            'avg_duration_min': float(duration),
            'n_sessions': 5,
            'cluster_size': 0.75,
            'magnitude_consistency': 0.9,
            'duration_consistency': 0.9,
            'recurrence_regularity': 0.6,
        },
        'classification_reason': f'recurring pattern #{pattern_id}',
        'start': '2020-01-01T08:00:00',
        'end': '2020-01-01T08:40:00',
        'duration_minutes': float(duration),
        'phases': [phase],
        'cycle_count': 1,
        'avg_cycle_magnitude_w': float(magnitude),
    }


def _make_unknown_session():
    """Create an unknown session (should be ignored)."""
    return {
        'session_id': 's_unknown_1',
        'device_type': 'unknown',
        'confidence': 0.3,
        'confidence_breakdown': {},
        'start': '2020-01-01T10:00:00',
        'end': '2020-01-01T10:30:00',
        'duration_minutes': 30,
        'phases': ['w1'],
        'cycle_count': 1,
        'avg_cycle_magnitude_w': 500,
    }


def _write_house_json(experiment_dir, house_id, sessions):
    """Write a device_sessions JSON file for testing."""
    sessions_dir = Path(experiment_dir) / 'device_sessions'
    os.makedirs(sessions_dir, exist_ok=True)
    data = {
        'house_id': house_id,
        'sessions': sessions,
        'summary': {},
    }
    path = sessions_dir / f'device_sessions_{house_id}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return path


def _make_signature(house_id, pattern_id, magnitude, duration, n_sessions=5, confidence=0.8):
    """Create a signature dict for direct testing."""
    return {
        'house_id': str(house_id),
        'pattern_id': int(pattern_id),
        'avg_magnitude_w': float(magnitude),
        'avg_duration_min': float(duration),
        'n_sessions': n_sessions,
        'confidence': confidence,
    }


# ============================================================================
# Tests: Signature extraction
# ============================================================================

class TestExtractSignatures:
    """Test signature extraction from session JSON data."""

    def test_extract_recurring_pattern(self, tmp_path):
        """Sessions with recurring_pattern type yield signatures."""
        _write_house_json(tmp_path, '305', [
            _make_session(1, magnitude=1200, duration=40),
        ])
        sigs = _extract_pattern_signatures(tmp_path, ['305'])
        assert len(sigs) == 1
        assert sigs[0]['house_id'] == '305'
        assert sigs[0]['pattern_id'] == 1
        assert sigs[0]['avg_magnitude_w'] == 1200
        assert sigs[0]['avg_duration_min'] == 40

    def test_skip_non_recurring(self, tmp_path):
        """Non-recurring sessions are ignored."""
        _write_house_json(tmp_path, '305', [
            _make_unknown_session(),
        ])
        sigs = _extract_pattern_signatures(tmp_path, ['305'])
        assert len(sigs) == 0

    def test_empty_sessions(self, tmp_path):
        """No sessions -> empty list."""
        _write_house_json(tmp_path, '305', [])
        sigs = _extract_pattern_signatures(tmp_path, ['305'])
        assert len(sigs) == 0

    def test_multiple_patterns_per_house(self, tmp_path):
        """House with pattern_id 1 and 2 yields two signatures."""
        _write_house_json(tmp_path, '305', [
            _make_session(1, magnitude=1200, duration=40),
            _make_session(1, magnitude=1200, duration=40),  # same pattern
            _make_session(2, magnitude=2500, duration=10),
        ])
        sigs = _extract_pattern_signatures(tmp_path, ['305'])
        assert len(sigs) == 2
        pids = sorted(s['pattern_id'] for s in sigs)
        assert pids == [1, 2]

    def test_missing_house_file(self, tmp_path):
        """Missing JSON file is silently skipped."""
        sigs = _extract_pattern_signatures(tmp_path, ['999'])
        assert len(sigs) == 0

    def test_multiple_houses(self, tmp_path):
        """Signatures extracted from multiple houses."""
        _write_house_json(tmp_path, '305', [
            _make_session(1, magnitude=1200, duration=40),
        ])
        _write_house_json(tmp_path, '2008', [
            _make_session(1, magnitude=1250, duration=38),
        ])
        sigs = _extract_pattern_signatures(tmp_path, ['305', '2008'])
        assert len(sigs) == 2
        houses = set(s['house_id'] for s in sigs)
        assert houses == {'305', '2008'}


# ============================================================================
# Tests: Signature matching
# ============================================================================

class TestSignatureMatching:
    """Test pairwise signature matching."""

    def test_identical_match(self):
        """Identical signatures match."""
        s1 = _make_signature('A', 1, 1200, 40)
        s2 = _make_signature('B', 1, 1200, 40)
        assert _signatures_match(s1, s2, 0.20, 0.30)

    def test_within_tolerance(self):
        """Signatures within tolerance match."""
        s1 = _make_signature('A', 1, 1200, 40)
        s2 = _make_signature('B', 1, 1400, 48)  # 14.3% mag, 16.7% dur
        assert _signatures_match(s1, s2, 0.20, 0.30)

    def test_beyond_magnitude_tolerance(self):
        """Signatures beyond magnitude tolerance don't match."""
        s1 = _make_signature('A', 1, 1000, 40)
        s2 = _make_signature('B', 1, 1500, 40)  # 33% mag diff
        assert not _signatures_match(s1, s2, 0.20, 0.30)

    def test_beyond_duration_tolerance(self):
        """Signatures beyond duration tolerance don't match."""
        s1 = _make_signature('A', 1, 1200, 20)
        s2 = _make_signature('B', 1, 1200, 40)  # 50% dur diff
        assert not _signatures_match(s1, s2, 0.20, 0.30)

    def test_zero_magnitude(self):
        """Zero magnitude never matches."""
        s1 = _make_signature('A', 1, 0, 40)
        s2 = _make_signature('B', 1, 0, 40)
        assert not _signatures_match(s1, s2, 0.20, 0.30)


# ============================================================================
# Tests: Grouping
# ============================================================================

class TestGroupSignatures:
    """Test complete-linkage grouping logic."""

    def test_two_matching_houses(self):
        """Two similar signatures from different houses form a group."""
        sigs = [
            _make_signature('A', 1, 1200, 40),
            _make_signature('B', 1, 1250, 38),
        ]
        groups, unmatched = _group_signatures(sigs, 0.20, 0.30)
        assert len(groups) == 1
        assert len(groups[0]) == 2
        assert len(unmatched) == 0

    def test_same_house_no_match(self):
        """Two signatures from the same house never match."""
        sigs = [
            _make_signature('A', 1, 1200, 40),
            _make_signature('A', 2, 1200, 40),
        ]
        groups, unmatched = _group_signatures(sigs, 0.20, 0.30)
        assert len(groups) == 0
        assert len(unmatched) == 2

    def test_no_transitive_chaining(self):
        """Complete-linkage prevents transitive chaining: A~B and B~C but A!~C -> no single group.

        A(1000W) and C(1300W) differ by 23% (> 20% tolerance), so they can't
        be in the same group even though B matches both individually.
        Result: one group of 2 + one unmatched signature.
        """
        sigs = [
            _make_signature('house1', 1, 1000, 30),
            _make_signature('house2', 1, 1150, 33),   # matches house1 (13%) and house3 (11.5%)
            _make_signature('house3', 1, 1300, 36),    # matches house2 but NOT house1 (23%)
        ]
        groups, unmatched = _group_signatures(sigs, 0.20, 0.30)
        # Complete-linkage: one group of 2, one unmatched
        assert len(groups) == 1
        assert len(groups[0]) == 2
        assert len(unmatched) == 1

    def test_all_pairs_match_forms_single_group(self):
        """When ALL cross-house pairs match, complete-linkage allows a 3-house group."""
        sigs = [
            _make_signature('house1', 1, 1000, 30),
            _make_signature('house2', 1, 1050, 31),   # 5% from house1, 9.5% from house3
            _make_signature('house3', 1, 1100, 33),    # 10% from house1
        ]
        groups, unmatched = _group_signatures(sigs, 0.20, 0.30)
        assert len(groups) == 1
        assert len(groups[0]) == 3
        assert len(unmatched) == 0

    def test_no_match_all_unmatched(self):
        """Widely different signatures all become unmatched."""
        sigs = [
            _make_signature('A', 1, 500, 10),
            _make_signature('B', 1, 2000, 60),
            _make_signature('C', 1, 3500, 120),
        ]
        groups, unmatched = _group_signatures(sigs, 0.20, 0.30)
        assert len(groups) == 0
        assert len(unmatched) == 3

    def test_two_separate_groups(self):
        """Two distinct clusters form two separate groups."""
        sigs = [
            # Group 1: ~1200W/~40min
            _make_signature('A', 1, 1200, 40),
            _make_signature('B', 1, 1250, 38),
            # Group 2: ~3000W/~10min
            _make_signature('C', 1, 3000, 10),
            _make_signature('D', 1, 3100, 11),
        ]
        groups, unmatched = _group_signatures(sigs, 0.20, 0.30)
        assert len(groups) == 2
        assert len(unmatched) == 0

    def test_empty_input(self):
        """No signatures -> no groups."""
        groups, unmatched = _group_signatures([], 0.20, 0.30)
        assert groups == []
        assert unmatched == []

    def test_single_signature(self):
        """Single signature is unmatched."""
        sigs = [_make_signature('A', 1, 1200, 40)]
        groups, unmatched = _group_signatures(sigs, 0.20, 0.30)
        assert len(groups) == 0
        assert len(unmatched) == 1


# ============================================================================
# Tests: Naming
# ============================================================================

class TestAssignGlobalNames:
    """Test naming convention."""

    def test_sorted_by_magnitude(self):
        """Higher magnitude pattern gets earlier letter."""
        groups = [
            [_make_signature('A', 1, 800, 20), _make_signature('B', 1, 850, 22)],
            [_make_signature('C', 1, 2000, 40), _make_signature('D', 1, 2100, 38)],
        ]
        named = _assign_global_names(groups)
        assert len(named) == 2
        assert named[0]['global_name'] == 'Device A'
        assert named[0]['avg_magnitude_w'] > named[1]['avg_magnitude_w']
        assert named[1]['global_name'] == 'Device B'

    def test_descriptive_name_format(self):
        """Descriptive name follows ~{mag}W / ~{dur}min format."""
        groups = [
            [_make_signature('A', 1, 1882, 14), _make_signature('B', 1, 1920, 15)],
        ]
        named = _assign_global_names(groups)
        desc = named[0]['descriptive_name']
        assert '~' in desc
        assert 'W' in desc
        assert 'min' in desc

    def test_empty_groups(self):
        """No groups -> empty result."""
        assert _assign_global_names([]) == []

    def test_match_quality_included(self):
        """Each named pattern includes match_quality score."""
        groups = [
            [_make_signature('A', 1, 1200, 40), _make_signature('B', 1, 1200, 40)],
        ]
        named = _assign_global_names(groups)
        assert 'match_quality' in named[0]
        assert 0 <= named[0]['match_quality'] <= 1


# ============================================================================
# Tests: Match quality
# ============================================================================

class TestMatchQuality:
    """Test match quality scoring."""

    def test_tight_match_high_quality(self):
        """Nearly identical signatures -> high quality."""
        group = [
            _make_signature('A', 1, 1200, 40),
            _make_signature('B', 1, 1200, 40),
            _make_signature('C', 1, 1200, 40),
        ]
        quality = _compute_match_quality(group)
        assert quality >= 0.7

    def test_looser_match_lower_quality(self):
        """Signatures at tolerance edges -> lower quality."""
        group = [
            _make_signature('A', 1, 1000, 30),
            _make_signature('B', 1, 1180, 38),
        ]
        quality = _compute_match_quality(group)
        assert quality < 0.9


# ============================================================================
# Tests: safe_cv helper
# ============================================================================

class TestSafeCV:
    def test_empty(self):
        assert _safe_cv([]) == 0.0

    def test_single(self):
        assert _safe_cv([42]) == 0.0

    def test_constant(self):
        assert _safe_cv([5, 5, 5]) == 0.0


# ============================================================================
# Tests: JSON updates
# ============================================================================

class TestUpdateHouseJsons:
    """Test per-house JSON update functionality."""

    def test_updates_matching_sessions(self, tmp_path):
        """Global name is added to matching recurring_pattern sessions."""
        _write_house_json(tmp_path, '305', [
            _make_session(1, magnitude=1200, duration=40),
            _make_session(2, magnitude=2500, duration=10),
        ])

        house_updates = {
            '305': {
                1: {'global_pattern_name': 'Device A', 'global_descriptive_name': '~1200W / ~40min'},
            },
        }
        count = update_house_jsons(tmp_path, house_updates)
        assert count == 1

        # Verify the JSON was updated
        json_path = tmp_path / 'device_sessions' / 'device_sessions_305.json'
        with open(json_path) as f:
            data = json.load(f)

        for s in data['sessions']:
            bd = s['confidence_breakdown']
            if bd.get('pattern_id') == 1:
                assert bd['global_pattern_name'] == 'Device A'
                assert bd['global_descriptive_name'] == '~1200W / ~40min'
            elif bd.get('pattern_id') == 2:
                assert 'global_pattern_name' not in bd

    def test_idempotent(self, tmp_path):
        """Running update twice produces same result."""
        _write_house_json(tmp_path, '305', [
            _make_session(1, magnitude=1200, duration=40),
        ])
        house_updates = {
            '305': {1: {'global_pattern_name': 'Device A', 'global_descriptive_name': '~1200W / ~40min'}},
        }
        count1 = update_house_jsons(tmp_path, house_updates)
        count2 = update_house_jsons(tmp_path, house_updates)
        assert count1 == 1
        assert count2 == 1

    def test_missing_house(self, tmp_path):
        """Missing house JSON is silently skipped."""
        house_updates = {
            '999': {1: {'global_pattern_name': 'Device A', 'global_descriptive_name': '~1200W / ~40min'}},
        }
        count = update_house_jsons(tmp_path, house_updates)
        assert count == 0


# ============================================================================
# Tests: Save summary
# ============================================================================

class TestSaveSummary:
    def test_saves_json(self, tmp_path):
        """Summary JSON is saved correctly."""
        result = {
            'settings': {'magnitude_tolerance': 0.20, 'duration_tolerance': 0.30},
            'summary': {'total_global_patterns': 1},
            'global_patterns': [{'global_name': 'Device A'}],
            'unmatched_patterns': [],
        }
        path = save_cross_house_summary(tmp_path, result)
        assert path.exists()

        with open(path) as f:
            saved = json.load(f)
        assert saved['summary']['total_global_patterns'] == 1
        assert 'generated_at' in saved


# ============================================================================
# Tests: End-to-end
# ============================================================================

class TestEndToEnd:
    """Integration tests using synthetic JSON data."""

    def test_two_houses_matching_pattern(self, tmp_path):
        """Two houses with similar pattern -> one global pattern."""
        _write_house_json(tmp_path, '305', [
            _make_session(1, magnitude=1200, duration=40),
        ])
        _write_house_json(tmp_path, '2008', [
            _make_session(1, magnitude=1250, duration=38),
        ])

        result = match_patterns_across_houses(tmp_path, ['305', '2008'])

        assert result['summary']['total_global_patterns'] == 1
        assert result['summary']['total_unmatched'] == 0
        gp = result['global_patterns'][0]
        assert gp['global_name'] == 'Device A'
        assert gp['n_houses'] == 2

    def test_three_houses_two_patterns(self, tmp_path):
        """Three houses: pattern A in house 1+2, pattern B in house 2+3."""
        _write_house_json(tmp_path, 'h1', [
            _make_session(1, magnitude=1200, duration=40),
        ])
        _write_house_json(tmp_path, 'h2', [
            _make_session(1, magnitude=1250, duration=38),   # matches h1
            _make_session(2, magnitude=3000, duration=10),   # matches h3
        ])
        _write_house_json(tmp_path, 'h3', [
            _make_session(1, magnitude=3100, duration=11),   # matches h2 pattern 2
        ])

        result = match_patterns_across_houses(tmp_path, ['h1', 'h2', 'h3'])

        assert result['summary']['total_global_patterns'] == 2
        names = sorted(gp['global_name'] for gp in result['global_patterns'])
        assert names == ['Device A', 'Device B']

    def test_no_patterns_no_crash(self, tmp_path):
        """No recurring_pattern sessions -> empty result, no crash."""
        _write_house_json(tmp_path, '305', [_make_unknown_session()])
        _write_house_json(tmp_path, '2008', [_make_unknown_session()])

        result = match_patterns_across_houses(tmp_path, ['305', '2008'])

        assert result['summary']['total_global_patterns'] == 0
        assert result['summary']['total_signatures'] == 0
        assert result['house_updates'] == {}

    def test_single_house(self, tmp_path):
        """Single house can't form cross-house patterns."""
        _write_house_json(tmp_path, '305', [
            _make_session(1, magnitude=1200, duration=40),
        ])
        result = match_patterns_across_houses(tmp_path, ['305'])

        assert result['summary']['total_global_patterns'] == 0
        assert result['summary']['total_unmatched'] == 1

    def test_no_houses(self, tmp_path):
        """Empty house list -> empty result."""
        result = match_patterns_across_houses(tmp_path, [])
        assert result['summary']['total_global_patterns'] == 0

    def test_full_flow_with_json_update(self, tmp_path):
        """Full flow: match, update JSONs, save summary."""
        _write_house_json(tmp_path, 'A', [
            _make_session(1, magnitude=1200, duration=40),
        ])
        _write_house_json(tmp_path, 'B', [
            _make_session(1, magnitude=1250, duration=38),
        ])

        result = match_patterns_across_houses(tmp_path, ['A', 'B'])
        assert result['summary']['total_global_patterns'] == 1

        # Update JSONs
        count = update_house_jsons(tmp_path, result['house_updates'])
        assert count == 2  # one session in each house

        # Save summary
        summary_path = save_cross_house_summary(tmp_path, result)
        assert summary_path.exists()

        # Verify JSON update
        with open(tmp_path / 'device_sessions' / 'device_sessions_A.json') as f:
            data_a = json.load(f)
        rp_session = [s for s in data_a['sessions'] if s['device_type'] == 'recurring_pattern'][0]
        assert rp_session['confidence_breakdown']['global_pattern_name'] == 'Device A'
