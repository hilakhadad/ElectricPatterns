"""
Unit tests for identification.session_output — JSON output builder.

Tests cover:
  1. build_session_json() — session JSON file creation and structure
  2. Backward-compatible activations JSON format
  3. Serialization helpers (_ts, _val, _json_serializer)
  4. _build_summary() — summary statistics
  5. _build_session_entry() — individual session entry construction
"""
import json
import math
import pytest
import numpy as np
import pandas as pd
import sys
from datetime import datetime
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from identification.session_output import (
    build_session_json,
    _build_session_entry,
    _build_summary,
    _ts,
    _val,
    _json_serializer,
)
from identification.session_classifier import ClassifiedSession
from identification.session_grouper import (
    Session,
    MultiPhaseSession,
    build_single_event_session,
    _build_session_from_dicts,
)

# ============================================================================
# Helpers
# ============================================================================

BASE_TIME = pd.Timestamp('2020-01-01 00:00:00')


def _time(minutes):
    """Return BASE_TIME + minutes offset."""
    return BASE_TIME + pd.Timedelta(minutes=minutes)


def make_match_row(
    on_start_min, duration, magnitude, phase='w1',
    tag='EXACT-MEDIUM', iteration=0, threshold=2000,
):
    """Create a single match row dict."""
    on_start = _time(on_start_min)
    off_end = _time(on_start_min + duration)
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


def _make_boiler_classified_session(phase='w1'):
    """Create a ClassifiedSession for a boiler."""
    event = make_match_row(0, 30, 2000, phase)
    session = build_single_event_session(event, phase)
    return ClassifiedSession(
        session=session,
        device_type='boiler',
        reason='Single event >=15min, >=1500W, isolated',
        confidence=0.85,
        confidence_breakdown={'duration': 0.9, 'magnitude': 0.8, 'isolation': 1.0},
    )


def _make_ac_classified_session(phase='w1'):
    """Create a ClassifiedSession for a regular AC."""
    events = []
    cursor = 0
    for i in range(5):
        events.append(make_match_row(cursor, 10, 1200, phase))
        cursor += 15
    session = _build_session_from_dicts(events, phase)
    return ClassifiedSession(
        session=session,
        device_type='regular_ac',
        reason='5 cycles, CV=0.00',
        confidence=0.75,
        confidence_breakdown={'cycle_count': 0.7, 'magnitude_cv': 1.0, 'magnitude': 0.75},
    )


def _make_central_ac_classified_session():
    """Create a ClassifiedSession for a central AC (multi-phase)."""
    w1_events = [make_match_row(i * 15, 10, 1200, 'w1') for i in range(4)]
    w2_events = [make_match_row(i * 15 + 2, 10, 1100, 'w2') for i in range(4)]
    s1 = _build_session_from_dicts(w1_events, 'w1')
    s2 = _build_session_from_dicts(w2_events, 'w2')
    mp = MultiPhaseSession(
        session_id='mp_test123',
        phases=['w1', 'w2'],
        phase_sessions={'w1': s1, 'w2': s2},
        start=min(s1.start, s2.start),
        end=max(s1.end, s2.end),
        total_magnitude=s1.avg_magnitude + s2.avg_magnitude,
    )
    return ClassifiedSession(
        session=mp,
        device_type='central_ac',
        reason='Independent AC cycling on 2 phases',
        confidence=0.80,
        confidence_breakdown={'phase_count': 0.7, 'sync_quality': 0.9},
    )


def _make_unknown_classified_session(phase='w1'):
    """Create a ClassifiedSession for an unknown device."""
    event = make_match_row(0, 5, 500, phase)
    session = build_single_event_session(event, phase)
    return ClassifiedSession(
        session=session,
        device_type='unknown',
        reason='not_boiler: 80% | not_ac: 90%',
        confidence=0.2,
        confidence_breakdown={'not_boiler': 0.8, 'not_ac': 0.9},
    )


def _make_classified_sessions_dict():
    """Build a Dict[str, List[ClassifiedSession]] like classify_events() returns."""
    return {
        'boiler': [_make_boiler_classified_session()],
        'three_phase_device': [],
        'central_ac': [_make_central_ac_classified_session()],
        'regular_ac': [_make_ac_classified_session()],
        'unknown': [_make_unknown_classified_session()],
    }


# ============================================================================
# Tests: Serialization helpers
# ============================================================================

class TestTimestampSerializer:
    """Test _ts() timestamp serializer."""

    def test_pd_timestamp_to_iso(self):
        ts = pd.Timestamp('2020-01-01 10:00:00')
        assert _ts(ts) == '2020-01-01T10:00:00'

    def test_none_returns_none(self):
        assert _ts(None) is None

    def test_nan_returns_none(self):
        assert _ts(float('nan')) is None

    def test_string_passthrough(self):
        assert _ts('2020-01-01T10:00:00') == '2020-01-01T10:00:00'

    def test_nat_returns_string_representation(self):
        """pd.NaT falls through to str() in the current implementation."""
        result = _ts(pd.NaT)
        # pd.NaT is not a float NaN and may not be caught by isinstance(pd.Timestamp)
        # depending on pandas version; accept either None or 'NaT' string
        assert result is None or result == 'NaT'


class TestValueSerializer:
    """Test _val() numeric value serializer."""

    def test_none_returns_none(self):
        assert _val(None) is None

    def test_nan_returns_none(self):
        assert _val(float('nan')) is None

    def test_inf_returns_none(self):
        assert _val(float('inf')) is None

    def test_float_rounded(self):
        assert _val(1.23456) == 1.23

    def test_int_passthrough(self):
        assert _val(42) == 42

    def test_numpy_int_converted(self):
        val = np.int64(42)
        result = _val(val)
        assert result == 42
        assert isinstance(result, int)

    def test_numpy_float_converted(self):
        val = np.float64(3.14159)
        result = _val(val)
        assert result == 3.14
        assert isinstance(result, float)


class TestJsonSerializer:
    """Test _json_serializer() fallback serializer."""

    def test_pd_timestamp(self):
        ts = pd.Timestamp('2020-06-15 12:00:00')
        assert _json_serializer(ts) == '2020-06-15T12:00:00'

    def test_datetime(self):
        dt = datetime(2020, 6, 15, 12, 0)
        assert _json_serializer(dt) == '2020-06-15T12:00:00'

    def test_numpy_int(self):
        val = np.int64(42)
        assert _json_serializer(val) == 42

    def test_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = _json_serializer(arr)
        assert result == [1, 2, 3]


# ============================================================================
# Tests: _build_session_entry
# ============================================================================

class TestBuildSessionEntry:
    """Test individual session entry construction."""

    def test_single_phase_session_entry(self):
        cs = _make_boiler_classified_session()
        entry = _build_session_entry(cs)
        assert entry['device_type'] == 'boiler'
        assert entry['confidence'] == 0.85
        assert entry['phases'] == ['w1']
        assert 'constituent_events' in entry
        assert len(entry['constituent_events']) == 1
        assert 'session_id' in entry
        assert 'start' in entry
        assert 'end' in entry
        assert 'duration_minutes' in entry
        assert 'cycle_count' in entry

    def test_multi_phase_session_entry(self):
        cs = _make_central_ac_classified_session()
        entry = _build_session_entry(cs)
        assert entry['device_type'] == 'central_ac'
        assert set(entry['phases']) == {'w1', 'w2'}
        assert 'constituent_events' in entry
        # 4 events per phase * 2 phases = 8
        assert len(entry['constituent_events']) == 8
        assert entry['phase_presence']['w1'] == 'V'
        assert entry['phase_presence']['w2'] == 'V'
        assert entry['phase_presence']['w3'] == 'X'

    def test_entry_has_phase_magnitudes(self):
        cs = _make_boiler_classified_session()
        entry = _build_session_entry(cs)
        assert 'phase_magnitudes' in entry
        assert 'w1' in entry['phase_magnitudes']

    def test_entry_has_classification_reason(self):
        cs = _make_boiler_classified_session()
        entry = _build_session_entry(cs)
        assert entry['classification_reason'] == cs.reason

    def test_entry_has_confidence_breakdown(self):
        cs = _make_boiler_classified_session()
        entry = _build_session_entry(cs)
        assert entry['confidence_breakdown'] == cs.confidence_breakdown

    def test_constituent_events_have_expected_fields(self):
        cs = _make_ac_classified_session()
        entry = _build_session_entry(cs)
        ev = entry['constituent_events'][0]
        assert 'phase' in ev
        assert 'on_start' in ev
        assert 'off_end' in ev
        assert 'on_magnitude' in ev
        assert 'duration' in ev
        assert 'tag' in ev
        assert 'iteration' in ev


# ============================================================================
# Tests: _build_summary
# ============================================================================

class TestBuildSummary:
    """Test summary statistics builder."""

    def test_summary_total_count(self):
        classified = _make_classified_sessions_dict()
        summary = _build_summary(classified)
        # 1 boiler + 1 central_ac + 1 regular_ac + 1 unknown = 4
        assert summary['total_sessions'] == 4

    def test_summary_by_device_type(self):
        classified = _make_classified_sessions_dict()
        summary = _build_summary(classified)
        by_type = summary['by_device_type']
        assert 'boiler' in by_type
        assert by_type['boiler']['count'] == 1
        assert 'regular_ac' in by_type
        assert by_type['regular_ac']['count'] == 1

    def test_empty_types_not_in_summary(self):
        classified = _make_classified_sessions_dict()
        summary = _build_summary(classified)
        by_type = summary['by_device_type']
        assert 'three_phase_device' not in by_type  # count is 0

    def test_summary_has_duration_and_cycles(self):
        classified = _make_classified_sessions_dict()
        summary = _build_summary(classified)
        by_type = summary['by_device_type']
        for dtype, entry in by_type.items():
            assert 'avg_duration_min' in entry
            assert 'avg_cycle_count' in entry


# ============================================================================
# Tests: build_session_json (full integration)
# ============================================================================

class TestBuildSessionJson:
    """Test the full JSON output pipeline."""

    def test_creates_session_json_file(self, tmp_path):
        classified = _make_classified_sessions_dict()
        result_path = build_session_json(
            classified_sessions=classified,
            house_id='test_305',
            threshold_schedule=[2000, 1500, 1100, 800],
            experiment_dir=tmp_path,
        )
        assert result_path.exists()
        assert result_path.name == 'device_sessions_test_305.json'

    def test_creates_backward_compatible_json(self, tmp_path):
        classified = _make_classified_sessions_dict()
        build_session_json(
            classified_sessions=classified,
            house_id='test_305',
            threshold_schedule=[2000, 1500, 1100, 800],
            experiment_dir=tmp_path,
        )
        activations_path = tmp_path / 'device_activations' / 'device_activations_test_305.json'
        assert activations_path.exists()

    def test_session_json_structure(self, tmp_path):
        classified = _make_classified_sessions_dict()
        result_path = build_session_json(
            classified_sessions=classified,
            house_id='test_305',
            threshold_schedule=[2000, 1500, 1100, 800],
            experiment_dir=tmp_path,
        )
        with open(result_path, 'r') as f:
            data = json.load(f)

        assert data['house_id'] == 'test_305'
        assert 'generated_at' in data
        assert data['threshold_schedule'] == [2000, 1500, 1100, 800]
        assert 'summary' in data
        assert 'sessions' in data
        assert isinstance(data['sessions'], list)
        assert len(data['sessions']) > 0

    def test_session_json_sessions_have_correct_fields(self, tmp_path):
        classified = _make_classified_sessions_dict()
        result_path = build_session_json(
            classified_sessions=classified,
            house_id='test_305',
            threshold_schedule=[2000, 1500, 1100, 800],
            experiment_dir=tmp_path,
        )
        with open(result_path, 'r') as f:
            data = json.load(f)

        for session_entry in data['sessions']:
            assert 'session_id' in session_entry
            assert 'device_type' in session_entry
            assert 'confidence' in session_entry
            assert 'start' in session_entry
            assert 'end' in session_entry
            assert 'phases' in session_entry
            assert 'constituent_events' in session_entry

    def test_backward_compatible_json_structure(self, tmp_path):
        classified = _make_classified_sessions_dict()
        build_session_json(
            classified_sessions=classified,
            house_id='test_305',
            threshold_schedule=[2000, 1500, 1100, 800],
            experiment_dir=tmp_path,
        )
        activations_path = tmp_path / 'device_activations' / 'device_activations_test_305.json'
        with open(activations_path, 'r') as f:
            data = json.load(f)

        assert data['house_id'] == 'test_305'
        assert 'total_activations' in data
        assert 'activations' in data
        assert isinstance(data['activations'], list)

    def test_backward_compatible_activations_have_device_type(self, tmp_path):
        classified = _make_classified_sessions_dict()
        build_session_json(
            classified_sessions=classified,
            house_id='test_305',
            threshold_schedule=[2000, 1500, 1100, 800],
            experiment_dir=tmp_path,
        )
        activations_path = tmp_path / 'device_activations' / 'device_activations_test_305.json'
        with open(activations_path, 'r') as f:
            data = json.load(f)

        for act in data['activations']:
            assert 'device_type' in act
            assert 'session_id' in act
            assert 'confidence' in act

    def test_empty_classification_creates_valid_json(self, tmp_path):
        """Empty classification should still produce valid JSON."""
        classified = {
            'boiler': [],
            'three_phase_device': [],
            'central_ac': [],
            'regular_ac': [],
            'unknown': [],
        }
        result_path = build_session_json(
            classified_sessions=classified,
            house_id='test_empty',
            threshold_schedule=[2000, 1500, 1100, 800],
            experiment_dir=tmp_path,
        )
        assert result_path.exists()
        with open(result_path, 'r') as f:
            data = json.load(f)
        assert data['sessions'] == []
        assert data['summary']['total_sessions'] == 0

    def test_sessions_sorted_by_start_time(self, tmp_path):
        classified = _make_classified_sessions_dict()
        result_path = build_session_json(
            classified_sessions=classified,
            house_id='test_305',
            threshold_schedule=[2000, 1500, 1100, 800],
            experiment_dir=tmp_path,
        )
        with open(result_path, 'r') as f:
            data = json.load(f)

        starts = [s['start'] for s in data['sessions'] if s['start']]
        assert starts == sorted(starts)

    def test_spike_stats_included(self, tmp_path):
        """Spike stats should be included in session JSON if provided."""
        classified = _make_classified_sessions_dict()
        spike_stats = {'spike_count': 5, 'kept_count': 20}
        result_path = build_session_json(
            classified_sessions=classified,
            house_id='test_305',
            threshold_schedule=[2000, 1500, 1100, 800],
            experiment_dir=tmp_path,
            spike_stats=spike_stats,
        )
        with open(result_path, 'r') as f:
            data = json.load(f)
        assert data['spike_filter']['spike_count'] == 5
        assert data['spike_filter']['kept_count'] == 20

    def test_json_serializable_with_numpy_types(self, tmp_path):
        """JSON output should handle numpy types without errors."""
        event = make_match_row(0, 30, np.float64(2000.0), 'w1')
        event['on_magnitude'] = np.float64(2000.0)
        event['iteration'] = np.int64(0)
        session = build_single_event_session(event, 'w1')
        cs = ClassifiedSession(
            session=session,
            device_type='boiler',
            reason='test',
            confidence=np.float64(0.85),
            confidence_breakdown={'duration': np.float64(0.9)},
        )
        classified = {
            'boiler': [cs],
            'three_phase_device': [],
            'central_ac': [],
            'regular_ac': [],
            'unknown': [],
        }
        # This should not raise TypeError for numpy types
        result_path = build_session_json(
            classified_sessions=classified,
            house_id='test_numpy',
            threshold_schedule=[2000, 1500, 1100, 800],
            experiment_dir=tmp_path,
        )
        assert result_path.exists()
        # Verify it's valid JSON
        with open(result_path, 'r') as f:
            data = json.load(f)
        assert len(data['sessions']) == 1
