"""
Unit tests for identification.session_grouper — event loading, filtering, and session grouping.

Tests cover:
  1. filter_transient_events() — spike removal based on duration
  2. group_into_sessions() — time-proximity grouping per phase
  3. Session data structure — statistics and properties
  4. _split_sessions_at_initial_run() — prefix event splitting
  5. build_single_event_session() — single-event session creation
  6. detect_phase_synchronized_groups() — multi-phase session detection
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from identification.session_grouper import (
    filter_transient_events,
    group_into_sessions,
    build_single_event_session,
    detect_phase_synchronized_groups,
    Session,
    MultiPhaseSession,
    _build_session_from_dicts,
    _split_sessions_at_initial_run,
)
from identification.config import (
    MIN_EVENT_DURATION_MINUTES,
    DEFAULT_SESSION_GAP_MINUTES,
    CENTRAL_AC_SYNC_TOLERANCE,
    PHASES,
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
    """Create a single match row dict with expected columns."""
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


# ============================================================================
# Tests: filter_transient_events()
# ============================================================================

class TestFilterTransientEvents:
    """Test spike/transient event filtering."""

    def test_removes_events_shorter_than_threshold(self):
        """Events with duration < MIN_EVENT_DURATION_MINUTES should be removed."""
        rows = [
            make_match_row(0, 1, 1000, 'w1'),    # 1 min -> spike
            make_match_row(10, 10, 1500, 'w1'),   # 10 min -> kept
        ]
        df = make_matches_df(rows)
        filtered, stats = filter_transient_events(df)
        assert len(filtered) == 1
        assert filtered.iloc[0]['duration'] == 10

    def test_keeps_events_at_exact_threshold(self):
        """Events at exactly MIN_EVENT_DURATION_MINUTES should be kept."""
        rows = [
            make_match_row(0, MIN_EVENT_DURATION_MINUTES, 1000, 'w1'),
        ]
        df = make_matches_df(rows)
        filtered, stats = filter_transient_events(df)
        assert len(filtered) == 1

    def test_removes_events_just_below_threshold(self):
        """Events at MIN_EVENT_DURATION_MINUTES - 0.5 should be removed."""
        rows = [
            make_match_row(0, MIN_EVENT_DURATION_MINUTES - 0.5, 1000, 'w1'),
        ]
        df = make_matches_df(rows)
        # Manually set the duration to be just below threshold
        df.loc[0, 'duration'] = MIN_EVENT_DURATION_MINUTES - 0.5
        filtered, stats = filter_transient_events(df)
        assert len(filtered) == 0

    def test_empty_input_returns_empty(self):
        """Empty DataFrame should return empty filtered + empty stats."""
        df = pd.DataFrame()
        filtered, stats = filter_transient_events(df)
        assert len(filtered) == 0
        assert stats['spike_count'] == 0

    def test_all_spikes_returns_empty_filtered(self):
        """If all events are spikes, filtered should be empty."""
        rows = [
            make_match_row(0, 1, 1000, 'w1'),
            make_match_row(5, 1, 1200, 'w2'),
        ]
        df = make_matches_df(rows)
        filtered, stats = filter_transient_events(df)
        assert len(filtered) == 0
        assert stats['spike_count'] == 2

    def test_no_spikes_returns_all(self):
        """If no events are spikes, all should be kept."""
        rows = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(20, 15, 1500, 'w2'),
        ]
        df = make_matches_df(rows)
        filtered, stats = filter_transient_events(df)
        assert len(filtered) == 2
        assert stats['spike_count'] == 0

    def test_spike_stats_structure(self):
        """Spike stats should have expected keys."""
        rows = [
            make_match_row(0, 1, 1000, 'w1'),
            make_match_row(10, 10, 1500, 'w1'),
        ]
        df = make_matches_df(rows)
        _, stats = filter_transient_events(df)
        assert 'spike_count' in stats
        assert 'spike_total_minutes' in stats
        assert 'kept_count' in stats
        assert 'kept_total_minutes' in stats
        assert 'by_iteration' in stats
        assert 'by_phase' in stats

    def test_spike_stats_counts_correct(self):
        """Spike stats numbers should match actual filtering."""
        rows = [
            make_match_row(0, 1, 1000, 'w1'),
            make_match_row(5, 1, 800, 'w2'),
            make_match_row(10, 10, 1500, 'w1'),
            make_match_row(30, 20, 2000, 'w2'),
        ]
        df = make_matches_df(rows)
        filtered, stats = filter_transient_events(df)
        assert stats['spike_count'] == 2
        assert stats['spike_total_minutes'] == 2.0
        assert stats['kept_count'] == 2
        assert stats['kept_total_minutes'] == 30.0

    def test_custom_min_duration(self):
        """Custom min_duration parameter should override default."""
        rows = [
            make_match_row(0, 4, 1000, 'w1'),
            make_match_row(10, 6, 1500, 'w1'),
        ]
        df = make_matches_df(rows)
        filtered, stats = filter_transient_events(df, min_duration=5)
        assert len(filtered) == 1
        assert filtered.iloc[0]['duration'] == 6

    def test_missing_duration_column_skips_filter(self):
        """If 'duration' column is missing, skip filter and return all."""
        df = pd.DataFrame({
            'on_start': [_ts(0)],
            'on_magnitude': [1000],
            'phase': ['w1'],
            'iteration': [0],
        })
        filtered, stats = filter_transient_events(df)
        assert len(filtered) == 1

    def test_by_phase_breakdown(self):
        """Spike stats by_phase should break down counts per phase."""
        rows = [
            make_match_row(0, 1, 1000, 'w1'),     # spike on w1
            make_match_row(10, 10, 1500, 'w1'),    # kept on w1
            make_match_row(20, 1, 800, 'w2'),      # spike on w2
        ]
        df = make_matches_df(rows)
        _, stats = filter_transient_events(df)
        assert 'w1' in stats['by_phase']
        assert stats['by_phase']['w1']['spike_count'] == 1
        assert stats['by_phase']['w1']['kept_count'] == 1
        assert 'w2' in stats['by_phase']
        assert stats['by_phase']['w2']['spike_count'] == 1


# ============================================================================
# Tests: group_into_sessions()
# ============================================================================

class TestGroupIntoSessions:
    """Test session grouping by time proximity and phase."""

    def test_empty_input_returns_empty(self):
        """Empty DataFrame -> empty session list."""
        assert group_into_sessions(pd.DataFrame()) == []

    def test_single_event_single_session(self):
        """A single event should produce a single session."""
        rows = [make_match_row(0, 10, 1000, 'w1')]
        df = make_matches_df(rows)
        sessions = group_into_sessions(df)
        assert len(sessions) == 1
        assert sessions[0].phase == 'w1'
        assert sessions[0].cycle_count == 1

    def test_close_events_same_session(self):
        """Events within session_gap minutes should be in the same session."""
        rows = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(15, 10, 1100, 'w1'),   # gap = 5 min < 30
            make_match_row(30, 10, 1050, 'w1'),    # gap = 5 min < 30
        ]
        df = make_matches_df(rows)
        sessions = group_into_sessions(df)
        assert len(sessions) == 1
        assert sessions[0].cycle_count == 3

    def test_far_apart_events_separate_sessions(self):
        """Events more than session_gap apart should be in different sessions."""
        rows = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(100, 10, 1100, 'w1'),  # gap = 80 min > 30
        ]
        df = make_matches_df(rows)
        sessions = group_into_sessions(df)
        assert len(sessions) == 2

    def test_different_phases_separate_sessions(self):
        """Events on different phases should always be in separate sessions."""
        rows = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(5, 10, 1000, 'w2'),    # same time, different phase
        ]
        df = make_matches_df(rows)
        sessions = group_into_sessions(df)
        assert len(sessions) == 2
        phases = {s.phase for s in sessions}
        assert phases == {'w1', 'w2'}

    def test_session_start_end_timestamps(self):
        """Session start/end should match the earliest/latest event times."""
        rows = [
            make_match_row(0, 10, 1000, 'w1'),    # on_start=0, off_end=10
            make_match_row(15, 10, 1100, 'w1'),   # on_start=15, off_end=25
        ]
        df = make_matches_df(rows)
        sessions = group_into_sessions(df)
        assert len(sessions) == 1
        s = sessions[0]
        assert s.start == _ts(0)
        assert s.end == _ts(25)

    def test_session_statistics_computed(self):
        """Session should have correct avg_magnitude, magnitude_cv, etc."""
        rows = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(15, 10, 1000, 'w1'),
        ]
        df = make_matches_df(rows)
        sessions = group_into_sessions(df)
        s = sessions[0]
        assert s.avg_magnitude == 1000.0
        assert s.magnitude_cv == 0.0  # identical magnitudes
        assert s.cycle_count == 2

    def test_session_has_events_list(self):
        """Session should carry the list of constituent events."""
        rows = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(15, 10, 1100, 'w1'),
        ]
        df = make_matches_df(rows)
        sessions = group_into_sessions(df)
        assert len(sessions[0].events) == 2

    def test_custom_gap_minutes(self):
        """Custom gap_minutes should change the grouping threshold."""
        rows = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(20, 10, 1100, 'w1'),   # gap = 10 min
        ]
        df = make_matches_df(rows)
        # With gap=5, they should be in different sessions
        sessions = group_into_sessions(df, gap_minutes=5)
        assert len(sessions) == 2
        # With gap=15, they should be in the same session
        sessions = group_into_sessions(df, gap_minutes=15)
        assert len(sessions) == 1

    def test_three_phases_all_grouped_independently(self):
        """Events on w1, w2, w3 should produce independent sessions per phase."""
        rows = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(0, 10, 1000, 'w2'),
            make_match_row(0, 10, 1000, 'w3'),
        ]
        df = make_matches_df(rows)
        sessions = group_into_sessions(df)
        assert len(sessions) == 3

    def test_events_sorted_by_start_within_session(self):
        """Events should be properly ordered within a session even if input is unsorted."""
        rows = [
            make_match_row(20, 10, 1100, 'w1'),
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(40, 10, 1050, 'w1'),
        ]
        df = make_matches_df(rows)
        sessions = group_into_sessions(df)
        assert len(sessions) == 1  # all within gap
        s = sessions[0]
        assert s.start == _ts(0)
        assert s.end == _ts(50)

    def test_gap_at_exact_threshold_splits(self):
        """A gap of exactly DEFAULT_SESSION_GAP_MINUTES should split sessions."""
        gap = DEFAULT_SESSION_GAP_MINUTES
        rows = [
            make_match_row(0, 10, 1000, 'w1'),
            # off_end of first = minute 10, on_start of second = minute 10 + gap + 1
            make_match_row(10 + gap + 1, 10, 1000, 'w1'),
        ]
        df = make_matches_df(rows)
        sessions = group_into_sessions(df)
        assert len(sessions) == 2


# ============================================================================
# Tests: _split_sessions_at_initial_run()
# ============================================================================

class TestSplitSessionsAtInitialRun:
    """Test the session splitting logic for prefix events."""

    def _make_session_with_events(self, event_dicts, phase='w1'):
        """Build a Session from event dicts."""
        return _build_session_from_dicts(event_dicts, phase)

    def test_no_split_for_short_sessions(self):
        """Sessions with < 3 events should not be split."""
        events = [
            make_match_row(0, 5, 1000, 'w1'),
            make_match_row(10, 15, 1200, 'w1'),
        ]
        session = self._make_session_with_events(events)
        result = _split_sessions_at_initial_run([session])
        assert len(result) == 1

    def test_splits_when_long_event_after_short_prefix(self):
        """[short, short, LONG, ...] -> split into prefix + main sessions."""
        events = [
            make_match_row(0, 3, 1000, 'w1'),      # short prefix
            make_match_row(5, 3, 900, 'w1'),        # short prefix
            make_match_row(10, 20, 1200, 'w1'),     # LONG -> should trigger split
            make_match_row(35, 10, 1100, 'w1'),
        ]
        session = self._make_session_with_events(events)
        result = _split_sessions_at_initial_run([session])
        assert len(result) == 2
        # First session has the prefix events
        assert result[0].cycle_count == 2
        # Second session has the main events
        assert result[1].cycle_count == 2

    def test_no_split_when_first_event_is_longest(self):
        """If the first event is already the longest, no split needed."""
        events = [
            make_match_row(0, 20, 1200, 'w1'),     # LONG first
            make_match_row(25, 10, 1100, 'w1'),
            make_match_row(40, 10, 1050, 'w1'),
        ]
        session = self._make_session_with_events(events)
        result = _split_sessions_at_initial_run([session])
        assert len(result) == 1


# ============================================================================
# Tests: build_single_event_session()
# ============================================================================

class TestBuildSingleEventSession:
    """Test single-event session creation for boiler-type events."""

    def test_creates_session_with_one_event(self):
        event = make_match_row(0, 30, 2000, 'w1')
        session = build_single_event_session(event, 'w1')
        assert isinstance(session, Session)
        assert session.cycle_count == 1
        assert session.phase == 'w1'
        assert len(session.events) == 1

    def test_session_timestamps_match_event(self):
        event = make_match_row(10, 30, 2000, 'w2')
        session = build_single_event_session(event, 'w2')
        assert session.start == _ts(10)
        assert session.end == _ts(40)

    def test_session_magnitude_matches_event(self):
        event = make_match_row(0, 30, 2500, 'w1')
        session = build_single_event_session(event, 'w1')
        assert session.avg_magnitude == 2500.0
        assert session.max_magnitude == 2500.0
        assert session.magnitude_cv == 0.0  # single event


# ============================================================================
# Tests: detect_phase_synchronized_groups()
# ============================================================================

class TestDetectPhaseSynchronizedGroups:
    """Test multi-phase synchronization detection."""

    def _make_session(self, start_min, end_min, phase, n_events=3, magnitude=1200):
        """Create a Session object with specified parameters."""
        events = []
        duration_per = (end_min - start_min) / n_events
        cursor = start_min
        for i in range(n_events):
            events.append(make_match_row(
                cursor, duration_per, magnitude, phase,
            ))
            cursor += duration_per
        return _build_session_from_dicts(events, phase)

    def test_empty_input_returns_empty(self):
        assert detect_phase_synchronized_groups([]) == []

    def test_single_phase_no_multi_phase(self):
        """Sessions on only one phase should produce no multi-phase groups."""
        s1 = self._make_session(0, 30, 'w1')
        s2 = self._make_session(100, 130, 'w1')
        result = detect_phase_synchronized_groups([s1, s2])
        assert len(result) == 0

    def test_overlapping_sessions_on_two_phases(self):
        """Overlapping sessions on two phases should be grouped."""
        s1 = self._make_session(0, 60, 'w1')
        s2 = self._make_session(5, 55, 'w2')
        result = detect_phase_synchronized_groups([s1, s2])
        assert len(result) == 1
        assert isinstance(result[0], MultiPhaseSession)
        assert len(result[0].phases) == 2

    def test_non_overlapping_sessions_not_grouped(self):
        """Non-overlapping sessions on different phases should not be grouped."""
        s1 = self._make_session(0, 30, 'w1')
        s2 = self._make_session(200, 230, 'w2')
        result = detect_phase_synchronized_groups([s1, s2])
        assert len(result) == 0

    def test_three_phase_overlap(self):
        """Overlapping sessions on all three phases."""
        s1 = self._make_session(0, 60, 'w1')
        s2 = self._make_session(2, 58, 'w2')
        s3 = self._make_session(5, 55, 'w3')
        result = detect_phase_synchronized_groups([s1, s2, s3])
        assert len(result) == 1
        assert len(result[0].phases) == 3

    def test_multi_phase_session_start_end(self):
        """MultiPhaseSession start/end should span all constituent sessions."""
        s1 = self._make_session(0, 60, 'w1')
        s2 = self._make_session(5, 65, 'w2')
        result = detect_phase_synchronized_groups([s1, s2])
        assert len(result) == 1
        mp = result[0]
        assert mp.start == min(s1.start, s2.start)
        assert mp.end == max(s1.end, s2.end)


# ============================================================================
# Tests: _build_session_from_dicts
# ============================================================================

class TestBuildSessionFromDicts:
    """Test Session construction from event dictionaries."""

    def test_basic_session_creation(self):
        events = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(15, 10, 1200, 'w1'),
        ]
        session = _build_session_from_dicts(events, 'w1')
        assert isinstance(session, Session)
        assert session.cycle_count == 2
        assert session.phase == 'w1'
        assert session.start == _ts(0)
        assert session.end == _ts(25)

    def test_magnitude_cv_computed(self):
        """CV should be computed from event magnitudes."""
        events = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(15, 10, 1000, 'w1'),
        ]
        session = _build_session_from_dicts(events, 'w1')
        assert session.magnitude_cv == 0.0  # identical

    def test_magnitude_cv_nonzero_for_varying_magnitudes(self):
        events = [
            make_match_row(0, 10, 1000, 'w1'),
            make_match_row(15, 10, 2000, 'w1'),
        ]
        session = _build_session_from_dicts(events, 'w1')
        assert session.magnitude_cv > 0

    def test_thresholds_collected(self):
        events = [
            make_match_row(0, 10, 1000, 'w1', threshold=2000),
            make_match_row(15, 10, 1200, 'w1', threshold=1500),
        ]
        session = _build_session_from_dicts(events, 'w1')
        assert 2000 in session.thresholds
        assert 1500 in session.thresholds
