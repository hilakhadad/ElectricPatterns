"""
Unit tests for identification.session_classifier — device classification logic.

Tests the classify-first, group-second architecture:
  1. Boiler identification (event-level, before grouping)
  2. AC identification (cycling pattern detection)
  3. Central AC (cross-phase overlap of independently-detected AC patterns)
  4. Unknown (everything else)
  5. Phase exclusivity for boilers

Uses synthetic DataFrames with realistic column structure matching
the output of filter_transient_events().
"""
import pytest
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from identification.session_classifier import (
    classify_events,
    ClassifiedSession,
    _identify_boiler_events,
    _is_boiler_candidate_event,
    _enforce_boiler_phase_exclusivity,
    _identify_ac_sessions,
    _is_regular_ac_session,
    _is_central_ac_candidate,
    _linear_score,
    _inverse_linear_score,
    _boiler_confidence,
    _regular_ac_confidence,
    _magnitude_monotonicity,
    _cycling_regularity,
)
from identification.session_grouper import Session, MultiPhaseSession, build_single_event_session
from identification.config import (
    BOILER_MIN_DURATION,
    BOILER_MIN_MAGNITUDE,
    BOILER_ISOLATION_WINDOW,
    AC_MIN_MAGNITUDE,
    AC_MIN_CYCLE_DURATION,
    AC_MAX_CYCLE_DURATION,
    AC_MIN_INITIAL_DURATION,
    AC_MIN_FOLLOWING_CYCLES,
    AC_MAX_MAGNITUDE_CV,
    DEFAULT_SESSION_GAP_MINUTES,
    CENTRAL_AC_SYNC_TOLERANCE,
    MIN_EVENT_DURATION_MINUTES,
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


def _make_boiler_event(on_start_min=0, duration=30, magnitude=2000,
                       phase='w1', **kwargs):
    """Create a single boiler-like match row."""
    return make_match_row(on_start_min, duration, magnitude, phase, **kwargs)


def _make_ac_cycling_events(
    start_min=0, phase='w1', initial_duration=15,
    cycle_duration=10, n_cycles=5, magnitude=1200,
    gap_between=5, iteration=0, threshold=2000,
):
    """Create a sequence of AC-like cycling events.

    Returns list of match row dicts: one initial long event + n_cycles short cycles.
    """
    events = []
    # Initial long activation
    events.append(make_match_row(
        start_min, initial_duration, magnitude, phase,
        iteration=iteration, threshold=threshold,
    ))
    cursor = start_min + initial_duration + gap_between
    for i in range(n_cycles):
        events.append(make_match_row(
            cursor, cycle_duration, magnitude, phase,
            iteration=iteration, threshold=threshold,
        ))
        cursor += cycle_duration + gap_between
    return events


# ============================================================================
# Tests: classify_events() — main entry point
# ============================================================================

class TestClassifyEventsEntryPoint:
    """Test the public classify_events() function."""

    def test_empty_input_returns_empty_result(self):
        result = classify_events(pd.DataFrame())
        assert isinstance(result, dict)
        for key in ['boiler', 'three_phase_device', 'central_ac', 'regular_ac', 'recurring_pattern', 'unknown']:
            assert key in result
            assert result[key] == []

    def test_returns_all_device_type_keys(self):
        row = _make_boiler_event(magnitude=2500, duration=40)
        df = make_matches_df([row])
        result = classify_events(df)
        expected_keys = {'boiler', 'three_phase_device', 'central_ac', 'regular_ac', 'recurring_pattern', 'unknown'}
        assert set(result.keys()) == expected_keys

    def test_single_boiler_event_classified(self):
        """A single isolated high-power long event should be classified as boiler."""
        row = _make_boiler_event(magnitude=2500, duration=40, phase='w1')
        df = make_matches_df([row])
        result = classify_events(df)
        assert len(result['boiler']) == 1
        cs = result['boiler'][0]
        assert isinstance(cs, ClassifiedSession)
        assert cs.device_type == 'boiler'

    def test_classified_sessions_have_confidence(self):
        """All classified sessions should have a confidence score between 0 and 1."""
        row = _make_boiler_event(magnitude=2500, duration=40)
        df = make_matches_df([row])
        result = classify_events(df)
        for device_type, sessions in result.items():
            for cs in sessions:
                assert 0 <= cs.confidence <= 1, (
                    f"{device_type} session has confidence {cs.confidence} outside [0,1]"
                )

    def test_unknown_for_unclassifiable_event(self):
        """An event that doesn't match any pattern should be classified as unknown."""
        # Short duration, low magnitude, single event -> not boiler, not AC
        row = make_match_row(0, 5, 500, 'w1')
        df = make_matches_df([row])
        result = classify_events(df)
        assert len(result['unknown']) >= 1
        assert len(result['boiler']) == 0
        assert len(result['regular_ac']) == 0


# ============================================================================
# Tests: Boiler identification
# ============================================================================

class TestBoilerIdentification:
    """Test _identify_boiler_events() and helper functions."""

    def test_typical_boiler_classified(self):
        """Event with duration>=15min, magnitude>=1500W, isolated -> boiler."""
        row = _make_boiler_event(duration=30, magnitude=2000)
        df = make_matches_df([row])
        boilers, three_phase, remaining = _identify_boiler_events(df)
        assert len(boilers) == 1
        assert boilers[0].device_type == 'boiler'
        assert len(remaining) == 0

    def test_short_event_not_boiler(self):
        """Event with duration < BOILER_MIN_DURATION should NOT be boiler."""
        row = make_match_row(0, BOILER_MIN_DURATION - 1, 2000, 'w1')
        df = make_matches_df([row])
        boilers, _, remaining = _identify_boiler_events(df)
        assert len(boilers) == 0
        assert len(remaining) == 1

    def test_low_magnitude_not_boiler(self):
        """Event with magnitude < BOILER_MIN_MAGNITUDE should NOT be boiler."""
        row = make_match_row(0, 30, BOILER_MIN_MAGNITUDE - 100, 'w1')
        df = make_matches_df([row])
        boilers, _, remaining = _identify_boiler_events(df)
        assert len(boilers) == 0
        assert len(remaining) == 1

    def test_boiler_at_exact_thresholds(self):
        """Event at exactly BOILER_MIN_DURATION and BOILER_MIN_MAGNITUDE should be classified."""
        row = make_match_row(0, BOILER_MIN_DURATION, BOILER_MIN_MAGNITUDE, 'w1')
        df = make_matches_df([row])
        boilers, _, remaining = _identify_boiler_events(df)
        assert len(boilers) == 1

    def test_boiler_not_isolated_near_compressor_cycles(self):
        """A boiler candidate near AC-like compressor cycles should NOT be boiler."""
        rows = [
            # Boiler candidate
            _make_boiler_event(on_start_min=0, duration=30, magnitude=2000, phase='w1'),
            # Nearby medium-duration events (compressor cycles) on same phase
            make_match_row(35, 10, 1200, 'w1'),
            make_match_row(50, 10, 1200, 'w1'),
            make_match_row(65, 10, 1200, 'w1'),
        ]
        df = make_matches_df(rows)
        boilers, _, remaining = _identify_boiler_events(df)
        # The boiler candidate should be rejected because of nearby cycling
        assert len(boilers) == 0

    def test_two_consecutive_boilers_not_disqualify_each_other(self):
        """Two consecutive boiler activations should not disqualify each other as compressor cycles."""
        rows = [
            _make_boiler_event(on_start_min=0, duration=30, magnitude=2000, phase='w1'),
            _make_boiler_event(on_start_min=90, duration=25, magnitude=1800, phase='w1'),
        ]
        df = make_matches_df(rows)
        boilers, _, remaining = _identify_boiler_events(df)
        assert len(boilers) == 2

    def test_boiler_removed_from_remaining(self):
        """Classified boiler events should not appear in remaining."""
        rows = [
            _make_boiler_event(on_start_min=0, duration=30, magnitude=2000, phase='w1'),
            make_match_row(200, 5, 500, 'w1'),  # unrelated short event
        ]
        df = make_matches_df(rows)
        boilers, _, remaining = _identify_boiler_events(df)
        assert len(boilers) == 1
        assert len(remaining) == 1
        # The remaining event should be the short one
        assert remaining.iloc[0]['duration'] == 5


# ============================================================================
# Tests: Boiler phase exclusivity
# ============================================================================

class TestBoilerPhaseExclusivity:
    """Test _enforce_boiler_phase_exclusivity()."""

    def _make_classified_boiler(self, phase='w1'):
        """Create a minimal ClassifiedSession for a boiler on the given phase."""
        event = _make_boiler_event(phase=phase, duration=30, magnitude=2000)
        session = build_single_event_session(event, phase)
        return ClassifiedSession(
            session=session,
            device_type='boiler',
            reason='test',
            confidence=0.8,
        )

    def test_single_phase_kept_unchanged(self):
        """If all boilers on one phase, nothing is demoted."""
        sessions = [self._make_classified_boiler('w1'), self._make_classified_boiler('w1')]
        kept, demoted = _enforce_boiler_phase_exclusivity(sessions)
        assert len(kept) == 2
        assert len(demoted) == 0

    def test_multi_phase_demotes_minority(self):
        """If boilers on multiple phases, minority phase is demoted."""
        sessions = [
            self._make_classified_boiler('w1'),
            self._make_classified_boiler('w1'),
            self._make_classified_boiler('w1'),
            self._make_classified_boiler('w2'),
        ]
        kept, demoted = _enforce_boiler_phase_exclusivity(sessions)
        assert len(kept) == 3
        assert len(demoted) == 1
        assert all(cs.session.phase == 'w1' for cs in kept)
        assert demoted[0].session.phase == 'w2'

    def test_empty_input_returns_empty(self):
        kept, demoted = _enforce_boiler_phase_exclusivity([])
        assert kept == []
        assert demoted == []


# ============================================================================
# Tests: Three-phase device detection
# ============================================================================

class TestThreePhaseDetection:
    """Test that simultaneous long events on multiple phases are classified as three_phase_device."""

    def test_simultaneous_events_on_three_phases(self):
        """Long events overlapping on 3 phases -> three_phase_device."""
        rows = [
            make_match_row(0, 30, 2000, 'w1'),
            make_match_row(2, 28, 1900, 'w2'),
            make_match_row(1, 29, 2100, 'w3'),
        ]
        df = make_matches_df(rows)
        boilers, three_phase, remaining = _identify_boiler_events(df)
        assert len(three_phase) >= 1
        tp = three_phase[0]
        assert tp.device_type == 'three_phase_device'
        assert len(tp.session.phases) >= 2

    def test_non_overlapping_phases_not_three_phase(self):
        """Long events on different phases but not overlapping -> NOT three_phase."""
        rows = [
            make_match_row(0, 30, 2000, 'w1'),
            make_match_row(200, 30, 2000, 'w2'),  # far apart in time
        ]
        df = make_matches_df(rows)
        boilers, three_phase, remaining = _identify_boiler_events(df)
        # They should be individual boilers, not three_phase_device
        assert len(three_phase) == 0

    def test_very_different_durations_not_three_phase(self):
        """Overlapping events with very different durations -> NOT three_phase."""
        rows = [
            make_match_row(0, 30, 2000, 'w1'),
            make_match_row(0, 100, 2000, 'w2'),  # 3.3x ratio > 2.0 limit
        ]
        df = make_matches_df(rows)
        boilers, three_phase, remaining = _identify_boiler_events(df)
        assert len(three_phase) == 0


# ============================================================================
# Tests: AC identification
# ============================================================================

class TestACIdentification:
    """Test AC cycling pattern detection."""

    def test_typical_ac_pattern_detected(self):
        """Standard AC cycling (1 anchor + 3+ short cycles) should be regular_ac."""
        events = _make_ac_cycling_events(
            start_min=0, phase='w1',
            initial_duration=15, cycle_duration=10,
            n_cycles=4, magnitude=1200, gap_between=5,
        )
        df = make_matches_df(events)
        central, regular, remaining = _identify_ac_sessions(df)
        assert len(regular) >= 1
        assert regular[0].device_type == 'regular_ac'

    def test_too_few_cycles_not_ac(self):
        """Fewer than AC_MIN_FOLLOWING_CYCLES following cycles -> not AC."""
        # 1 anchor + 1 cycle = not enough
        events = _make_ac_cycling_events(
            n_cycles=1, magnitude=1200, gap_between=5,
        )
        df = make_matches_df(events)
        central, regular, remaining = _identify_ac_sessions(df)
        assert len(regular) == 0

    def test_low_magnitude_not_ac(self):
        """Events below AC_MIN_MAGNITUDE should not be AC."""
        events = _make_ac_cycling_events(
            magnitude=AC_MIN_MAGNITUDE - 100,
            n_cycles=5,
        )
        df = make_matches_df(events)
        central, regular, remaining = _identify_ac_sessions(df)
        assert len(regular) == 0

    def test_ac_events_removed_from_remaining(self):
        """Events classified as AC should not appear in remaining."""
        ac_events = _make_ac_cycling_events(
            start_min=0, phase='w1', n_cycles=4, magnitude=1200,
        )
        other = make_match_row(500, 5, 500, 'w2')
        df = make_matches_df(ac_events + [other])
        central, regular, remaining = _identify_ac_sessions(df)
        assert len(regular) >= 1
        # The 'other' event should still be in remaining
        assert len(remaining) >= 1

    def test_ac_on_different_phases_independent(self):
        """AC patterns on different phases should be detected independently."""
        w1_events = _make_ac_cycling_events(
            start_min=0, phase='w1', n_cycles=4, magnitude=1200,
        )
        w2_events = _make_ac_cycling_events(
            start_min=500, phase='w2', n_cycles=4, magnitude=1100,
        )
        df = make_matches_df(w1_events + w2_events)
        central, regular, remaining = _identify_ac_sessions(df)
        assert len(regular) >= 2


# ============================================================================
# Tests: Central AC detection
# ============================================================================

class TestCentralACDetection:
    """Test that overlapping AC patterns on multiple phases are classified as central_ac."""

    def test_overlapping_ac_on_two_phases_is_central(self):
        """AC cycling on 2 phases overlapping in time -> central_ac."""
        w1_events = _make_ac_cycling_events(
            start_min=0, phase='w1', n_cycles=4, magnitude=1200,
        )
        w2_events = _make_ac_cycling_events(
            start_min=5, phase='w2', n_cycles=4, magnitude=1100,
        )
        df = make_matches_df(w1_events + w2_events)
        central, regular, remaining = _identify_ac_sessions(df)
        assert len(central) >= 1
        assert central[0].device_type == 'central_ac'

    def test_non_overlapping_ac_not_central(self):
        """AC patterns on different phases with no time overlap -> regular_ac, not central."""
        w1_events = _make_ac_cycling_events(
            start_min=0, phase='w1', n_cycles=4, magnitude=1200,
        )
        # Start W2 events far away so no overlap
        w2_events = _make_ac_cycling_events(
            start_min=1000, phase='w2', n_cycles=4, magnitude=1100,
        )
        df = make_matches_df(w1_events + w2_events)
        central, regular, remaining = _identify_ac_sessions(df)
        assert len(central) == 0
        assert len(regular) >= 2

    def test_single_phase_ac_not_central(self):
        """AC on only one phase should be regular_ac, never central_ac."""
        events = _make_ac_cycling_events(
            start_min=0, phase='w1', n_cycles=4, magnitude=1200,
        )
        df = make_matches_df(events)
        central, regular, remaining = _identify_ac_sessions(df)
        assert len(central) == 0
        assert len(regular) >= 1


# ============================================================================
# Tests: Unknown classification
# ============================================================================

class TestUnknownClassification:
    """Test that events not matching any pattern are classified as unknown."""

    def test_short_low_power_event_is_unknown(self):
        """Short, low-power event that doesn't match any device type."""
        row = make_match_row(0, 5, 500, 'w1')
        df = make_matches_df([row])
        result = classify_events(df)
        assert len(result['unknown']) >= 1
        assert len(result['boiler']) == 0
        assert len(result['regular_ac']) == 0
        assert len(result['central_ac']) == 0

    def test_unknown_has_exclusion_confidence(self):
        """Unknown sessions should have not_boiler and not_ac in their breakdown."""
        row = make_match_row(0, 5, 500, 'w1')
        df = make_matches_df([row])
        result = classify_events(df)
        assert len(result['unknown']) >= 1
        cs = result['unknown'][0]
        assert 'not_boiler' in cs.confidence_breakdown
        assert 'not_ac' in cs.confidence_breakdown

    def test_medium_event_below_boiler_thresholds_is_unknown(self):
        """Event with decent duration but below boiler magnitude -> unknown."""
        row = make_match_row(0, 20, 1000, 'w1')  # duration OK but magnitude < 1500
        df = make_matches_df([row])
        result = classify_events(df)
        assert len(result['boiler']) == 0
        assert len(result['unknown']) >= 1


# ============================================================================
# Tests: _is_boiler_candidate_event
# ============================================================================

class TestIsBoilerCandidateEvent:
    """Test the individual boiler candidate check."""

    def test_qualifies_when_isolated(self):
        """Event meeting all criteria and isolated -> candidate."""
        row_dict = _make_boiler_event(duration=30, magnitude=2000)
        row = pd.Series(row_dict)
        pool = make_matches_df([row_dict])
        assert _is_boiler_candidate_event(row, pool) is True

    def test_rejects_short_duration(self):
        row_dict = make_match_row(0, BOILER_MIN_DURATION - 1, 2000, 'w1')
        row = pd.Series(row_dict)
        pool = make_matches_df([row_dict])
        assert _is_boiler_candidate_event(row, pool) is False

    def test_rejects_low_magnitude(self):
        row_dict = make_match_row(0, 30, BOILER_MIN_MAGNITUDE - 1, 'w1')
        row = pd.Series(row_dict)
        pool = make_matches_df([row_dict])
        assert _is_boiler_candidate_event(row, pool) is False


# ============================================================================
# Tests: _is_regular_ac_session
# ============================================================================

class TestIsRegularACSession:
    """Test the AC session validation helper."""

    def _make_ac_session(self, n_events=5, magnitude=1200, phase='w1',
                         initial_duration=15, cycle_duration=10, gap=5,
                         iteration=0, threshold=2000):
        """Build a Session object from synthetic AC cycling events."""
        events = _make_ac_cycling_events(
            start_min=0, phase=phase,
            initial_duration=initial_duration,
            cycle_duration=cycle_duration,
            n_cycles=n_events - 1,  # n_events includes anchor
            magnitude=magnitude,
            gap_between=gap,
            iteration=iteration,
            threshold=threshold,
        )
        from identification.session_grouper import _build_session_from_dicts
        return _build_session_from_dicts(events, phase)

    def test_valid_ac_session(self):
        """Session with cycling pattern, high magnitude, low CV -> AC."""
        session = self._make_ac_session(n_events=5, magnitude=1200)
        assert _is_regular_ac_session(session) is True

    def test_rejects_low_magnitude(self):
        session = self._make_ac_session(n_events=5, magnitude=AC_MIN_MAGNITUDE - 100)
        assert _is_regular_ac_session(session) is False

    def test_rejects_too_few_cycles(self):
        """Session with fewer than required cycles -> not AC."""
        session = self._make_ac_session(n_events=2, magnitude=1200)
        assert _is_regular_ac_session(session) is False


# ============================================================================
# Tests: Confidence scoring helpers
# ============================================================================

class TestLinearScore:
    """Test _linear_score and _inverse_linear_score."""

    def test_below_low_returns_min(self):
        assert _linear_score(0, 10, 20) == 0.5

    def test_above_high_returns_max(self):
        assert _linear_score(30, 10, 20) == 1.0

    def test_midpoint_returns_midvalue(self):
        result = _linear_score(15, 10, 20)
        assert abs(result - 0.75) < 0.01

    def test_inverse_below_low_returns_max(self):
        assert _inverse_linear_score(0, 0, 10) == 1.0

    def test_inverse_above_high_returns_min(self):
        assert _inverse_linear_score(15, 0, 10) == 0.5

    def test_inverse_midpoint(self):
        result = _inverse_linear_score(5, 0, 10)
        assert abs(result - 0.75) < 0.01


class TestBoilerConfidence:
    """Test boiler confidence scoring."""

    def test_high_confidence_for_ideal_boiler(self):
        """Long duration, high magnitude, single event, isolated -> high confidence."""
        event = _make_boiler_event(duration=50, magnitude=3000)
        session = build_single_event_session(event, 'w1')
        all_matches = make_matches_df([event])
        conf, breakdown = _boiler_confidence(session, all_matches)
        assert conf >= 0.8
        assert 'duration' in breakdown
        assert 'magnitude' in breakdown
        assert 'isolation' in breakdown

    def test_lower_confidence_near_minimum(self):
        """Event at minimum thresholds -> lower confidence."""
        event = _make_boiler_event(
            duration=BOILER_MIN_DURATION, magnitude=BOILER_MIN_MAGNITUDE
        )
        session = build_single_event_session(event, 'w1')
        all_matches = make_matches_df([event])
        conf, breakdown = _boiler_confidence(session, all_matches)
        # Should be lower than ideal
        assert conf < 0.9


class TestRegularACConfidence:
    """Test regular AC confidence scoring."""

    def test_has_expected_breakdown_keys(self):
        events = _make_ac_cycling_events(
            n_cycles=5, magnitude=1200,
        )
        from identification.session_grouper import _build_session_from_dicts
        session = _build_session_from_dicts(events, 'w1')
        conf, breakdown = _regular_ac_confidence(session)
        assert 'cycle_count' in breakdown
        assert 'magnitude_cv' in breakdown
        assert 'magnitude' in breakdown
        assert 0 <= conf <= 1


# ============================================================================
# Tests: Magnitude monotonicity
# ============================================================================

class TestMagnitudeMonotonicity:
    """Test _magnitude_monotonicity()."""

    def test_perfectly_monotonic_increasing(self):
        """Magnitudes that only increase -> monotonicity = 1.0."""
        events = [
            {'on_start': _ts(i * 15), 'on_magnitude': 1000 + i * 100}
            for i in range(5)
        ]
        session = Session(
            session_id='test', phase='w1', events=events,
            start=_ts(0), end=_ts(60), cycle_count=5,
        )
        mono = _magnitude_monotonicity(session)
        assert mono == 1.0

    def test_fewer_than_3_events_returns_1(self):
        """With < 3 events, not enough data to penalize -> returns 1.0."""
        events = [
            {'on_start': _ts(0), 'on_magnitude': 1000},
            {'on_start': _ts(15), 'on_magnitude': 1200},
        ]
        session = Session(
            session_id='test', phase='w1', events=events,
            start=_ts(0), end=_ts(15), cycle_count=2,
        )
        assert _magnitude_monotonicity(session) == 1.0


# ============================================================================
# Tests: Cycling regularity
# ============================================================================

class TestCyclingRegularity:
    """Test _cycling_regularity()."""

    def test_regular_cycling_low_cv(self):
        """Events with identical durations and gaps -> low CVs."""
        events = []
        cursor = 0
        for i in range(5):
            events.append({
                'on_start': _ts(cursor),
                'off_end': _ts(cursor + 10),
                'duration': 10,
            })
            cursor += 15  # 10 min duration + 5 min gap
        session = Session(
            session_id='test', phase='w1', events=events,
            start=_ts(0), end=_ts(70), cycle_count=5,
        )
        dur_cv, gap_cv = _cycling_regularity(session)
        assert dur_cv == 0.0  # all same duration
        assert gap_cv == 0.0  # all same gap

    def test_fewer_than_3_events_returns_zero(self):
        """With < 3 events, cannot compute CVs -> returns (0, 0)."""
        events = [
            {'on_start': _ts(0), 'off_end': _ts(10), 'duration': 10},
        ]
        session = Session(
            session_id='test', phase='w1', events=events,
            start=_ts(0), end=_ts(10), cycle_count=1,
        )
        dur_cv, gap_cv = _cycling_regularity(session)
        assert dur_cv == 0.0
        assert gap_cv == 0.0


# ============================================================================
# Tests: End-to-end integration
# ============================================================================

class TestClassifyEventsIntegration:
    """End-to-end tests combining boiler + AC + unknown."""

    def test_mixed_devices_classified_correctly(self):
        """A mix of boiler events, AC cycling, and unclassifiable events."""
        boiler = _make_boiler_event(
            on_start_min=0, duration=30, magnitude=2000, phase='w2',
        )
        # AC cycling on w1 far from boiler
        ac_events = _make_ac_cycling_events(
            start_min=500, phase='w1', n_cycles=4, magnitude=1200,
        )
        # Short unknown event on w3
        unknown = make_match_row(1000, 5, 500, 'w3')

        all_rows = [boiler] + ac_events + [unknown]
        df = make_matches_df(all_rows)
        result = classify_events(df)

        # Should have at least one boiler, at least one AC, and at least one unknown
        assert len(result['boiler']) >= 1
        assert len(result['regular_ac']) >= 1
        assert len(result['unknown']) >= 1

    def test_no_event_double_classified(self):
        """Every event should appear in exactly one classification category."""
        boiler = _make_boiler_event(
            on_start_min=0, duration=30, magnitude=2000, phase='w2',
        )
        ac_events = _make_ac_cycling_events(
            start_min=500, phase='w1', n_cycles=4, magnitude=1200,
        )
        unknown = make_match_row(1000, 5, 500, 'w3')

        all_rows = [boiler] + ac_events + [unknown]
        df = make_matches_df(all_rows)
        result = classify_events(df)

        # Count total classified events across all categories
        total_classified = 0
        for dtype, sessions in result.items():
            for cs in sessions:
                session = cs.session
                if isinstance(session, MultiPhaseSession):
                    for ps in session.phase_sessions.values():
                        total_classified += len(ps.events)
                else:
                    total_classified += len(session.events)

        assert total_classified == len(all_rows), (
            f"Expected {len(all_rows)} events total across all categories, "
            f"got {total_classified}"
        )
