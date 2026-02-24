"""
Unknown device classifier.

Computes confidence and reason for sessions that don't match any known
device type. The confidence score represents how confidently "truly unknown"
the session is (low = borderline, close to some pattern; high = truly unknown).
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..config import (
    BOILER_MIN_DURATION,
    BOILER_MIN_MAGNITUDE,
    BOILER_ISOLATION_WINDOW,
    AC_MIN_MAGNITUDE,
    AC_MIN_FOLLOWING_CYCLES,
    AC_MIN_INITIAL_DURATION,
    AC_MAX_MAGNITUDE_CV,
)
from ..session_grouper import Session
from .scoring_utils import (
    _linear_score,
    _inverse_linear_score,
    _min_distance_to_medium_events,
)
from .ac_classifier import _find_anchor_duration

logger = logging.getLogger(__name__)


def _unknown_confidence(
    session: Session,
    all_matches: pd.DataFrame,
) -> tuple:
    """Compute two exclusion confidences for an unknown session.

    For each known device type, scores how confident we are that the session
    is **not** that type.  Low ``not_X`` -> the session looks like X, might
    be misclassified.  High ``not_X`` -> definitely not X.

    Returns:
        (confidence, breakdown) where:
        - confidence = min(not_boiler, not_ac) -- how confidently *unknown*
          (low = borderline, close to some pattern; high = truly unknown).
        - breakdown contains ``not_boiler``, ``not_ac`` and per-criterion
          detail scores.
    """
    breakdown = {}

    # --- Boiler proximity (how boiler-like is it, 0-1) ---
    b_dur = round(_linear_score(
        session.avg_cycle_duration, 0, BOILER_MIN_DURATION
    ), 2)
    b_mag = round(_linear_score(
        session.avg_magnitude, 0, BOILER_MIN_MAGNITUDE
    ), 2)
    if session.cycle_count <= 2:
        b_cycles = 1.0
    elif session.cycle_count <= 4:
        b_cycles = 0.5
    else:
        b_cycles = max(0.0, 1.0 - session.cycle_count * 0.1)

    min_dist = _min_distance_to_medium_events(session, all_matches)
    if min_dist is None:
        b_isolation = 1.0
    else:
        b_isolation = round(_linear_score(
            min_dist, 0, BOILER_ISOLATION_WINDOW
        ), 2)

    boiler_proximity = float(np.mean([b_dur, b_mag, b_cycles, b_isolation]))
    not_boiler = round(1.0 - boiler_proximity, 2)

    breakdown['not_boiler'] = not_boiler
    breakdown['boiler_dur'] = b_dur
    breakdown['boiler_mag'] = b_mag
    breakdown['boiler_cycles'] = round(b_cycles, 2)
    breakdown['boiler_isolation'] = b_isolation

    # --- Regular AC proximity (how AC-like is it, 0-1) ---
    a_cycles = round(_linear_score(
        session.cycle_count, 1, 1 + AC_MIN_FOLLOWING_CYCLES
    ), 2)
    a_mag = round(_linear_score(
        session.avg_magnitude, 0, AC_MIN_MAGNITUDE
    ), 2)
    anchor_dur = _find_anchor_duration(session)
    a_initial = round(_linear_score(
        anchor_dur, 0, AC_MIN_INITIAL_DURATION
    ), 2)
    if session.magnitude_cv is not None and session.magnitude_cv >= 0:
        a_cv = round(_inverse_linear_score(
            session.magnitude_cv, 0, AC_MAX_MAGNITUDE_CV * 2
        ), 2)
    else:
        a_cv = 0.5

    ac_proximity = float(np.mean([a_cycles, a_mag, a_initial, a_cv]))
    not_ac = round(1.0 - ac_proximity, 2)

    breakdown['not_ac'] = not_ac
    breakdown['ac_cycles'] = a_cycles
    breakdown['ac_mag'] = a_mag
    breakdown['ac_initial_dur'] = a_initial
    breakdown['ac_cv'] = a_cv

    # Overall: how confidently "truly unknown" (low = borderline)
    confidence = round(min(not_boiler, not_ac), 2)
    return confidence, breakdown


def _unknown_reason(session: Session, breakdown: dict) -> str:
    """Generate descriptive reason for unknown classification.

    Shows both exclusion confidences and which criteria failed for the
    nearest device type.
    """
    not_boiler = breakdown.get('not_boiler', 1.0)
    not_ac = breakdown.get('not_ac', 1.0)

    parts = []

    # Boiler side
    boiler_fails = []
    if session.avg_cycle_duration < BOILER_MIN_DURATION:
        boiler_fails.append(f"dur={session.avg_cycle_duration:.0f}/{BOILER_MIN_DURATION}min")
    if session.avg_magnitude < BOILER_MIN_MAGNITUDE:
        boiler_fails.append(f"mag={session.avg_magnitude:.0f}/{BOILER_MIN_MAGNITUDE}W")
    if session.cycle_count > 2:
        boiler_fails.append(f"cycles={session.cycle_count} (max 2)")
    boiler_str = f"not_boiler: {not_boiler:.0%}"
    if boiler_fails:
        boiler_str += f" ({', '.join(boiler_fails)})"
    parts.append(boiler_str)

    # AC side
    ac_fails = []
    if session.cycle_count < (1 + AC_MIN_FOLLOWING_CYCLES):
        ac_fails.append(f"cycles={session.cycle_count} (min {1 + AC_MIN_FOLLOWING_CYCLES})")
    if session.avg_magnitude < AC_MIN_MAGNITUDE:
        ac_fails.append(f"mag={session.avg_magnitude:.0f}/{AC_MIN_MAGNITUDE}W")
    anchor_dur = _find_anchor_duration(session)
    if anchor_dur < AC_MIN_INITIAL_DURATION:
        ac_fails.append(f"init_dur={anchor_dur:.0f}/{AC_MIN_INITIAL_DURATION}min")
    if session.magnitude_cv > AC_MAX_MAGNITUDE_CV:
        ac_fails.append(f"CV={session.magnitude_cv:.2f} (max {AC_MAX_MAGNITUDE_CV})")
    ac_str = f"not_ac: {not_ac:.0%}"
    if ac_fails:
        ac_str += f" ({', '.join(ac_fails)})"
    parts.append(ac_str)

    return ' | '.join(parts)
