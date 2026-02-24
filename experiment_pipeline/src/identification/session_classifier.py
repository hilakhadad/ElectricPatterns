"""
Event-level device classifier (classify-first, group-second).

Priority-based classification that identifies devices BEFORE grouping:
  1. **Boiler** -- single long high-power isolated event (event-level)
     Multi-phase check: similar events on other phases -> three_phase_device
  2. **AC** -- cycling patterns detected from remaining events
     Cross-phase check: overlapping AC patterns -> central_ac vs regular_ac
  2.5 **Recurring pattern** -- per-house per-phase DBSCAN clustering of
     remaining sessions.  Discovers consistent behavioural patterns among
     unclassified events (NILM "discovered load").
  3. **Unknown** -- everything else (grouped by time proximity)

Implementation is split across classifiers/ sub-package:
  - classifiers/boiler_classifier.py  -- boiler + three-phase identification
  - classifiers/ac_classifier.py      -- AC cycling pattern detection
  - classifiers/central_ac_classifier.py -- cross-phase AC overlap
  - classifiers/recurring_pattern_classifier.py -- per-house pattern discovery
  - classifiers/unknown_classifier.py -- unknown confidence/reason
  - classifiers/scoring_utils.py      -- shared scoring functions
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Union

import pandas as pd

from .session_grouper import Session, MultiPhaseSession

# Re-export from classifiers sub-package
from .classifiers.boiler_classifier import (
    _identify_boiler_events,
    _is_boiler_candidate_event,
    _find_simultaneous_long_events,
    _enforce_boiler_phase_exclusivity,
    _three_phase_confidence,
    _has_nearby_compressor_cycles,
    _boiler_confidence,
)
from .classifiers.ac_classifier import (
    _identify_ac_sessions,
    _find_cycling_sequences,
    _build_session_from_events,
    _session_ac_likeness,
    _find_anchor_duration,
    _regular_ac_confidence,
    _find_event_indices,
    _is_regular_ac_session,
)
from .classifiers.central_ac_classifier import (
    _check_central_ac_overlap,
    _is_central_ac_candidate,
    _has_ac_pattern_in_dominant_iteration,
    _central_ac_confidence,
)
from .classifiers.unknown_classifier import (
    _unknown_confidence,
    _unknown_reason,
)
from .classifiers.recurring_pattern_classifier import (
    _identify_recurring_patterns,
)
from .classifiers.scoring_utils import (
    _linear_score,
    _inverse_linear_score,
    _magnitude_monotonicity,
    _cycling_regularity,
    _min_distance_to_medium_events,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class ClassifiedSession:
    """A session with device classification attached."""
    session: Union[Session, MultiPhaseSession]
    device_type: str          # boiler | central_ac | regular_ac | recurring_pattern | unknown
    reason: str               # human-readable classification reason
    confidence: float = 0.0   # 0-1 overall confidence score
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Public API
# ============================================================================

def classify_events(
    filtered_matches: pd.DataFrame,
) -> Dict[str, List[ClassifiedSession]]:
    """Classify events using priority-based approach (classify-first, group-second).

    Pipeline:
        Step 1: Boiler identification (event-level, before any grouping)
                Multi-phase check -> three_phase_device
                Phase exclusivity for remaining boilers
        Step 2: AC identification (cycling pattern detection from remaining events)
                Cross-phase overlap -> central_ac vs regular_ac
        Step 2.5: Recurring pattern discovery (DBSCAN per-phase on remaining)
        Step 3: Unknown (leftover sessions that didn't cluster)

    Args:
        filtered_matches: DataFrame from filter_transient_events() with columns:
            on_start, off_end, on_magnitude, duration, phase, tag, iteration, threshold

    Returns:
        Dict mapping device_type -> list of ClassifiedSession.
        Keys: 'boiler', 'three_phase_device', 'central_ac', 'regular_ac',
              'recurring_pattern', 'unknown'
    """
    from .session_grouper import group_into_sessions

    result: Dict[str, List[ClassifiedSession]] = {
        'boiler': [],
        'three_phase_device': [],
        'central_ac': [],
        'regular_ac': [],
        'recurring_pattern': [],
        'unknown': [],
    }

    if filtered_matches.empty:
        return result

    # Ensure timestamps
    for col in ['on_start', 'off_end', 'off_start']:
        if col in filtered_matches.columns and not pd.api.types.is_datetime64_any_dtype(filtered_matches[col]):
            filtered_matches[col] = pd.to_datetime(filtered_matches[col])

    # --- Step 1: Boiler (event-level) -------------------------------------
    boiler_list, three_phase_list, remaining = _identify_boiler_events(filtered_matches)
    result['boiler'] = boiler_list
    result['three_phase_device'] = three_phase_list

    n_boiler = len(boiler_list)
    n_3phase = len(three_phase_list)
    if n_boiler or n_3phase:
        logger.info(
            f"  Step 1: {n_boiler} boiler + {n_3phase} three_phase_device "
            f"({len(filtered_matches) - len(remaining)}/{len(filtered_matches)} events consumed)"
        )

    # --- Step 2: AC (pattern-based grouping) ------------------------------
    central_list, regular_list, remaining = _identify_ac_sessions(remaining)
    result['central_ac'] = central_list
    result['regular_ac'] = regular_list

    n_central = len(central_list)
    n_regular = len(regular_list)
    if n_central or n_regular:
        logger.info(f"  Step 2: {n_central} central_ac + {n_regular} regular_ac")

    # --- Step 2.5: Recurring patterns (per-phase DBSCAN) ------------------
    recurring_list, leftover_sessions = _identify_recurring_patterns(
        remaining, filtered_matches
    )
    result['recurring_pattern'] = recurring_list
    n_recurring = len(recurring_list)

    # --- Step 3: Unknown (leftover sessions that didn't cluster) ----------
    for s in leftover_sessions:
        conf, breakdown = _unknown_confidence(s, filtered_matches)
        reason = _unknown_reason(s, breakdown)
        result['unknown'].append(ClassifiedSession(
            session=s,
            device_type='unknown',
            reason=reason,
            confidence=conf,
            confidence_breakdown=breakdown,
        ))

    n_unknown = len(result['unknown'])
    logger.info(
        f"Classification: {n_boiler} boiler, {n_3phase} three_phase, "
        f"{n_central} central_ac, {n_regular} regular_ac, "
        f"{n_recurring} recurring_pattern, {n_unknown} unknown"
    )

    return result


def classify_sessions(
    sessions: List[Session],
    all_matches: pd.DataFrame,
) -> Dict[str, List[ClassifiedSession]]:
    """Backward-compatible wrapper -- delegates to classify_events().

    .. deprecated:: Use :func:`classify_events` directly.
    """
    logger.info("classify_sessions() delegating to classify_events()")
    return classify_events(all_matches)
