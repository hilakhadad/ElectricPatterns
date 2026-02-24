"""
Session grouper -- loads matched events from all iterations, filters
transient noise, and groups events into device usage sessions.

A *session* is a contiguous cluster of matched ON->OFF pairs on the same
phase that are separated by less than ``session_gap`` minutes.  For
central AC, sessions overlapping across 2+ phases are merged into
multi-phase sessions.

Implementation is split across:
  - session_grouper.py (this file) -- data structures + public API
  - session_builder.py -- _build_session, _build_session_from_dicts, build_single_event_session
  - spike_stats.py -- _empty_spike_stats, _compute_spike_stats
"""
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    MIN_EVENT_DURATION_MINUTES,
    DEFAULT_SESSION_GAP_MINUTES,
    CENTRAL_AC_SYNC_TOLERANCE,
    PHASES,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class Session:
    """A group of related matched events on a single phase."""
    session_id: str
    phase: str
    events: List[dict] = field(repr=False)
    start: pd.Timestamp = None
    end: pd.Timestamp = None
    cycle_count: int = 0
    total_duration_minutes: float = 0.0
    avg_magnitude: float = 0.0
    max_magnitude: float = 0.0
    magnitude_cv: float = 0.0
    avg_cycle_duration: float = 0.0
    thresholds: List[int] = field(default_factory=list)


@dataclass
class MultiPhaseSession:
    """Sessions synchronised across multiple phases (central AC candidate)."""
    session_id: str
    phases: List[str]
    phase_sessions: Dict[str, Session] = field(repr=False)
    start: pd.Timestamp = None
    end: pd.Timestamp = None
    total_magnitude: float = 0.0


# ============================================================================
# Public API
# ============================================================================

def load_all_matches(
    experiment_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
) -> pd.DataFrame:
    """Load ALL matched events from ALL iterations into one DataFrame.

    Reads ``matches/`` directories (not classification/) from each run.
    Adds ``iteration`` and ``threshold`` columns for provenance tracking.
    """
    all_dfs: List[pd.DataFrame] = []

    for run_number, threshold in enumerate(threshold_schedule):
        run_dir = _find_run_dir(experiment_dir, run_number, threshold)
        if run_dir is None:
            continue

        matches_dir = run_dir / f"house_{house_id}" / "matches"
        if not matches_dir.exists():
            continue

        for pkl_file in sorted(matches_dir.glob(f"matches_{house_id}_*.pkl")):
            try:
                df = pd.read_pickle(pkl_file)
                if df.empty:
                    continue
                df['iteration'] = run_number
                df['threshold'] = threshold
                all_dfs.append(df)
            except Exception as exc:
                logger.warning(f"Failed to load {pkl_file}: {exc}")

    # Load wave recovery matches from run_post/ (if any)
    post_dir = experiment_dir / "run_post" / f"house_{house_id}" / "matches"
    if post_dir.exists():
        for pkl_file in sorted(post_dir.glob(f"matches_{house_id}_*.pkl")):
            try:
                df = pd.read_pickle(pkl_file)
                if df.empty:
                    continue
                # Wave matches already have iteration/threshold from wave_recovery_step
                if 'iteration' not in df.columns:
                    df['iteration'] = len(threshold_schedule)
                if 'threshold' not in df.columns:
                    df['threshold'] = 0
                all_dfs.append(df)
            except Exception as exc:
                logger.warning(f"Failed to load wave match {pkl_file}: {exc}")

    if not all_dfs:
        logger.info(f"No match files found for house {house_id}")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    wave_count = combined['tag'].str.startswith('WAVE-').sum() if 'tag' in combined.columns else 0
    logger.info(f"Loaded {len(combined)} matches across {len(threshold_schedule)} iterations"
                f"{f' + {wave_count} wave matches' if wave_count > 0 else ''}")
    return combined


def filter_transient_events(
    all_matches: pd.DataFrame,
    min_duration: float = MIN_EVENT_DURATION_MINUTES,
) -> tuple:
    """Filter out transient (spike) events that are too short to classify.

    Events shorter than ``min_duration`` minutes (currently 2 min -- filters
    only 1-min spikes) represent transient noise from appliances not targeted
    by identification (microwave, oven, washing machine motor starts, etc.).

    Returns:
        (filtered_df, spike_stats) -- filtered matches and statistics about
        the removed transient events.
    """
    from .spike_stats import _empty_spike_stats, _compute_spike_stats

    if all_matches.empty:
        return all_matches, _empty_spike_stats()

    has_duration = 'duration' in all_matches.columns
    if has_duration:
        spike_mask = all_matches['duration'] < min_duration
    else:
        logger.warning("No 'duration' column -- skipping transient filter")
        return all_matches, _empty_spike_stats()

    spikes = all_matches[spike_mask]
    filtered = all_matches[~spike_mask].reset_index(drop=True)

    # Compute stats for reporting
    spike_stats = _compute_spike_stats(spikes, filtered, min_duration)

    if len(spikes) > 0:
        logger.info(
            f"Filtered {len(spikes)} transient events (<{min_duration} min), "
            f"kept {len(filtered)} events for identification"
        )

    return filtered, spike_stats


def group_into_sessions(
    matches: pd.DataFrame,
    gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES,
) -> List[Session]:
    """Group matched events into sessions based on time proximity.

    For each phase independently:
    1. Sort events by on_start
    2. If gap between off_end of event N and on_start of event N+1 < gap_minutes,
       they belong to the same session
    3. Create :class:`Session` objects with pre-computed statistics
    """
    from .session_builder import _build_session, _split_sessions_at_initial_run

    if matches.empty:
        return []

    # Ensure timestamps
    for col in ['on_start', 'off_end', 'off_start']:
        if col in matches.columns and not pd.api.types.is_datetime64_any_dtype(matches[col]):
            matches[col] = pd.to_datetime(matches[col])

    all_sessions: List[Session] = []

    for phase in PHASES:
        phase_df = matches[matches['phase'] == phase]
        if phase_df.empty:
            continue

        sorted_df = phase_df.sort_values('on_start')
        rows = list(sorted_df.itertuples(index=False))

        current_group: List = [rows[0]]

        for i in range(1, len(rows)):
            prev = current_group[-1]
            curr = rows[i]

            prev_end = getattr(prev, 'off_end', None) or getattr(prev, 'off_start', None) or prev.on_start
            gap = (curr.on_start - prev_end).total_seconds() / 60

            if gap > gap_minutes:
                all_sessions.append(_build_session(current_group, phase))
                current_group = [curr]
            else:
                current_group.append(curr)

        all_sessions.append(_build_session(current_group, phase))

    # Post-processing: split sessions where a long event follows short ones.
    all_sessions = _split_sessions_at_initial_run(all_sessions)

    logger.info(f"Grouped {len(matches)} matches into {len(all_sessions)} sessions")
    return all_sessions


def detect_phase_synchronized_groups(
    sessions: List[Session],
    sync_tolerance_minutes: int = CENTRAL_AC_SYNC_TOLERANCE,
) -> List[MultiPhaseSession]:
    """Find sessions overlapping across 2+ phases (central AC candidates).

    Two sessions on different phases are considered synchronised if their
    time ranges overlap (with tolerance).
    """
    if not sessions:
        return []

    # Index sessions by phase
    by_phase: Dict[str, List[Session]] = {}
    for s in sessions:
        by_phase.setdefault(s.phase, []).append(s)

    phases_present = [p for p in PHASES if p in by_phase]
    if len(phases_present) < 2:
        return []

    tolerance = pd.Timedelta(minutes=sync_tolerance_minutes)

    # Use first available phase as reference
    ref_phase = phases_present[0]
    used_sessions: set = set()
    multi_phase_sessions: List[MultiPhaseSession] = []

    for ref_session in by_phase[ref_phase]:
        if ref_session.session_id in used_sessions:
            continue

        synced: Dict[str, Session] = {ref_phase: ref_session}

        for other_phase in phases_present[1:]:
            for other_session in by_phase[other_phase]:
                if other_session.session_id in used_sessions:
                    continue
                # Check overlap with tolerance
                if (ref_session.start - tolerance <= other_session.end and
                        other_session.start - tolerance <= ref_session.end):
                    synced[other_phase] = other_session
                    break

        if len(synced) >= 2:
            for s in synced.values():
                used_sessions.add(s.session_id)

            mp = MultiPhaseSession(
                session_id=f"mp_{uuid.uuid4().hex[:8]}",
                phases=sorted(synced.keys()),
                phase_sessions=synced,
                start=min(s.start for s in synced.values()),
                end=max(s.end for s in synced.values()),
                total_magnitude=sum(s.avg_magnitude for s in synced.values()),
            )
            multi_phase_sessions.append(mp)

    if multi_phase_sessions:
        logger.info(f"Found {len(multi_phase_sessions)} multi-phase session groups")

    return multi_phase_sessions


# Re-export from session_builder for backward compatibility
from .session_builder import (
    _build_session,
    _build_session_from_dicts,
    build_single_event_session,
    _split_sessions_at_initial_run,
)
from .spike_stats import _empty_spike_stats, _compute_spike_stats


# ============================================================================
# Helpers
# ============================================================================

def _find_run_dir(experiment_dir: Path, run_number: int, threshold: int) -> Optional[Path]:
    """Find run directory -- supports ``run_N`` and ``run_N_thXXXX`` naming."""
    exact = experiment_dir / f"run_{run_number}_th{threshold}"
    if exact.exists():
        return exact

    plain = experiment_dir / f"run_{run_number}"
    if plain.exists():
        return plain

    for d in experiment_dir.glob(f"run_{run_number}_th*"):
        if d.is_dir():
            return d

    return None
