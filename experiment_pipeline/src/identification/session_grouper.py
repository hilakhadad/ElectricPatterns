"""
Session grouper — loads matched events from all iterations, deduplicates
cross-iteration duplicates, and groups events into device usage sessions.

A *session* is a contiguous cluster of matched ON→OFF pairs on the same
phase that are separated by less than ``session_gap`` minutes.  For
central AC, sessions overlapping across 2+ phases are merged into
multi-phase sessions.
"""
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import (
    DEDUP_TIME_TOLERANCE_MINUTES,
    DEDUP_MAGNITUDE_TOLERANCE_W,
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

    if not all_dfs:
        logger.info(f"No match files found for house {house_id}")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} matches across {len(threshold_schedule)} iterations")
    return combined


def deduplicate_cross_iteration(all_matches: pd.DataFrame) -> pd.DataFrame:
    """Remove events detected at multiple threshold levels.

    Strategy:
    - Group by (phase, approximate on_start ±2 min, approximate on_magnitude ±50W)
    - Keep the version from the *highest* iteration (lowest threshold = most refined)
    - Add ``detected_at_thresholds`` column listing all thresholds where found
    """
    if all_matches.empty:
        return all_matches

    # Ensure on_start is Timestamp
    if not pd.api.types.is_datetime64_any_dtype(all_matches['on_start']):
        all_matches['on_start'] = pd.to_datetime(all_matches['on_start'])

    # Sort by phase, on_start for efficient grouping
    df = all_matches.sort_values(['phase', 'on_start']).reset_index(drop=True)

    visited = set()
    keep_indices: List[int] = []
    detected_thresholds: Dict[int, List[int]] = {}

    for i in range(len(df)):
        if i in visited:
            continue

        row_i = df.iloc[i]
        group = [i]
        visited.add(i)

        # Find all duplicates of this event
        for j in range(i + 1, len(df)):
            if j in visited:
                continue
            row_j = df.iloc[j]

            if row_j['phase'] != row_i['phase']:
                # Past this phase (sorted), break inner if different phase
                if row_j['phase'] > row_i['phase']:
                    break
                continue

            time_diff = abs((row_j['on_start'] - row_i['on_start']).total_seconds() / 60)
            if time_diff > DEDUP_TIME_TOLERANCE_MINUTES * 2:
                break  # too far ahead in time

            mag_diff = abs(abs(row_j['on_magnitude']) - abs(row_i['on_magnitude']))
            if time_diff <= DEDUP_TIME_TOLERANCE_MINUTES and mag_diff <= DEDUP_MAGNITUDE_TOLERANCE_W:
                group.append(j)
                visited.add(j)

        # Keep highest iteration (lowest threshold)
        best_idx = max(group, key=lambda idx: df.iloc[idx]['iteration'])
        keep_indices.append(best_idx)
        detected_thresholds[best_idx] = sorted(set(int(df.iloc[idx]['threshold']) for idx in group))

    deduped = df.iloc[keep_indices].copy()
    deduped['detected_at_thresholds'] = [detected_thresholds[i] for i in keep_indices]

    removed = len(df) - len(deduped)
    if removed > 0:
        logger.info(f"Deduplication removed {removed} duplicates, kept {len(deduped)} unique matches")

    return deduped.reset_index(drop=True)


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


# ============================================================================
# Helpers
# ============================================================================

def _build_session(rows, phase: str) -> Session:
    """Create a Session from a list of named-tuple rows."""
    events = [row._asdict() for row in rows]

    starts = [r.on_start for r in rows]
    ends = [getattr(r, 'off_end', None) or getattr(r, 'off_start', None) or r.on_start for r in rows]
    magnitudes = [abs(r.on_magnitude) for r in rows]
    durations = [r.duration for r in rows if hasattr(r, 'duration') and r.duration is not None]

    session_start = min(starts)
    session_end = max(ends)
    total_dur = (session_end - session_start).total_seconds() / 60

    mean_mag = float(np.mean(magnitudes)) if magnitudes else 0
    std_mag = float(np.std(magnitudes)) if len(magnitudes) > 1 else 0
    cv = std_mag / mean_mag if mean_mag > 0 else 0

    thresholds = sorted(set(
        int(r.threshold) for r in rows if hasattr(r, 'threshold')
    ))

    return Session(
        session_id=f"s_{uuid.uuid4().hex[:8]}",
        phase=phase,
        events=events,
        start=session_start,
        end=session_end,
        cycle_count=len(rows),
        total_duration_minutes=round(total_dur, 1),
        avg_magnitude=round(mean_mag, 1),
        max_magnitude=round(float(max(magnitudes)), 1) if magnitudes else 0,
        magnitude_cv=round(cv, 3),
        avg_cycle_duration=round(float(np.mean(durations)), 1) if durations else 0,
        thresholds=thresholds,
    )


def _find_run_dir(experiment_dir: Path, run_number: int, threshold: int) -> Optional[Path]:
    """Find run directory — supports ``run_N`` and ``run_N_thXXXX`` naming."""
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
