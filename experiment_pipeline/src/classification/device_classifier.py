"""
Device classification for dynamic threshold experiments.

Classifies matched ON/OFF pairs into device types:
- boiler: long duration (>=25min), high power (>=1500W), single-phase, isolated
- central_ac: synchronized across 2+ phases within 10 min
- regular_ac: 800W+, compressor cycling pattern (sessions with 4+ cycles)
- unclassified: everything else

Classification logic adapted from experiment_analysis/src/metrics/patterns.py.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

# ============================================================================
# Classification constants
# ============================================================================
BOILER_MIN_DURATION = 25       # minutes
BOILER_MIN_MAGNITUDE = 1500    # watts
BOILER_ISOLATION_WINDOW = 30   # minutes — no medium events within this window
BOILER_MIN_COUNT = 3           # minimum activations to confirm boiler

AC_SYNC_TOLERANCE = 10         # minutes — central AC phase sync tolerance
AC_MIN_MAGNITUDE = 800         # watts
AC_MIN_CYCLE_DURATION = 3      # minutes
AC_MAX_CYCLE_DURATION = 30     # minutes
AC_MIN_INITIAL_DURATION = 15   # minutes — first activation in session
AC_SESSION_GAP = 60            # minutes — gap between sessions
AC_MIN_FOLLOWING_CYCLES = 3    # minimum cycles after initial
AC_MAX_MAGNITUDE_STD_PCT = 0.20  # 20% magnitude consistency
AC_MIN_SESSION_COUNT = 3       # minimum sessions to confirm AC

# AC filtering for boiler candidates
AC_FILTER_WINDOW = 60          # minutes — window to search for compressor cycles
AC_FILTER_MIN_CYCLES = 2       # minimum nearby cycles to reclassify as AC
AC_FILTER_MIN_CYCLE_MAG = 800  # watts
AC_FILTER_MAG_RATIO = 0.50     # cycle must be >= 50% of boiler magnitude

MULTI_PHASE_WINDOW = 5         # minutes — window for multi-phase simultaneity check

PHASES = ['w1', 'w2', 'w3']


# ============================================================================
# Main entry points
# ============================================================================

def classify_iteration_matches(
    run_dir: Path,
    house_id: str,
    run_number: int,
    threshold: int,
    parent_logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Classify all matches from a single pipeline iteration.

    Loads all monthly match files for the given run, classifies each match
    as a device type, and saves classification results.

    Args:
        run_dir: Path to run directory (e.g., .../run_0_th2000)
        house_id: House ID being processed
        run_number: Iteration number
        threshold: Threshold used in this iteration
        parent_logger: Optional logger from the calling script

    Returns:
        DataFrame with all matches + 'device_type' column, or empty DataFrame
    """
    log = parent_logger or logger
    house_dir = run_dir / f"house_{house_id}"
    matches_dir = house_dir / "matches"

    if not matches_dir.exists():
        log.info(f"  Classification: no matches directory for run {run_number}")
        return pd.DataFrame()

    # Load all monthly match files into one DataFrame
    all_matches = _load_all_matches(matches_dir, house_id)
    if all_matches.empty:
        log.info(f"  Classification: no matches found for run {run_number}")
        return pd.DataFrame()

    log.info(f"  Classification: {len(all_matches)} matches to classify (TH={threshold}W)")

    # Initialize all as unclassified
    all_matches['device_type'] = 'unclassified'

    # Step 1: Classify boilers (highest priority — long, high-power, isolated)
    boiler_mask = _classify_boilers(all_matches, log)
    all_matches.loc[boiler_mask, 'device_type'] = 'boiler'
    n_boiler = boiler_mask.sum()

    # Step 2: Classify central AC (multi-phase synchronized)
    remaining = all_matches['device_type'] == 'unclassified'
    central_mask = _classify_central_ac(all_matches, remaining, log)
    all_matches.loc[central_mask, 'device_type'] = 'central_ac'
    n_central = central_mask.sum()

    # Step 3: Classify regular AC (single-phase cycling pattern)
    remaining = all_matches['device_type'] == 'unclassified'
    regular_mask = _classify_regular_ac(all_matches, remaining, log)
    all_matches.loc[regular_mask, 'device_type'] = 'regular_ac'
    n_regular = regular_mask.sum()

    n_unclassified = (all_matches['device_type'] == 'unclassified').sum()
    log.info(f"  Classification results: {n_boiler} boiler, {n_central} central_ac, "
             f"{n_regular} regular_ac, {n_unclassified} unclassified")

    # Save classification files per month
    _save_classification(all_matches, house_dir, house_id)

    return all_matches


def generate_activation_list(
    experiment_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
) -> Dict[str, Any]:
    """
    Generate per-iteration activation list for a house.

    Reads classification files from all iterations, produces a structured
    summary of what devices were detected at each threshold level.

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID
        threshold_schedule: List of thresholds [2000, 1500, 1100, 800]

    Returns:
        Activation list dict (also saved as JSON)
    """
    result = {
        'house_id': house_id,
        'threshold_schedule': threshold_schedule,
        'iterations': [],
        'summary': {},
    }

    total_by_type = {'boiler': 0, 'central_ac': 0, 'regular_ac': 0, 'unclassified': 0}

    for run_number, threshold in enumerate(threshold_schedule):
        run_dir = _find_run_dir(experiment_dir, run_number, threshold)
        if run_dir is None:
            continue

        house_dir = run_dir / f"house_{house_id}"
        classification_dir = house_dir / "classification"

        if not classification_dir.exists():
            continue

        # Load all classification files for this iteration
        classified = _load_all_pkl(classification_dir, f"classification_{house_id}_*.pkl")
        if classified.empty:
            continue

        iteration_info = _build_iteration_info(classified, run_number, threshold)
        result['iterations'].append(iteration_info)

        for dtype, info in iteration_info['devices'].items():
            total_by_type[dtype] += info['count']

    total_all = sum(total_by_type.values())
    total_classified = total_all - total_by_type['unclassified']
    result['summary'] = {
        'total_matches': total_all,
        'total_classified': total_classified,
        'total_unclassified': total_by_type['unclassified'],
        'overall_classified_rate': round(total_classified / total_all, 3) if total_all > 0 else 0,
        'by_device_type': total_by_type,
    }

    # Save to JSON
    activation_lists_dir = experiment_dir / "activation_lists"
    activation_lists_dir.mkdir(parents=True, exist_ok=True)
    output_path = activation_lists_dir / f"activation_list_{house_id}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ============================================================================
# Classification logic
# ============================================================================

def _classify_boilers(matches: pd.DataFrame, log: logging.Logger) -> pd.Series:
    """
    Classify matches as boilers.

    Criteria:
    1. Duration >= 25 min
    2. Magnitude >= 1500W
    3. Single-phase (no synchronized event on other phases within ±5 min)
    4. Isolated (no medium-duration events (3-24 min) within ±30 min on same phase)
    5. No AC compressor cycling nearby (±60 min, 2+ cycles of 3-24 min, 800W+)
    """
    mask = pd.Series(False, index=matches.index)

    if matches.empty:
        return mask

    # Basic filters
    duration_ok = matches['duration'] >= BOILER_MIN_DURATION
    magnitude_ok = matches['on_magnitude'].abs() >= BOILER_MIN_MAGNITUDE
    candidates = duration_ok & magnitude_ok

    if candidates.sum() == 0:
        return mask

    candidate_df = matches[candidates].copy()
    medium_df = matches[
        (matches['duration'] >= AC_MIN_CYCLE_DURATION) &
        (matches['duration'] <= AC_MAX_CYCLE_DURATION)
    ]

    valid_indices = []

    for phase in PHASES:
        phase_candidates = candidate_df[candidate_df['phase'] == phase]
        if phase_candidates.empty:
            continue

        phase_medium = medium_df[medium_df['phase'] == phase]

        for idx, row in phase_candidates.iterrows():
            # Isolation check: no medium events nearby on same phase
            if not phase_medium.empty:
                time_diffs = (phase_medium['on_start'] - row['on_start']).abs()
                if hasattr(time_diffs.iloc[0], 'total_seconds'):
                    nearby = (time_diffs.dt.total_seconds() / 60) <= BOILER_ISOLATION_WINDOW
                else:
                    nearby = time_diffs <= BOILER_ISOLATION_WINDOW
                if nearby.any():
                    continue

            # Multi-phase check: no synchronized events on other phases
            other_phases = [p for p in PHASES if p != phase]
            is_multi_phase = False
            for other_phase in other_phases:
                other_matches = matches[matches['phase'] == other_phase]
                if other_matches.empty:
                    continue
                time_diffs = (other_matches['on_start'] - row['on_start']).abs()
                if hasattr(time_diffs.iloc[0], 'total_seconds'):
                    nearby = (time_diffs.dt.total_seconds() / 60) <= MULTI_PHASE_WINDOW
                else:
                    nearby = time_diffs <= MULTI_PHASE_WINDOW
                if nearby.any():
                    is_multi_phase = True
                    break

            if is_multi_phase:
                continue

            # AC filter: check for compressor cycling nearby
            if _has_nearby_compressor_cycles(row, matches, phase):
                continue

            valid_indices.append(idx)

    mask.loc[valid_indices] = True
    return mask


def _has_nearby_compressor_cycles(
    boiler_row: pd.Series,
    all_matches: pd.DataFrame,
    phase: str,
) -> bool:
    """Check if there are compressor cycles near a boiler candidate."""
    phase_matches = all_matches[all_matches['phase'] == phase]
    if phase_matches.empty:
        return False

    boiler_mag = abs(boiler_row['on_magnitude'])
    boiler_start = boiler_row['on_start']
    boiler_end = boiler_row.get('off_end', boiler_row.get('off_start', boiler_start))

    # Search window: from boiler_start - 60min to boiler_end + 60min
    window_start = boiler_start - pd.Timedelta(minutes=AC_FILTER_WINDOW)
    window_end = boiler_end + pd.Timedelta(minutes=AC_FILTER_WINDOW)

    nearby = phase_matches[
        (phase_matches['on_start'] >= window_start) &
        (phase_matches['on_start'] <= window_end)
    ]

    # Filter for compressor-like cycles
    cycles = nearby[
        (nearby['duration'] >= AC_MIN_CYCLE_DURATION) &
        (nearby['duration'] <= AC_MAX_CYCLE_DURATION) &
        (nearby['on_magnitude'].abs() >= AC_FILTER_MIN_CYCLE_MAG) &
        (nearby['on_magnitude'].abs() >= boiler_mag * AC_FILTER_MAG_RATIO)
    ]

    # Exclude the boiler candidate itself
    cycles = cycles[cycles.index != boiler_row.name]

    return len(cycles) >= AC_FILTER_MIN_CYCLES


def _classify_central_ac(
    matches: pd.DataFrame,
    remaining_mask: pd.Series,
    log: logging.Logger,
) -> pd.Series:
    """
    Classify matches as central AC.

    Criteria: synchronized events on 2+ phases within ±10 minutes.
    Uses merge_asof to find phase-synchronized events.
    """
    mask = pd.Series(False, index=matches.index)
    remaining = matches[remaining_mask].copy()

    if remaining.empty:
        return mask

    # Need events on at least 2 phases
    phases_with_data = [p for p in PHASES if (remaining['phase'] == p).any()]
    if len(phases_with_data) < 2:
        return mask

    # Use first phase as reference
    ref_phase = phases_with_data[0]
    ref_matches = remaining[remaining['phase'] == ref_phase].copy()
    ref_matches = ref_matches.sort_values('on_start')

    if ref_matches.empty:
        return mask

    # Find synchronized events on other phases
    synchronized_indices = set()

    for other_phase in phases_with_data[1:]:
        other_matches = remaining[remaining['phase'] == other_phase].copy()
        if other_matches.empty:
            continue

        other_matches = other_matches.sort_values('on_start')

        # merge_asof: find nearest match on other phase within tolerance
        merged = pd.merge_asof(
            ref_matches[['on_start']].reset_index(),
            other_matches[['on_start']].reset_index().rename(
                columns={'index': 'other_index', 'on_start': 'other_on_start'}
            ),
            left_on='on_start',
            right_on='other_on_start',
            tolerance=pd.Timedelta(minutes=AC_SYNC_TOLERANCE),
            direction='nearest',
        )

        # Matches where both phases have events
        synced = merged.dropna(subset=['other_index'])
        for _, row in synced.iterrows():
            synchronized_indices.add(int(row['index']))
            synchronized_indices.add(int(row['other_index']))

    mask.loc[list(synchronized_indices)] = True
    return mask


def _classify_regular_ac(
    matches: pd.DataFrame,
    remaining_mask: pd.Series,
    log: logging.Logger,
) -> pd.Series:
    """
    Classify matches as regular AC.

    Criteria: 800W+, belongs to valid session (initial >=15min + 3+ following cycles).
    Cycling events: 3-30 min duration. Initial run events: >=15 min.
    Sessions: gap < 60 min between events on same phase.
    """
    mask = pd.Series(False, index=matches.index)
    remaining = matches[remaining_mask].copy()

    if remaining.empty:
        return mask

    for phase in PHASES:
        phase_remaining = remaining[remaining['phase'] == phase]
        if phase_remaining.empty:
            continue

        # Candidate pool: cycling (3-30 min) OR initial (>=15 min), all >= 800W
        magnitude_ok = phase_remaining['on_magnitude'].abs() >= AC_MIN_MAGNITUDE
        cycling = (
            (phase_remaining['duration'] >= AC_MIN_CYCLE_DURATION) &
            (phase_remaining['duration'] <= AC_MAX_CYCLE_DURATION)
        )
        initial = phase_remaining['duration'] >= AC_MIN_INITIAL_DURATION
        candidates = phase_remaining[magnitude_ok & (cycling | initial)].copy()

        if candidates.empty:
            continue

        # Group into sessions
        sessions = _group_into_sessions(candidates)

        for session_indices in sessions:
            session_df = candidates.loc[session_indices]
            if _is_valid_ac_session(session_df):
                mask.loc[session_indices] = True

    return mask


def _group_into_sessions(matches: pd.DataFrame) -> List[List]:
    """
    Group matches into sessions based on time gaps.

    Events with gap < 60 min belong to the same session.

    Returns:
        List of lists, each containing indices of matches in one session.
    """
    if matches.empty:
        return []

    sorted_matches = matches.sort_values('on_start')
    sessions = []
    current_session = [sorted_matches.index[0]]

    for i in range(1, len(sorted_matches)):
        prev_end = sorted_matches.iloc[i - 1].get('off_end',
                    sorted_matches.iloc[i - 1].get('off_start',
                    sorted_matches.iloc[i - 1]['on_start']))
        curr_start = sorted_matches.iloc[i]['on_start']

        gap = (curr_start - prev_end)
        if hasattr(gap, 'total_seconds'):
            gap_minutes = gap.total_seconds() / 60
        else:
            gap_minutes = gap

        if gap_minutes > AC_SESSION_GAP:
            sessions.append(current_session)
            current_session = [sorted_matches.index[i]]
        else:
            current_session.append(sorted_matches.index[i])

    sessions.append(current_session)
    return sessions


def _is_valid_ac_session(session_df: pd.DataFrame) -> bool:
    """
    Validate an AC session.

    Rules:
    1. Total cycles >= 4 (1 initial + 3 following)
    2. First activation duration >= 15 min
    3. Magnitude consistency: std/mean <= 20%
    """
    if len(session_df) < (1 + AC_MIN_FOLLOWING_CYCLES):
        return False

    sorted_session = session_df.sort_values('on_start')

    # First activation must be long enough
    first_duration = sorted_session.iloc[0]['duration']
    if first_duration < AC_MIN_INITIAL_DURATION:
        return False

    # Magnitude consistency
    magnitudes = sorted_session['on_magnitude'].abs()
    mean_mag = magnitudes.mean()
    if mean_mag == 0:
        return False

    std_pct = magnitudes.std() / mean_mag
    if std_pct > AC_MAX_MAGNITUDE_STD_PCT:
        return False

    return True


# ============================================================================
# Data loading helpers
# ============================================================================

def _load_all_matches(matches_dir: Path, house_id: str) -> pd.DataFrame:
    """Load all monthly match files into one DataFrame."""
    return _load_all_pkl(matches_dir, f"matches_{house_id}_*.pkl")


def _load_all_pkl(directory: Path, pattern: str) -> pd.DataFrame:
    """Load all pkl files matching pattern into one DataFrame."""
    files = sorted(directory.glob(pattern))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=False)


def _find_run_dir(experiment_dir: Path, run_number: int, threshold: int) -> Optional[Path]:
    """Find run directory for dynamic threshold naming."""
    # Try exact dynamic naming
    exact = experiment_dir / f"run_{run_number}_th{threshold}"
    if exact.exists():
        return exact

    # Try plain naming
    plain = experiment_dir / f"run_{run_number}"
    if plain.exists():
        return plain

    # Try glob
    for d in experiment_dir.glob(f"run_{run_number}_th*"):
        if d.is_dir():
            return d

    return None


def _save_classification(
    classified_df: pd.DataFrame,
    house_dir: Path,
    house_id: str,
) -> None:
    """Save classification results as monthly pkl files."""
    classification_dir = house_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)

    if 'on_start' not in classified_df.columns:
        # Save as single file
        classified_df.to_pickle(classification_dir / f"classification_{house_id}.pkl")
        return

    # Group by month and save per-month files
    classified_df['_month'] = classified_df['on_start'].dt.month
    classified_df['_year'] = classified_df['on_start'].dt.year

    for (month, year), group in classified_df.groupby(['_month', '_year']):
        output = group.drop(columns=['_month', '_year'])
        output.to_pickle(
            classification_dir / f"classification_{house_id}_{int(month):02d}_{int(year)}.pkl"
        )

    classified_df.drop(columns=['_month', '_year'], inplace=True)


def _build_iteration_info(
    classified: pd.DataFrame,
    run_number: int,
    threshold: int,
) -> Dict[str, Any]:
    """Build per-iteration info for the activation list."""
    target_map = {
        2000: 'Boilers, water heaters (very high power)',
        1500: 'Strong ACs, central AC units',
        1100: 'Medium AC units',
        800: 'Small ACs, additional devices',
    }

    devices = {}
    for dtype in ['boiler', 'central_ac', 'regular_ac', 'unclassified']:
        subset = classified[classified['device_type'] == dtype]
        activations = []
        for _, row in subset.iterrows():
            activation = {
                'date': str(row['on_start'].date()) if pd.notna(row.get('on_start')) else None,
                'phase': row.get('phase', ''),
                'magnitude': round(abs(row.get('on_magnitude', 0))),
                'duration_min': round(row.get('duration', 0), 1),
            }
            activations.append(activation)

        devices[dtype] = {
            'count': len(subset),
            'avg_magnitude': round(subset['on_magnitude'].abs().mean()) if not subset.empty else 0,
            'avg_duration': round(subset['duration'].mean(), 1) if not subset.empty else 0,
            'activations': activations,
        }

    total = len(classified)
    classified_count = total - devices['unclassified']['count']

    return {
        'run_number': run_number,
        'threshold': threshold,
        'target_description': target_map.get(threshold, f'Threshold {threshold}W'),
        'devices': devices,
        'total_matches': total,
        'classified_count': classified_count,
        'classified_rate': round(classified_count / total, 3) if total > 0 else 0,
    }
