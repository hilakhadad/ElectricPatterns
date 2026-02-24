"""
Cross-house recurring pattern matching.

Compares recurring_pattern signatures across all houses to identify
the same device type appearing in different households.  Each cross-house
match receives a global name (Device A, Device B, ...).

Runs after all per-house pipelines complete.  Input is the set of
device_sessions_*.json files in the experiment directory.
"""
import json
import logging
import os
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================
CROSS_HOUSE_MAGNITUDE_TOLERANCE = 0.20   # 20% relative magnitude difference
CROSS_HOUSE_DURATION_TOLERANCE = 0.30    # 30% relative duration difference
CROSS_HOUSE_MIN_HOUSES = 2              # minimum distinct houses to form global pattern
CROSS_HOUSE_MIN_CYCLES = 2             # sessions must have >1 cycle (multi-cycle device)
CROSS_HOUSE_MIN_SESSION_DURATION = 10.0 # sessions must average >10 min duration


# ============================================================================
# Public API
# ============================================================================

def match_patterns_across_houses(
    experiment_dir,               # type: Path
    house_ids,                    # type: List[str]
    magnitude_tolerance=None,     # type: Optional[float]
    duration_tolerance=None,      # type: Optional[float]
):
    # type: (...) -> Dict[str, Any]
    """Discover cross-house pattern matches from device session JSONs.

    Args:
        experiment_dir: Root experiment output directory.
        house_ids: List of house IDs to analyze.
        magnitude_tolerance: Override for magnitude matching tolerance (default 0.20).
        duration_tolerance: Override for duration matching tolerance (default 0.30).

    Returns:
        Dict with keys:
            global_patterns: list of global pattern dicts
            unmatched_patterns: patterns found in only one house
            summary: count statistics
            house_updates: {house_id: {pattern_id: {global_pattern_name, global_descriptive_name}}}
    """
    experiment_dir = Path(experiment_dir)
    mag_tol = magnitude_tolerance if magnitude_tolerance is not None else CROSS_HOUSE_MAGNITUDE_TOLERANCE
    dur_tol = duration_tolerance if duration_tolerance is not None else CROSS_HOUSE_DURATION_TOLERANCE

    # Extract one signature per pattern per house
    signatures = _extract_pattern_signatures(experiment_dir, house_ids)

    if not signatures:
        return _empty_result(len(house_ids))

    # Group signatures via connected components
    groups, unmatched_indices = _group_signatures(signatures, mag_tol, dur_tol)

    # Assign global names
    global_patterns = _assign_global_names(groups)
    unmatched_patterns = [signatures[i] for i in unmatched_indices]

    # Build house_updates mapping
    house_updates = _build_house_updates(global_patterns)

    houses_with_patterns = len(set(s['house_id'] for s in signatures))

    result = {
        'global_patterns': global_patterns,
        'unmatched_patterns': unmatched_patterns,
        'summary': {
            'houses_analyzed': len(house_ids),
            'houses_with_patterns': houses_with_patterns,
            'total_signatures': len(signatures),
            'total_global_patterns': len(global_patterns),
            'total_unmatched': len(unmatched_patterns),
        },
        'house_updates': house_updates,
        'settings': {
            'magnitude_tolerance': mag_tol,
            'duration_tolerance': dur_tol,
        },
    }

    n_global = len(global_patterns)
    if n_global > 0:
        logger.info(
            f"Cross-house matching: {n_global} global patterns "
            f"from {len(signatures)} signatures across {houses_with_patterns} houses"
        )

    return result


# ============================================================================
# Signature extraction
# ============================================================================

def _extract_pattern_signatures(
    experiment_dir,  # type: Path
    house_ids,       # type: List[str]
):
    # type: (...) -> List[Dict[str, Any]]
    """Extract one signature per recurring_pattern pattern_id per house.

    Reads device_sessions_{house_id}.json, filters recurring_pattern sessions,
    groups by pattern_id, and computes cluster-level summary.
    """
    sessions_dir = experiment_dir / 'device_sessions'
    signatures = []

    for house_id in house_ids:
        json_path = sessions_dir / f'device_sessions_{house_id}.json'
        if not json_path.exists():
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load {json_path}: {e}")
            continue

        sessions = data.get('sessions', [])
        rp_sessions = [s for s in sessions if s.get('device_type') == 'recurring_pattern']

        if not rp_sessions:
            continue

        # Group by pattern_id
        pattern_groups = {}  # type: Dict[int, List[dict]]
        for s in rp_sessions:
            pid = s.get('confidence_breakdown', {}).get('pattern_id')
            if pid is not None:
                pattern_groups.setdefault(pid, []).append(s)

        for pid, group in pattern_groups.items():
            # All sessions in a per-house pattern share the same cluster stats
            bd = group[0].get('confidence_breakdown', {})
            avg_mag = bd.get('avg_magnitude_w', 0)
            avg_dur = bd.get('avg_duration_min', 0)

            # If cluster-level stats are missing, compute from sessions
            if avg_mag == 0:
                mags = [s.get('avg_cycle_magnitude_w', 0) or 0 for s in group]
                avg_mag = float(np.mean(mags)) if mags else 0
            if avg_dur == 0:
                durs = [s.get('duration_minutes', 0) or 0 for s in group]
                avg_dur = float(np.mean(durs)) if durs else 0

            # Compute session-level stats for filtering
            cycle_counts = [s.get('cycle_count', 1) or 1 for s in group]
            session_durations = [s.get('duration_minutes', 0) or 0 for s in group]
            avg_cycles = float(np.mean(cycle_counts))
            avg_sess_dur = float(np.mean(session_durations))

            # Filter: only multi-cycle, long-duration patterns qualify
            if avg_cycles < CROSS_HOUSE_MIN_CYCLES:
                continue
            if avg_sess_dur < CROSS_HOUSE_MIN_SESSION_DURATION:
                continue

            signatures.append({
                'house_id': str(house_id),
                'pattern_id': int(pid),
                'avg_magnitude_w': float(avg_mag),
                'avg_duration_min': float(avg_dur),
                'avg_cycle_count': round(avg_cycles, 1),
                'avg_session_duration_min': round(avg_sess_dur, 1),
                'n_sessions': len(group),
                'confidence': float(group[0].get('confidence', 0)),
            })

    return signatures


# ============================================================================
# Grouping via connected components
# ============================================================================

def _group_signatures(
    signatures,    # type: List[Dict[str, Any]]
    mag_tolerance, # type: float
    dur_tolerance, # type: float
):
    # type: (...) -> Tuple[List[List[Dict[str, Any]]], List[int]]
    """Group signatures into cross-house matches using complete-linkage.

    Uses agglomerative clustering with complete linkage: every pair within
    a group must be within tolerance.  This prevents transitive chaining
    where A~B and B~C pull A and C together even though A and C are far
    apart.

    Two signatures match if:
      - They are from DIFFERENT houses
      - |mag_a - mag_b| / max(mag_a, mag_b) <= mag_tolerance
      - |dur_a - dur_b| / max(dur_a, dur_b) <= dur_tolerance

    Returns:
        (groups, unmatched_indices)
        - groups: list of lists of signature dicts (spanning 2+ houses each)
        - unmatched_indices: indices into original signatures list
    """
    n = len(signatures)
    if n == 0:
        return [], []

    # Build pairwise cross-house match matrix
    match_matrix = [[False] * n for _ in range(n)]
    for i in range(n):
        match_matrix[i][i] = True  # self-match
        for j in range(i + 1, n):
            if signatures[i]['house_id'] == signatures[j]['house_id']:
                continue  # Same house → never match
            if _signatures_match(signatures[i], signatures[j], mag_tolerance, dur_tolerance):
                match_matrix[i][j] = True
                match_matrix[j][i] = True

    # Complete-linkage agglomerative clustering
    # Start with each signature in its own cluster
    clusters = [[i] for i in range(n)]

    merged = True
    while merged:
        merged = False
        best_pair = None
        best_score = -1  # prefer merging smaller clusters first

        for ci in range(len(clusters)):
            for cj in range(ci + 1, len(clusters)):
                # Check complete linkage: every cross-house pair must match
                if _can_merge_complete(clusters[ci], clusters[cj],
                                       match_matrix, signatures):
                    # Score: number of distinct houses in merged cluster
                    merged_houses = set()
                    for idx in clusters[ci] + clusters[cj]:
                        merged_houses.add(signatures[idx]['house_id'])
                    score = len(merged_houses)
                    if score > best_score:
                        best_score = score
                        best_pair = (ci, cj)

        if best_pair is not None:
            ci, cj = best_pair
            clusters[ci] = clusters[ci] + clusters[cj]
            del clusters[cj]
            merged = True

    # Split into multi-house groups and unmatched
    groups = []
    unmatched_indices = []

    for cluster in clusters:
        house_ids_in_cluster = set(signatures[i]['house_id'] for i in cluster)
        if len(house_ids_in_cluster) >= CROSS_HOUSE_MIN_HOUSES:
            groups.append([signatures[i] for i in cluster])
        else:
            unmatched_indices.extend(cluster)

    return groups, unmatched_indices


def _can_merge_complete(cluster_a, cluster_b, match_matrix, signatures):
    # type: (List[int], List[int], List[List[bool]], List[Dict]) -> bool
    """Check if two clusters can merge under complete linkage.

    Rules:
    1. Every cross-house pair between the two clusters must be within tolerance.
    2. No duplicate houses: merged cluster must have at most one signature per house.
       (If per-house DBSCAN says two patterns are different devices, we respect that.)
    3. Merged cluster must span 2+ distinct houses.
    """
    # Rule 2: no duplicate houses in merged cluster
    all_indices = cluster_a + cluster_b
    houses = [signatures[i]['house_id'] for i in all_indices]
    if len(houses) != len(set(houses)):
        return False  # Would create duplicate house entries

    # Rule 3: must span 2+ houses
    if len(set(houses)) < 2:
        return False

    # Rule 1: every cross-house pair must match
    for i in cluster_a:
        for j in cluster_b:
            if not match_matrix[i][j]:
                return False
    return True


def _signatures_match(
    sig_a,         # type: Dict[str, Any]
    sig_b,         # type: Dict[str, Any]
    mag_tolerance, # type: float
    dur_tolerance, # type: float
):
    # type: (...) -> bool
    """Check if two signatures are within tolerance."""
    mag_a = sig_a['avg_magnitude_w']
    mag_b = sig_b['avg_magnitude_w']
    dur_a = sig_a['avg_duration_min']
    dur_b = sig_b['avg_duration_min']

    # Avoid division by zero
    max_mag = max(mag_a, mag_b)
    max_dur = max(dur_a, dur_b)

    if max_mag == 0 or max_dur == 0:
        return False

    mag_diff = abs(mag_a - mag_b) / max_mag
    dur_diff = abs(dur_a - dur_b) / max_dur

    return mag_diff <= mag_tolerance and dur_diff <= dur_tolerance


# ============================================================================
# Naming
# ============================================================================

def _assign_global_names(
    groups,  # type: List[List[Dict[str, Any]]]
):
    # type: (...) -> List[Dict[str, Any]]
    """Assign Device A, B, C... names to matched groups.

    Groups are sorted by average magnitude descending, so "Device A"
    is the highest-power cross-house device.
    """
    if not groups:
        return []

    # Compute group-level averages for sorting
    group_summaries = []
    for group in groups:
        magnitudes = [s['avg_magnitude_w'] for s in group]
        durations = [s['avg_duration_min'] for s in group]
        avg_mag = float(np.mean(magnitudes))
        avg_dur = float(np.mean(durations))
        group_summaries.append((avg_mag, avg_dur, group))

    # Sort by magnitude descending, then duration descending
    group_summaries.sort(key=lambda x: (-x[0], -x[1]))

    result = []
    for idx, (avg_mag, avg_dur, group) in enumerate(group_summaries):
        letter = chr(ord('A') + idx) if idx < 26 else f'#{idx + 1}'
        global_name = f'Device {letter}'
        descriptive_name = f'~{avg_mag:,.0f}W / ~{avg_dur:.0f}min'

        magnitudes = [s['avg_magnitude_w'] for s in group]
        durations = [s['avg_duration_min'] for s in group]

        # Match quality score
        match_quality = _compute_match_quality(group)

        result.append({
            'global_name': global_name,
            'descriptive_name': descriptive_name,
            'avg_magnitude_w': round(avg_mag, 0),
            'avg_duration_min': round(avg_dur, 1),
            'magnitude_range': [round(min(magnitudes), 0), round(max(magnitudes), 0)],
            'duration_range': [round(min(durations), 1), round(max(durations), 1)],
            'houses': sorted(group, key=lambda s: s['house_id']),
            'match_quality': match_quality,
            'n_houses': len(set(s['house_id'] for s in group)),
        })

    return result


def _compute_match_quality(group):
    # type: (List[Dict[str, Any]]) -> float
    """Compute match quality score (0-1) for a cross-house group.

    Factors:
      - house_count: more houses → higher quality (2→0.58, 5+→1.0)
      - magnitude_consistency: low CV → higher quality
      - duration_consistency: low CV → higher quality
    """
    magnitudes = [s['avg_magnitude_w'] for s in group]
    durations = [s['avg_duration_min'] for s in group]

    house_count = len(set(s['house_id'] for s in group))
    house_score = min(1.0, 0.3 + 0.14 * house_count)

    mag_cv = _safe_cv(magnitudes)
    dur_cv = _safe_cv(durations)

    # CV=0 → 1.0, CV=0.20 → 0.0
    mag_score = max(0.0, 1.0 - 5.0 * mag_cv)
    # CV=0 → 1.0, CV=0.30 → 0.0
    dur_score = max(0.0, 1.0 - 3.33 * dur_cv)

    return round(float(np.mean([house_score, mag_score, dur_score])), 2)


# ============================================================================
# House JSON updates
# ============================================================================

def _build_house_updates(global_patterns):
    # type: (List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, str]]]
    """Build mapping {house_id: {pattern_id: {global_pattern_name, global_descriptive_name}}}.

    Used by update_house_jsons() to write back into per-house JSONs.
    """
    updates = {}  # type: Dict[str, Dict[int, Dict[str, str]]]
    for gp in global_patterns:
        global_name = gp['global_name']
        descriptive = gp['descriptive_name']
        for sig in gp['houses']:
            house_id = sig['house_id']
            pid = sig['pattern_id']
            updates.setdefault(house_id, {})[pid] = {
                'global_pattern_name': global_name,
                'global_descriptive_name': descriptive,
            }
    return updates


def update_house_jsons(
    experiment_dir,  # type: Path
    house_updates,   # type: Dict[str, Dict[int, Dict[str, str]]]
):
    # type: (...) -> int
    """Update per-house device_sessions JSON with global_pattern_name.

    Adds 'global_pattern_name' and 'global_descriptive_name' to the
    confidence_breakdown of matching recurring_pattern sessions.
    Idempotent: overwrites existing values if present.

    Returns count of updated sessions.
    """
    experiment_dir = Path(experiment_dir)
    sessions_dir = experiment_dir / 'device_sessions'
    updated_count = 0

    for house_id, pattern_map in house_updates.items():
        json_path = sessions_dir / f'device_sessions_{house_id}.json'
        if not json_path.exists():
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        modified = False
        for session in data.get('sessions', []):
            if session.get('device_type') != 'recurring_pattern':
                continue
            bd = session.get('confidence_breakdown', {})
            pid = bd.get('pattern_id')
            if pid is not None and pid in pattern_map:
                info = pattern_map[pid]
                bd['global_pattern_name'] = info['global_pattern_name']
                bd['global_descriptive_name'] = info['global_descriptive_name']
                session['confidence_breakdown'] = bd
                modified = True
                updated_count += 1

        if modified:
            # Write to temp file first, then atomic replace
            tmp_path = json_path.with_suffix('.json.tmp')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            # os.replace is atomic on Windows and overwrites existing
            os.replace(tmp_path, json_path)

    return updated_count


def save_cross_house_summary(
    experiment_dir,  # type: Path
    match_result,    # type: Dict[str, Any]
):
    # type: (...) -> Path
    """Save cross_house_patterns.json in the experiment directory.

    Returns path to saved file.
    """
    experiment_dir = Path(experiment_dir)
    output_path = experiment_dir / 'cross_house_patterns.json'

    output_data = {
        'generated_at': datetime.now().isoformat(),
        'settings': match_result.get('settings', {}),
        'summary': match_result['summary'],
        'global_patterns': match_result['global_patterns'],
        'unmatched_patterns': match_result['unmatched_patterns'],
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Cross-house patterns saved to {output_path}")
    return output_path


# ============================================================================
# Helpers
# ============================================================================

def _safe_cv(values):
    # type: (list) -> float
    """Coefficient of variation, safe for constant or empty lists."""
    if not values or len(values) < 2:
        return 0.0
    mean = float(np.mean(values))
    if mean == 0:
        return 0.0
    return float(np.std(values) / mean)


def _empty_result(n_houses):
    # type: (int) -> Dict[str, Any]
    """Return empty result when no patterns exist."""
    return {
        'global_patterns': [],
        'unmatched_patterns': [],
        'summary': {
            'houses_analyzed': n_houses,
            'houses_with_patterns': 0,
            'total_signatures': 0,
            'total_global_patterns': 0,
            'total_unmatched': 0,
        },
        'house_updates': {},
        'settings': {
            'magnitude_tolerance': CROSS_HOUSE_MAGNITUDE_TOLERANCE,
            'duration_tolerance': CROSS_HOUSE_DURATION_TOLERANCE,
        },
    }
