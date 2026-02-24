"""
Recurring pattern discovery via DBSCAN clustering.

After boiler and AC classification, the remaining unclassified events are
grouped into sessions and clustered per-phase.  Sessions that form tight
behavioural clusters (similar magnitude, duration, cycle count) are
promoted from *unknown* to *recurring_pattern* — a NILM "discovered load".

Sessions that don't cluster remain *unknown*.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from ..config import (
    RECURRING_PATTERN_EPS,
    RECURRING_PATTERN_MIN_SAMPLES,
    RECURRING_PATTERN_MAX_INTERNAL_CV,
    RECURRING_PATTERN_MIN_SESSIONS,
    PHASES,
)
from ..session_grouper import Session, group_into_sessions

logger = logging.getLogger(__name__)


# ============================================================================
# Public API
# ============================================================================

def _identify_recurring_patterns(
    remaining: pd.DataFrame,
    all_matches: pd.DataFrame,
) -> Tuple[list, List[Session]]:
    """Discover recurring behavioural patterns among unclassified events.

    Groups remaining events into sessions, then clusters per-phase using
    DBSCAN on normalised behavioural features.

    Args:
        remaining: Events not consumed by boiler/AC steps.
        all_matches: Full event DataFrame (for context scoring).

    Returns:
        (recurring_classified, leftover_sessions)
        - recurring_classified: list of ClassifiedSession with
          device_type='recurring_pattern'
        - leftover_sessions: sessions that didn't cluster (will become unknown)
    """
    # Avoid circular import — ClassifiedSession lives in session_classifier
    from ..session_classifier import ClassifiedSession

    if remaining.empty:
        return [], []

    # Group remaining events into sessions (same logic as unknown step)
    sessions = group_into_sessions(remaining)
    if not sessions:
        return [], []

    # Cluster per-phase
    recurring_classified: list = []
    leftover_sessions: List[Session] = []
    pattern_counter = 0

    for phase in PHASES:
        phase_sessions = [s for s in sessions if s.phase == phase]
        if len(phase_sessions) < RECURRING_PATTERN_MIN_SAMPLES:
            # Not enough sessions to form any cluster on this phase
            leftover_sessions.extend(phase_sessions)
            continue

        clusters, noise_indices = _cluster_sessions(phase_sessions)

        for cluster_indices in clusters:
            pattern_counter += 1
            cluster_sessions = [phase_sessions[i] for i in cluster_indices]

            # Validate cluster quality
            if not _is_valid_cluster(cluster_sessions):
                leftover_sessions.extend(cluster_sessions)
                continue

            # Build ClassifiedSession for each member
            conf, breakdown = _recurring_pattern_confidence(
                cluster_sessions, pattern_counter
            )
            reason = _recurring_pattern_reason(
                cluster_sessions, pattern_counter, phase
            )

            for s in cluster_sessions:
                recurring_classified.append(ClassifiedSession(
                    session=s,
                    device_type='recurring_pattern',
                    reason=reason,
                    confidence=conf,
                    confidence_breakdown={**breakdown, 'pattern_id': pattern_counter},
                ))

        # Noise points become leftover
        for i in noise_indices:
            leftover_sessions.append(phase_sessions[i])

    n_recurring = len(recurring_classified)
    n_leftover = len(leftover_sessions)
    if n_recurring > 0:
        logger.info(
            f"  Step 2.5: {n_recurring} sessions in {pattern_counter} "
            f"recurring patterns, {n_leftover} sessions remain unclassified"
        )

    return recurring_classified, leftover_sessions


# ============================================================================
# DBSCAN implementation (uses scipy, no sklearn dependency)
# ============================================================================

def _cluster_sessions(
    sessions: List[Session],
) -> Tuple[List[List[int]], List[int]]:
    """Run DBSCAN on a list of sessions from the same phase.

    Returns:
        (clusters, noise_indices)
        - clusters: list of lists, each containing session indices
        - noise_indices: indices of sessions not in any cluster
    """
    features = _extract_features(sessions)
    if features is None or len(features) < RECURRING_PATTERN_MIN_SAMPLES:
        return [], list(range(len(sessions)))

    # Normalise features (zero-mean, unit-variance per column)
    features_norm = _normalise_features(features)

    # Compute pairwise Euclidean distances
    if len(features_norm) < 2:
        return [], list(range(len(sessions)))

    dist_matrix = squareform(pdist(features_norm, metric='euclidean'))

    # DBSCAN
    labels = _dbscan(dist_matrix, RECURRING_PATTERN_EPS, RECURRING_PATTERN_MIN_SAMPLES)

    # Group by label
    clusters: Dict[int, List[int]] = {}
    noise_indices: List[int] = []

    for idx, label in enumerate(labels):
        if label == -1:
            noise_indices.append(idx)
        else:
            clusters.setdefault(label, []).append(idx)

    # Filter clusters by minimum size
    valid_clusters = []
    for cluster_indices in clusters.values():
        if len(cluster_indices) >= RECURRING_PATTERN_MIN_SESSIONS:
            valid_clusters.append(cluster_indices)
        else:
            noise_indices.extend(cluster_indices)

    return valid_clusters, noise_indices


def _dbscan(
    dist_matrix: np.ndarray,
    eps: float,
    min_samples: int,
) -> List[int]:
    """Minimal DBSCAN implementation using a precomputed distance matrix.

    Args:
        dist_matrix: NxN pairwise distance matrix.
        eps: Maximum neighbourhood radius.
        min_samples: Minimum neighbours to be a core point.

    Returns:
        List of cluster labels (-1 = noise).
    """
    n = len(dist_matrix)
    labels = [-1] * n
    visited = [False] * n
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        neighbours = _region_query(dist_matrix, i, eps)
        if len(neighbours) < min_samples:
            # Noise (may be claimed by a cluster later)
            continue

        # Expand cluster
        labels[i] = cluster_id
        seed_set = list(neighbours)
        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbours = _region_query(dist_matrix, q, eps)
                if len(q_neighbours) >= min_samples:
                    seed_set.extend(q_neighbours)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1

        cluster_id += 1

    return labels


def _region_query(
    dist_matrix: np.ndarray,
    point_idx: int,
    eps: float,
) -> List[int]:
    """Return indices of all points within eps distance of point_idx."""
    return [
        j for j, d in enumerate(dist_matrix[point_idx])
        if d <= eps and j != point_idx
    ]


# ============================================================================
# Feature extraction & normalisation
# ============================================================================

def _extract_features(sessions: List[Session]) -> Optional[np.ndarray]:
    """Extract behavioural feature vectors from sessions.

    Features:
        [avg_magnitude, avg_cycle_duration, cycle_count, magnitude_cv]
    """
    if not sessions:
        return None

    rows = []
    for s in sessions:
        rows.append([
            s.avg_magnitude,
            s.avg_cycle_duration,
            float(s.cycle_count),
            s.magnitude_cv if s.magnitude_cv is not None else 0.0,
        ])

    return np.array(rows, dtype=float)


def _normalise_features(features: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation per column.

    Columns with zero variance are left as zero (constant features
    don't contribute to distance).
    """
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    # Avoid division by zero for constant columns
    std[std == 0] = 1.0
    return (features - mean) / std


# ============================================================================
# Cluster validation & confidence
# ============================================================================

def _is_valid_cluster(sessions: List[Session]) -> bool:
    """Check if a cluster represents a genuine recurring pattern.

    Rejects clusters with too much internal variance in magnitude or
    duration — those are likely random groupings rather than a real device.
    """
    if len(sessions) < RECURRING_PATTERN_MIN_SESSIONS:
        return False

    magnitudes = [s.avg_magnitude for s in sessions]
    durations = [s.avg_cycle_duration for s in sessions]

    mag_cv = _safe_cv(magnitudes)
    dur_cv = _safe_cv(durations)

    if mag_cv > RECURRING_PATTERN_MAX_INTERNAL_CV:
        return False
    if dur_cv > RECURRING_PATTERN_MAX_INTERNAL_CV:
        return False

    return True


def _recurring_pattern_confidence(
    cluster_sessions: List[Session],
    pattern_id: int,
) -> Tuple[float, dict]:
    """Compute confidence score for a recurring pattern cluster.

    Factors:
        cluster_size:  more recurrences → higher confidence
        magnitude_consistency: low magnitude CV → higher confidence
        duration_consistency: low duration CV → higher confidence
        recurrence_regularity: regular time gaps between sessions → higher
    """
    magnitudes = [s.avg_magnitude for s in cluster_sessions]
    durations = [s.avg_cycle_duration for s in cluster_sessions]

    # Cluster size score: 3→0.5, 6→0.75, 10+→1.0
    size = len(cluster_sessions)
    size_score = min(1.0, 0.5 + 0.05 * (size - RECURRING_PATTERN_MIN_SESSIONS))

    # Magnitude consistency: CV=0→1.0, CV=0.25→0.5
    mag_cv = _safe_cv(magnitudes)
    mag_score = max(0.0, 1.0 - 2.0 * mag_cv)

    # Duration consistency: CV=0→1.0, CV=0.25→0.5
    dur_cv = _safe_cv(durations)
    dur_score = max(0.0, 1.0 - 2.0 * dur_cv)

    # Recurrence regularity: how regular are the gaps between sessions?
    reg_score = _recurrence_regularity_score(cluster_sessions)

    confidence = round(float(np.mean([size_score, mag_score, dur_score, reg_score])), 2)

    breakdown = {
        'cluster_size': round(size_score, 2),
        'magnitude_consistency': round(mag_score, 2),
        'duration_consistency': round(dur_score, 2),
        'recurrence_regularity': round(reg_score, 2),
        'magnitude_cv': round(mag_cv, 3),
        'duration_cv': round(dur_cv, 3),
        'n_sessions': size,
        'avg_magnitude_w': round(float(np.mean(magnitudes)), 0),
        'avg_duration_min': round(float(np.mean(durations)), 1),
    }

    return confidence, breakdown


def _recurring_pattern_reason(
    cluster_sessions: List[Session],
    pattern_id: int,
    phase: str,
) -> str:
    """Generate descriptive reason string for a recurring pattern."""
    magnitudes = [s.avg_magnitude for s in cluster_sessions]
    durations = [s.avg_cycle_duration for s in cluster_sessions]
    n = len(cluster_sessions)

    avg_mag = np.mean(magnitudes)
    avg_dur = np.mean(durations)

    return (
        f"recurring pattern #{pattern_id}: "
        f"{n} sessions, ~{avg_dur:.0f}min duration, "
        f"~{avg_mag:.0f}W magnitude, phase {phase}"
    )


def _recurrence_regularity_score(sessions: List[Session]) -> float:
    """Score how regular the time intervals between session occurrences are.

    Returns 0.5 (neutral) if too few sessions for meaningful gap analysis.
    Returns 1.0 for perfectly regular gaps, lower for irregular.
    """
    if len(sessions) < 3:
        return 0.5

    # Sort by start time
    sorted_sessions = sorted(sessions, key=lambda s: s.start)
    gaps = []
    for i in range(len(sorted_sessions) - 1):
        gap_hours = (sorted_sessions[i + 1].start - sorted_sessions[i].start).total_seconds() / 3600
        if gap_hours > 0:
            gaps.append(gap_hours)

    if len(gaps) < 2:
        return 0.5

    gap_cv = _safe_cv(gaps)
    # CV=0 → 1.0, CV=1.0 → 0.5, CV=2.0+ → 0.0
    return max(0.0, min(1.0, 1.0 - 0.5 * gap_cv))


# ============================================================================
# Helpers
# ============================================================================

def _safe_cv(values: list) -> float:
    """Coefficient of variation, safe for constant or empty lists."""
    if not values or len(values) < 2:
        return 0.0
    mean = float(np.mean(values))
    if mean == 0:
        return 0.0
    return float(np.std(values) / mean)
