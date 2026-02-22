"""
Per-activation confidence scoring for device classification.

Assigns a confidence score (0.0-1.0) to every activation based on:
  - Tag quality (match type: EXACT/CLOSE/APPROX/LOOSE, NOISY/PARTIAL prefixes)
  - Iteration (earlier iterations = higher confidence)
  - Magnitude fit (how well magnitude matches its device type distribution)
  - Duration fit (how well duration matches expected patterns)
  - Seasonal fit (does the activation occur in the expected season?)
  - Phase consistency (does this phase match the dominant phase for this device type?)

Input: device_activations_{house_id}.json
Output: per-activation confidence scores and summary statistics.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Israeli seasons: warm = May-October, cool = November-April
WARM_MONTHS = {5, 6, 7, 8, 9, 10}

# Confidence component weights
WEIGHTS = {
    'tag_quality': 0.25,
    'iteration': 0.15,
    'magnitude_fit': 0.20,
    'duration_fit': 0.15,
    'seasonal_fit': 0.10,
    'phase_consistency': 0.15,
}

# Confidence tier boundaries
TIER_HIGH = 0.80
TIER_MEDIUM = 0.40


# ============================================================================
# Main entry point
# ============================================================================

def calculate_confidence_scores(
    experiment_dir: Path,
    house_id: str,
) -> Dict[str, Any]:
    """
    Calculate confidence scores for all activations in a house.

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID string

    Returns:
        Dict with per-activation confidence scores and summary.
    """
    activations = _load_activations(experiment_dir, house_id)
    if activations is None:
        return _empty_confidence(house_id)

    matched = [a for a in activations if a.get('match_type') == 'matched']
    if not matched:
        return _empty_confidence(house_id)

    # Pre-compute population statistics for magnitude/duration fit
    type_stats = _compute_type_statistics(matched)

    # Pre-compute dominant phase per device type in this house
    dominant_phase = _compute_dominant_phases(matched)

    # Score each activation
    scored = []
    for idx, activation in enumerate(matched):
        components = _score_activation(activation, type_stats, dominant_phase)
        confidence = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)

        # Matched activation base boost
        confidence = min(1.0, confidence + 0.05)

        # Unclassified always get 0.0
        if activation.get('device_type', 'unclassified') == 'unclassified':
            confidence = 0.0
            components = {k: 0.0 for k in WEIGHTS}

        # Central AC 3-phase bonus
        if activation.get('device_type') == 'central_ac':
            phase = activation.get('phase', '')
            if phase and len(phase) > 2:  # multi-phase indicator
                confidence = min(1.0, confidence + 0.1)

        tier = _confidence_tier(confidence)

        scored.append({
            'index': idx,
            'confidence': round(confidence, 3),
            'confidence_tier': tier,
            'components': {k: round(v, 3) for k, v in components.items()},
        })

    # Summary statistics
    confidences = [s['confidence'] for s in scored]
    high_count = sum(1 for c in confidences if c >= TIER_HIGH)
    medium_count = sum(1 for c in confidences if TIER_MEDIUM <= c < TIER_HIGH)
    low_count = sum(1 for c in confidences if c < TIER_MEDIUM)

    return {
        'house_id': house_id,
        'total_activations': len(scored),
        'confidence_summary': {
            'mean': round(float(np.mean(confidences)), 3) if confidences else 0,
            'median': round(float(np.median(confidences)), 3) if confidences else 0,
            'high_count': high_count,
            'medium_count': medium_count,
            'low_count': low_count,
        },
        'activations': scored,
    }


# ============================================================================
# Per-activation scoring
# ============================================================================

def _score_activation(
    activation: Dict,
    type_stats: Dict,
    dominant_phase: Dict,
) -> Dict[str, float]:
    """Score a single activation across all confidence components."""
    dtype = activation.get('device_type', 'unclassified')
    tag = activation.get('tag', '')

    return {
        'tag_quality': _score_tag_quality(tag),
        'iteration': _score_iteration(activation.get('iteration', 0)),
        'magnitude_fit': _score_magnitude_fit(activation, type_stats, dtype),
        'duration_fit': _score_duration_fit(activation, type_stats, dtype),
        'seasonal_fit': _score_seasonal_fit(activation, dtype),
        'phase_consistency': _score_phase_consistency(activation, dominant_phase, dtype),
    }


def _score_tag_quality(tag: str) -> float:
    """
    Score match quality from tag string.

    Tag format: [NOISY-|PARTIAL-]{EXACT|CLOSE|APPROX|LOOSE}-{duration}[-CORRECTED][-TAIL]
    """
    if not tag:
        return 0.3

    tag_upper = tag.upper()

    # Base score from magnitude quality
    if 'EXACT' in tag_upper:
        base = 1.0
    elif 'CLOSE' in tag_upper:
        base = 0.8
    elif 'APPROX' in tag_upper:
        base = 0.6
    elif 'LOOSE' in tag_upper:
        base = 0.4
    else:
        base = 0.5

    # Prefix multiplier
    if 'PARTIAL' in tag_upper:
        base *= 0.5
    elif 'NOISY' in tag_upper:
        base *= 0.7

    # Correction penalty
    if 'CORRECTED' in tag_upper:
        base *= 0.9

    return min(1.0, max(0.0, base))


def _score_iteration(iteration: int) -> float:
    """Earlier iterations detect larger, more distinct devices."""
    scores = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4}
    return scores.get(iteration, 0.3)


def _score_magnitude_fit(
    activation: Dict,
    type_stats: Dict,
    dtype: str,
) -> float:
    """How well this activation's magnitude fits its device type distribution."""
    if dtype not in type_stats or dtype == 'unclassified':
        return 0.5

    stats = type_stats[dtype]
    mag = abs(activation.get('magnitude', 0))
    median = stats.get('mag_median', 0)
    mad = stats.get('mag_mad', 1)

    if mad == 0:
        return 1.0 if mag == median else 0.5

    deviation = abs(mag - median) / mad
    return max(0.0, 1.0 - min(1.0, deviation / 3.0))


def _score_duration_fit(
    activation: Dict,
    type_stats: Dict,
    dtype: str,
) -> float:
    """How well the duration fits expected patterns for this device type."""
    if dtype not in type_stats or dtype == 'unclassified':
        return 0.5

    stats = type_stats[dtype]
    dur = activation.get('duration', 0)
    median = stats.get('dur_median', 0)
    mad = stats.get('dur_mad', 1)

    if mad == 0:
        return 1.0 if dur == median else 0.5

    deviation = abs(dur - median) / mad
    return max(0.0, 1.0 - min(1.0, deviation / 3.0))


def _score_seasonal_fit(activation: Dict, dtype: str) -> float:
    """
    Score based on whether activation month matches expected season.

    AC should be in warm months, boiler year-round.
    """
    month = _extract_month(activation)
    if month is None:
        return 0.5

    is_warm = month in WARM_MONTHS

    if dtype in ('regular_ac', 'central_ac'):
        if is_warm:
            return 1.0
        # Shoulder months (April=4, November=11)
        if month in (4, 11):
            return 0.7
        return 0.3  # AC in deep winter
    elif dtype == 'boiler':
        return 0.8  # Boiler is year-round, slight seasonal variation is fine
    else:
        return 0.5  # Unclassified/other: neutral


def _score_phase_consistency(
    activation: Dict,
    dominant_phase: Dict,
    dtype: str,
) -> float:
    """
    Score based on whether this phase matches the most common phase for this device type.

    Central AC is multi-phase by definition, so always gets 1.0.
    """
    if dtype == 'central_ac':
        return 1.0  # Multi-phase by definition

    if dtype not in dominant_phase:
        return 0.5

    phase = activation.get('phase', '')
    if phase == dominant_phase[dtype]:
        return 1.0
    return 0.5


# ============================================================================
# Population statistics helpers
# ============================================================================

def _compute_type_statistics(matched: List[Dict]) -> Dict[str, Dict]:
    """Compute per-device-type magnitude and duration statistics."""
    by_type = defaultdict(list)
    for a in matched:
        dtype = a.get('device_type', 'unclassified')
        if dtype != 'unclassified':
            by_type[dtype].append(a)

    stats = {}
    for dtype, acts in by_type.items():
        magnitudes = [abs(a.get('magnitude', 0)) for a in acts]
        durations = [a.get('duration', 0) for a in acts]

        mag_arr = np.array(magnitudes) if magnitudes else np.array([0])
        dur_arr = np.array(durations) if durations else np.array([0])

        stats[dtype] = {
            'mag_median': float(np.median(mag_arr)),
            'mag_mad': float(np.median(np.abs(mag_arr - np.median(mag_arr)))) or 1.0,
            'dur_median': float(np.median(dur_arr)),
            'dur_mad': float(np.median(np.abs(dur_arr - np.median(dur_arr)))) or 1.0,
        }

    return stats


def _compute_dominant_phases(matched: List[Dict]) -> Dict[str, str]:
    """Find the most common phase for each device type."""
    phase_counts = defaultdict(lambda: defaultdict(int))
    for a in matched:
        dtype = a.get('device_type', 'unclassified')
        phase = a.get('phase', '')
        if dtype != 'unclassified' and phase:
            phase_counts[dtype][phase] += 1

    dominant = {}
    for dtype, counts in phase_counts.items():
        if counts:
            dominant[dtype] = max(counts, key=counts.get)

    return dominant


# ============================================================================
# Helpers
# ============================================================================

def _confidence_tier(confidence: float) -> str:
    """Convert confidence score to tier name."""
    if confidence >= TIER_HIGH:
        return 'high'
    if confidence >= TIER_MEDIUM:
        return 'medium'
    return 'low'


def _load_activations(experiment_dir: Path, house_id: str) -> Optional[List[Dict]]:
    """Load device activations list from JSON."""
    experiment_dir = Path(experiment_dir)
    json_path = experiment_dir / "device_activations" / f"device_activations_{house_id}.json"
    if not json_path.exists():
        json_path = experiment_dir / f"device_activations_{house_id}.json"

    if not json_path.exists():
        logger.warning(f"Device activations JSON not found for house {house_id}")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('activations', [])
    except Exception as e:
        logger.warning(f"Failed to load device activations: {e}")
        return None


def _extract_month(activation: Dict) -> Optional[int]:
    """Extract month number (1-12) from activation."""
    ts = activation.get('on_start', '')
    if not ts:
        return None
    try:
        return int(ts.split('-')[1])
    except (ValueError, IndexError):
        return None


def _empty_confidence(house_id: str) -> Dict[str, Any]:
    """Return empty confidence structure."""
    return {
        'house_id': house_id,
        'total_activations': 0,
        'confidence_summary': {
            'mean': 0, 'median': 0,
            'high_count': 0, 'medium_count': 0, 'low_count': 0,
        },
        'activations': [],
    }
