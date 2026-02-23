"""
Remaining-power event detection for false-negative analysis.

Scans the final remaining power (after all M1 iterations) per phase,
finds contiguous above-threshold regions, classifies their shape,
and returns a structured event list for the disaggregation report.

Shape classification (CV / zero-crossing based):
  - spike:   duration <= 3 min
  - cycling: >= 4 zero-crossings of median AND CV > 0.15  (AC-like)
  - flat:    CV <= 0.15  (constant load, boiler-like)
  - gradual: everything else  (ramp patterns)
"""
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MIN_THRESHOLD = 800   # watts above baseload
DEFAULT_MIN_DURATION = 3      # minutes
DEFAULT_MAX_DURATION = 1440   # minutes (24 hours) — no real device event lasts longer
CYCLING_MIN_CROSSINGS = 4     # zero-crossings to count as cycling
CYCLING_MIN_CV = 0.15         # minimum CV for cycling
FLAT_MAX_CV = 0.15            # maximum CV for flat
SPIKE_MAX_DURATION = 3        # minutes


def detect_remaining_events(
    baseline: pd.DataFrame,
    final: pd.DataFrame,
    phases: Dict[str, Dict],
    min_threshold: int = DEFAULT_MIN_THRESHOLD,
    min_duration: int = DEFAULT_MIN_DURATION,
    max_duration: int = DEFAULT_MAX_DURATION,
) -> Dict[str, Any]:
    """
    Detect events (contiguous above-threshold regions) in remaining power.

    Args:
        baseline: run_0 summarized DataFrame (has original_w1/w2/w3, timestamp)
        final: last-run summarized DataFrame (has remaining_w1/w2/w3, timestamp)
        phases: per-phase decomposition dict (from _compute_phase_decomposition),
                each phase has 'background_per_minute' key
        min_threshold: power above baseload (watts) to qualify as event
        min_duration: minimum contiguous minutes for an event
        max_duration: maximum minutes for a single event (longer regions are discarded
                      as they represent sustained elevated baseline, not device events)

    Returns:
        Dict with 'events' list and 'summary' dict.
    """
    all_events: List[Dict[str, Any]] = []

    for phase in ['w1', 'w2', 'w3']:
        remain_col = f'remaining_{phase}'
        if remain_col not in final.columns:
            continue

        remaining = final[remain_col].values.astype(float)
        timestamps = pd.to_datetime(final['timestamp']).values

        baseload = phases.get(phase, {}).get('background_per_minute', 0)
        above_base = np.clip(remaining - baseload, 0, None)

        regions = _find_above_threshold_regions(
            above_base, timestamps, min_threshold, min_duration, max_duration,
        )

        for start_idx, end_idx in regions:
            event = _build_event_dict(
                above_base[start_idx:end_idx + 1],
                remaining[start_idx:end_idx + 1],
                timestamps[start_idx:end_idx + 1],
                phase,
                baseload,
            )
            all_events.append(event)

    # Sort by start time
    all_events.sort(key=lambda e: e['start'])

    # Summary
    by_phase = {'w1': 0, 'w2': 0, 'w3': 0}
    by_shape = {'flat': 0, 'cycling': 0, 'spike': 0, 'gradual': 0}
    total_energy = 0.0

    for ev in all_events:
        by_phase[ev['phase']] += 1
        by_shape[ev['shape']] += 1
        total_energy += ev['energy_wh']

    return {
        'events': all_events,
        'summary': {
            'total_events': len(all_events),
            'by_phase': by_phase,
            'by_shape': by_shape,
            'total_energy_wh': round(total_energy, 1),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_above_threshold_regions(
    above_base: np.ndarray,
    timestamps: np.ndarray,
    threshold: float,
    min_duration: int,
    max_duration: int,
) -> List[Tuple[int, int]]:
    """
    Find contiguous regions where above_base >= threshold.

    Breaks regions at:
    - Points below threshold
    - Timestamp gaps > 2 minutes (non-contiguous data, e.g. month boundaries)

    Filters by min_duration and max_duration (in minutes = array elements).

    Returns list of (start_index, end_index) tuples (inclusive).
    """
    mask = above_base >= threshold

    # Detect timestamp gaps > 2 minutes (marks boundary between data chunks)
    ts_ns = timestamps.astype('datetime64[ns]')
    diffs = np.diff(ts_ns).astype('timedelta64[m]').astype(int)  # minutes
    gap_mask = np.zeros(len(timestamps), dtype=bool)
    gap_mask[0] = False  # first element never a gap
    gap_mask[1:] = diffs > 2  # gap if > 2 min between consecutive timestamps

    regions = []
    start = None

    for i in range(len(mask)):
        if gap_mask[i] and start is not None:
            # Timestamp gap — close current region
            duration = i - start
            if min_duration <= duration <= max_duration:
                regions.append((start, i - 1))
            start = None

        if mask[i] and start is None:
            start = i
        elif not mask[i] and start is not None:
            duration = i - start
            if min_duration <= duration <= max_duration:
                regions.append((start, i - 1))
            start = None

    # Handle region that extends to end
    if start is not None:
        duration = len(mask) - start
        if min_duration <= duration <= max_duration:
            regions.append((start, len(mask) - 1))

    return regions


def _classify_shape(power_values: np.ndarray) -> str:
    """
    Classify the shape of a remaining-power event.

    Args:
        power_values: above-baseload power values for the event region

    Returns:
        'spike', 'cycling', 'flat', or 'gradual'
    """
    duration = len(power_values)

    if duration <= SPIKE_MAX_DURATION:
        return 'spike'

    mean_val = np.mean(power_values)
    if mean_val <= 0:
        return 'spike'

    cv = float(np.std(power_values) / mean_val)

    # Zero-crossings of (signal - median)
    median_val = np.median(power_values)
    centered = power_values - median_val
    crossings = int(np.sum(np.diff(np.sign(centered)) != 0))

    if crossings >= CYCLING_MIN_CROSSINGS and cv > CYCLING_MIN_CV:
        return 'cycling'

    if cv <= FLAT_MAX_CV:
        return 'flat'

    return 'gradual'


def _build_event_dict(
    above_base_region: np.ndarray,
    remaining_region: np.ndarray,
    timestamps: np.ndarray,
    phase: str,
    baseload: float,
) -> Dict[str, Any]:
    """Build event metadata dict from a detected region."""
    start_ts = pd.Timestamp(timestamps[0])
    end_ts = pd.Timestamp(timestamps[-1])
    duration = len(above_base_region)

    peak = float(np.max(remaining_region))
    avg = float(np.mean(remaining_region))
    energy_wh = float(np.sum(remaining_region) / 60)

    mean_ab = float(np.mean(above_base_region))
    cv = float(np.std(above_base_region) / mean_ab) if mean_ab > 0 else 0.0

    shape = _classify_shape(above_base_region)

    return {
        'phase': phase,
        'start': start_ts,
        'end': end_ts,
        'duration_min': duration,
        'peak_power': round(peak, 0),
        'avg_power': round(avg, 0),
        'shape': shape,
        'cv': round(cv, 3),
        'energy_wh': round(energy_wh, 1),
    }
