"""
Hole repair for wave recovery.

Finds rectangle matches that created holes in remaining power (wave-shaped
events extracted as flat rectangles), undoes the rectangle extraction,
re-detects the wave, and re-extracts it properly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..detection.wave_detector import WavePattern
from ..segmentation.wave_segmentor import extract_wave_power
from .wave_io import _load_rectangle_matches

# Tags that suggest a wave was forced into a rectangle:
# ON magnitude != OFF magnitude (APPROX/LOOSE) + long duration (EXTENDED)
_HOLE_CANDIDATE_TAG_PATTERN = r'(?:APPROX|LOOSE).*EXTENDED'

# Hole detection: remaining during event must drop significantly from edge levels.
# A real hole: remaining drops from edge level (before/after) to near-zero during event.
# A correctly-extracted flat device (e.g. boiler): remaining is similar before/during/after.
_HOLE_MIN_EDGE_WATTS = 100    # edges must have meaningful remaining (W)
_HOLE_DROP_FRACTION = 0.50    # event median must be < 50% of edge level to count as hole
_HOLE_MIN_EDGE_RATIO = 0.15   # edge level must be >= 15% of on_magnitude (filters background noise)


def _repair_wave_holes(
    output_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
    updated_remaining: Dict[str, pd.Series],
    config,
    logger: logging.Logger,
) -> tuple:
    """
    Find rectangle matches that created holes, undo the rectangle extraction,
    re-detect as wave, and re-extract properly.

    Parameters
    ----------
    output_dir : Path
        Experiment output directory.
    house_id : str
        House identifier.
    threshold_schedule : list[int]
        M1 threshold schedule.
    updated_remaining : dict
        {phase: pd.Series} -- will be modified in-place for repaired matches.
    config : ExperimentConfig
        Wave detection config.
    logger : logging.Logger

    Returns
    -------
    (wave_match_records, repaired_ids) : tuple
        wave_match_records: list of new WAVE match dicts to save
        repaired_ids: list of (run_number, on_event_id) tuples of replaced matches
    """
    # Import here to avoid circular dependency at module level
    from .wave_recovery_step import _wave_to_match_record

    # Load all rectangle matches
    all_matches = _load_rectangle_matches(output_dir, house_id, threshold_schedule, logger)
    if all_matches.empty:
        return [], []

    # Filter candidates by tag:
    # 1. APPROX/LOOSE + EXTENDED: ON != OFF magnitude in long events (wave signature)
    # 2. CORRECTED + EXTENDED: validator reduced magnitude (rectangle hit remaining limit)
    approx_loose = all_matches['tag'].str.contains(_HOLE_CANDIDATE_TAG_PATTERN, na=False, regex=True)
    corrected_ext = (all_matches['tag'].str.contains('CORRECTED', na=False) &
                     all_matches['tag'].str.contains('EXTENDED', na=False))
    wave_mask = all_matches['tag'].str.startswith('WAVE-', na=False)
    candidates = all_matches[(approx_loose | corrected_ext) & ~wave_mask]

    if candidates.empty:
        logger.info(f"  Hole repair: 0 candidates (no APPROX/LOOSE EXTENDED matches)")
        return [], []

    logger.info(f"  Hole repair: {len(candidates)} candidate matches to check")

    wave_records = []
    repaired_ids = []

    for idx, match in candidates.iterrows():
        phase = match['phase']
        remaining = updated_remaining.get(phase)
        if remaining is None:
            continue

        on_start = pd.Timestamp(match['on_start'])
        on_end = pd.Timestamp(match['on_end'])
        off_start = pd.Timestamp(match['off_start'])
        off_end = pd.Timestamp(match['off_end'])

        # Hole detection: compare remaining DURING the event with remaining
        # BEFORE (5 min before on_start) and AFTER (5 min after off_end).
        event_mask = (remaining.index > on_end) & (remaining.index < off_start)
        event_region = remaining[event_mask]

        if len(event_region) < 3:
            continue

        on_mag = abs(match['on_magnitude'])

        pre_mask = (remaining.index >= on_start - pd.Timedelta(minutes=5)) & \
                   (remaining.index < on_start)
        post_mask = (remaining.index > off_end) & \
                    (remaining.index <= off_end + pd.Timedelta(minutes=5))
        pre_level = float(remaining[pre_mask].median()) if pre_mask.any() else 0.0
        post_level = float(remaining[post_mask].median()) if post_mask.any() else 0.0
        edge_level = (pre_level + post_level) / 2 if (pre_mask.any() and post_mask.any()) \
            else max(pre_level, post_level)
        event_median = float(event_region.median())

        # Three conditions must ALL hold for a real hole:
        # 1. Edge level is meaningful (>100W)
        # 2. Event median dropped significantly from edge level (<50%)
        # 3. Edge level is significant relative to device magnitude (>15%)
        edge_ratio = edge_level / on_mag if on_mag > 0 else 0
        no_hole = (
            edge_level < _HOLE_MIN_EDGE_WATTS or
            event_median >= edge_level * _HOLE_DROP_FRACTION or
            edge_ratio < _HOLE_MIN_EDGE_RATIO
        )
        if no_hole:
            logger.info(
                f"  Hole repair: no hole in {match.get('on_event_id', '?')} on {phase} "
                f"(edge={edge_level:.0f}W, event_median={event_median:.0f}W, "
                f"edge/mag={edge_ratio:.1%}) -- skipping"
            )
            continue

        # Found a hole! Build wave profile directly from match metadata.
        off_mag = abs(match.get('off_magnitude', 0))
        logger.info(
            f"  Hole repair: found hole in {match['on_event_id']} on {phase} "
            f"({on_start} -> {off_end}, edge={edge_level:.0f}W, event_median={event_median:.0f}W, "
            f"on_mag={on_mag:.0f}W, off_mag={off_mag:.0f}W)"
        )

        # Step 1: Restore the rectangle extraction (add constant power back)
        restored = remaining.copy()
        on_mask_local = (restored.index >= on_start) & (restored.index <= on_end)
        full_event_mask = (restored.index >= on_start) & (restored.index <= off_end)
        restored.loc[full_event_mask] += on_mag

        # Step 2: Estimate baseline from the minutes just before the match
        pre_match_mask = (remaining.index >= on_start - pd.Timedelta(minutes=5)) & \
                         (remaining.index < on_start)
        pre_match = remaining[pre_match_mask]
        baseline_power = float(pre_match.mean()) if len(pre_match) > 0 else 0.0

        # Step 3: Build a linear-decay wave profile from on_mag down to off_mag
        full_indices = restored.index[full_event_mask]
        duration_minutes = len(full_indices)
        if duration_minutes < 3:
            continue

        # Peak is on_magnitude above baseline, end is off_magnitude above baseline
        peak_above_baseline = on_mag
        end_above_baseline = min(off_mag, on_mag * 0.1)  # off_mag or 10% of on_mag
        wave_profile = np.linspace(peak_above_baseline, end_above_baseline, duration_minutes)
        wave_profile = np.clip(wave_profile, 0, None)

        peak_time = on_end if on_end > on_start else on_start + pd.Timedelta(minutes=1)
        best_wave = WavePattern(
            start=on_start,
            peak_time=peak_time,
            end=off_end,
            phase=phase,
            peak_power=float(baseline_power + peak_above_baseline),
            baseline_power=float(baseline_power),
            duration_minutes=duration_minutes,
            wave_profile=wave_profile,
        )

        # Step 4: Extract wave power from the restored remaining
        extracted, new_remaining_region = extract_wave_power(restored, best_wave)
        total_extracted = float(extracted.sum())

        if total_extracted < 100:
            logger.info(f"    Extracted too little power ({total_extracted:.0f}W-min), skipping")
            continue

        # Step 5: Update remaining -- replace with wave-extracted version
        updated_remaining[phase].loc[full_event_mask] = new_remaining_region.loc[full_event_mask]

        # Create wave match record
        wave_record = _wave_to_match_record(best_wave, total_extracted)
        wave_record['tag'] = 'WAVE-REPAIR-' + wave_record['tag'].replace('WAVE-', '')
        wave_records.append(wave_record)

        # Track the original match for removal
        run_number = match.get('iteration', 0)
        repaired_ids.append((int(run_number), match['on_event_id']))

        logger.info(
            f"    Repaired: wave on {phase} ({best_wave.start} -> {best_wave.end}, "
            f"{best_wave.duration_minutes} min, peak={best_wave.peak_power:.0f}W, "
            f"extracted={total_extracted:.0f}W-min)"
        )

    return wave_records, repaired_ids
