"""
Wave recovery pipeline step -- Post-M1 orchestration.

Runs AFTER all M1 iterations (rectangle matching) and BEFORE M2 (identification).

Two modes of operation:
  A) **New wave detection** -- scans the final remaining power for wave-shaped
     patterns (sharp rise -> gradual decay) that M1 missed entirely.
  B) **Hole repair** -- finds rectangle matches whose tags suggest a wave was
     extracted as a rectangle (APPROX/LOOSE + EXTENDED), detects the "hole"
     (remaining ~= 0 in the event region), restores the approximate rectangle
     power, re-detects the wave, and re-extracts it properly.

Output goes to ``run_post/house_{id}/`` so it is separated from the rectangle runs
but seamlessly picked up by session_grouper.load_all_matches().

Implementation is split across:
  - wave_recovery_step.py (this file) -- orchestration + tag helpers
  - wave_io.py -- I/O helpers (load/save matches and remaining)
  - hole_repair.py -- hole detection and repair logic
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ..detection.wave_detector import WavePattern, detect_wave_patterns
from ..matching.phase_matcher import find_cross_phase_waves
from ..segmentation.wave_segmentor import extract_wave_power
from ..segmentation.validator import validate_wave_extraction

from .wave_io import (
    _load_remaining,
    _save_wave_matches,
    _save_updated_remaining,
    _remove_repaired_matches,
)
from .hole_repair import _repair_wave_holes

PHASES = ['w1', 'w2', 'w3']

# Duration buckets for tags (same convention as rectangle matching)
_DURATION_TAGS = [
    (2, 'SPIKE'),
    (5, 'QUICK'),
    (25, 'MEDIUM'),
    (float('inf'), 'EXTENDED'),
]


def _wave_tag(duration_minutes: int) -> str:
    """Build a WAVE-* tag matching the rectangle tag convention."""
    for limit, label in _DURATION_TAGS:
        if duration_minutes <= limit:
            return f"WAVE-{label}"
    return "WAVE-EXTENDED"


def _wave_to_match_record(wave: WavePattern, extracted_power: float) -> dict:
    """
    Convert a WavePattern + extracted power into an M1-compatible match dict.

    Uses the same columns as ``matching_step._format_match()`` so that
    ``session_grouper.load_all_matches()`` handles wave matches identically.
    """
    tag = _wave_tag(wave.duration_minutes)
    on_magnitude = wave.peak_power - wave.baseline_power
    # OFF magnitude approximated as the decay endpoint above baseline
    # (for waves this is near zero, but use actual extracted mean)
    off_magnitude = -(on_magnitude * 0.5)  # Negative, approximate

    return {
        'on_event_id': f"wave_{uuid.uuid4().hex[:8]}",
        'off_event_id': f"wave_{uuid.uuid4().hex[:8]}",
        'on_start': wave.start,
        'on_end': wave.peak_time,
        'off_start': wave.end - pd.Timedelta(minutes=1),
        'off_end': wave.end,
        'duration': float(wave.duration_minutes),
        'on_magnitude': float(on_magnitude),
        'off_magnitude': float(off_magnitude),
        'correction': 0,
        'tag': tag,
        'phase': wave.phase,
    }


def process_wave_recovery(
    output_path: str,
    house_id: str,
    threshold_schedule: List[int],
    config,
    run_logger: logging.Logger,
) -> dict:
    """
    Post-M1 wave recovery step.

    1. Load final remaining from the last M1 run.
    2. Detect wave patterns per phase.
    3. Cross-phase matching: for each wave, search other phases.
    4. Extract wave power, validate.
    5. Save M1-compatible match records to ``run_post/``.
    6. Save updated remaining to ``run_post/``.

    Parameters
    ----------
    output_path : str
        Experiment output directory (contains run_0/, run_1/, ...).
    house_id : str
        House identifier.
    threshold_schedule : list[int]
        M1 threshold schedule (used to find last run).
    config : ExperimentConfig
        Must have wave recovery fields (wave_min_rise_watts, etc.).
    run_logger : logging.Logger
        Logger instance.

    Returns
    -------
    dict
        Summary: {'waves_detected': int, 'waves_extracted': int, 'phases': {...}}.
    """
    output_dir = Path(output_path)
    last_run = len(threshold_schedule) - 1

    # --- 1. Load final remaining from last M1 run ---
    t0 = time.time()
    remaining_by_phase, monthly_data = _load_remaining(output_dir, house_id, last_run, run_logger)
    if not remaining_by_phase:
        run_logger.info("Wave recovery: no remaining data found, skipping")
        return {'waves_detected': 0, 'waves_extracted': 0}
    run_logger.info(f"  Load remaining ({time.time() - t0:.1f}s)")

    # --- 2. Detect waves per phase ---
    t0 = time.time()
    detected: Dict[str, List[WavePattern]] = {}
    for phase in PHASES:
        if phase not in remaining_by_phase:
            detected[phase] = []
            continue
        t_phase = time.time()
        waves = detect_wave_patterns(remaining_by_phase[phase], phase, config, logger=run_logger)
        detected[phase] = waves
        run_logger.info(f"  {phase}: {len(waves)} waves detected ({time.time() - t_phase:.1f}s)")

    total_detected = sum(len(v) for v in detected.values())
    run_logger.info(f"  Detection total: {total_detected} waves ({time.time() - t0:.1f}s)")
    if total_detected == 0:
        run_logger.info("Wave recovery: no wave patterns found")
        return {'waves_detected': 0, 'waves_extracted': 0}

    # --- 3. Cross-phase matching ---
    t0 = time.time()
    extra = find_cross_phase_waves(detected, remaining_by_phase, config, logger=run_logger)
    for phase, extra_waves in extra.items():
        if extra_waves:
            detected[phase].extend(extra_waves)
            run_logger.info(f"  {phase}: {len(extra_waves)} cross-phase waves added")
    run_logger.info(f"  Cross-phase matching ({time.time() - t0:.1f}s)")

    total_with_cross = sum(len(v) for v in detected.values())

    # --- 4. Extract + validate ---
    t0 = time.time()
    all_match_records: List[dict] = []
    updated_remaining = {p: remaining_by_phase[p].copy() for p in remaining_by_phase}
    waves_extracted = 0
    phase_stats = {}

    for phase in PHASES:
        if phase not in detected or not detected[phase]:
            phase_stats[phase] = {'detected': 0, 'extracted': 0}
            continue

        # Sort by start time (important for sequential extraction)
        phase_waves = sorted(detected[phase], key=lambda w: w.start)
        extracted_count = 0

        for wave in phase_waves:
            remaining_series = updated_remaining.get(phase)
            if remaining_series is None:
                continue

            # Extract
            extracted_power, new_remaining = extract_wave_power(remaining_series, wave)

            # Validate
            valid, reason = validate_wave_extraction(
                remaining_series, new_remaining, extracted_power, wave
            )
            if not valid:
                run_logger.warning(
                    f"Wave recovery: rejected wave on {phase} "
                    f"({wave.start} -> {wave.end}): {reason}"
                )
                continue

            # Accept -- update remaining and create match record
            updated_remaining[phase] = new_remaining
            total_extracted_watts = float(extracted_power.sum())
            match_record = _wave_to_match_record(wave, total_extracted_watts)
            all_match_records.append(match_record)
            extracted_count += 1
            waves_extracted += 1

            run_logger.info(
                f"Wave recovery: extracted wave on {phase} "
                f"({wave.start} -> {wave.end}, {wave.duration_minutes} min, "
                f"peak={wave.peak_power:.0f}W, total={total_extracted_watts:.0f}W-min)"
            )

        phase_stats[phase] = {'detected': len(phase_waves), 'extracted': extracted_count}

    run_logger.info(f"  Extraction + validation: {waves_extracted} accepted ({time.time() - t0:.1f}s)")

    # --- 4b. Hole repair: fix rectangle matches that extracted waves as rectangles ---
    t0 = time.time()
    repaired_records, repaired_ids = _repair_wave_holes(
        output_dir, house_id, threshold_schedule, updated_remaining, config, run_logger,
    )
    holes_repaired = len(repaired_records)
    if holes_repaired > 0:
        all_match_records.extend(repaired_records)
        run_logger.info(f"  Hole repair: {holes_repaired} matches repaired ({time.time() - t0:.1f}s)")
    else:
        run_logger.info(f"  Hole repair: no candidates found ({time.time() - t0:.1f}s)")

    # --- 5. Save match records to run_post/ ---
    if all_match_records:
        _save_wave_matches(output_dir, house_id, all_match_records, threshold_schedule, run_logger)

    # --- 5b. Remove original rectangle matches that were replaced by wave repairs ---
    if repaired_ids:
        _remove_repaired_matches(output_dir, house_id, threshold_schedule, repaired_ids, run_logger)

    # --- 6. Save updated remaining to run_post/ ---
    if waves_extracted > 0 or holes_repaired > 0:
        _save_updated_remaining(output_dir, house_id, monthly_data, updated_remaining, run_logger)

    summary = {
        'waves_detected': total_with_cross,
        'waves_extracted': waves_extracted,
        'holes_repaired': holes_repaired,
        'phases': phase_stats,
    }
    run_logger.info(f"Wave recovery complete: {summary}")
    return summary
