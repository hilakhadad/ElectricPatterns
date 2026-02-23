"""
Wave recovery pipeline step — Post-M1 orchestration.

Runs AFTER all M1 iterations (rectangle matching) and BEFORE M2 (identification).
Scans the final remaining power for wave-shaped patterns (sharp rise → gradual decay)
that M1 missed, extracts them, and saves M1-compatible match records.

Output goes to ``run_post/house_{id}/`` so it is separated from the rectangle runs
but seamlessly picked up by session_grouper.load_all_matches().
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..detection.wave_detector import WavePattern, detect_wave_patterns
from ..matching.phase_matcher import find_cross_phase_waves
from ..segmentation.wave_segmentor import extract_wave_power
from ..segmentation.validator import validate_wave_extraction

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

            # Accept — update remaining and create match record
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

    # --- 5. Save match records to run_post/ ---
    if all_match_records:
        _save_wave_matches(output_dir, house_id, all_match_records, threshold_schedule, run_logger)

    # --- 6. Save updated remaining to run_post/ ---
    if waves_extracted > 0:
        _save_updated_remaining(output_dir, house_id, monthly_data, updated_remaining, run_logger)

    summary = {
        'waves_detected': total_with_cross,
        'waves_extracted': waves_extracted,
        'phases': phase_stats,
    }
    run_logger.info(f"Wave recovery complete: {summary}")
    return summary


# ============================================================================
# Internal helpers
# ============================================================================

def _load_remaining(
    output_dir: Path,
    house_id: str,
    last_run: int,
    logger: logging.Logger,
) -> tuple:
    """Load remaining power from last M1 run's summarized files.

    Returns
    -------
    (remaining_by_phase, monthly_data) : tuple
        remaining_by_phase: {phase: pd.Series indexed by timestamp}
        monthly_data: list of (month_tag, DataFrame) for saving later
    """
    # Find the last run directory
    summarized_dir = None
    for run_idx in range(last_run, -1, -1):
        candidate = output_dir / f"run_{run_idx}" / f"house_{house_id}" / "summarized"
        if candidate.is_dir():
            summarized_dir = candidate
            break

    if summarized_dir is None:
        return {}, []

    pkl_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.pkl"))
    if not pkl_files:
        return {}, []

    all_dfs = []
    monthly_data = []
    for f in pkl_files:
        df = pd.read_pickle(f)
        # Extract month tag from filename: summarized_{house_id}_{MM}_{YYYY}.pkl
        month_tag = f.stem.replace(f"summarized_{house_id}_", "")
        monthly_data.append((month_tag, df))
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Ensure timestamp is datetime
    if 'timestamp' in combined.columns:
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined = combined.sort_values('timestamp').drop_duplicates('timestamp', keep='first')
        combined = combined.set_index('timestamp')

    remaining_by_phase = {}
    for phase in PHASES:
        col = f'remaining_{phase}'
        if col in combined.columns:
            remaining_by_phase[phase] = combined[col].dropna()

    logger.info(f"Wave recovery: loaded remaining from {summarized_dir} "
                f"({len(combined)} rows, phases: {list(remaining_by_phase.keys())})")

    return remaining_by_phase, monthly_data


def _save_wave_matches(
    output_dir: Path,
    house_id: str,
    match_records: List[dict],
    threshold_schedule: List[int],
    logger: logging.Logger,
):
    """Save wave match records grouped by month to run_post/house_{id}/matches/."""
    matches_df = pd.DataFrame(match_records)

    # Add iteration/threshold columns for M2 compatibility
    # Wave matches get iteration = len(threshold_schedule) (after all M1 iterations)
    matches_df['iteration'] = len(threshold_schedule)
    matches_df['threshold'] = 0

    # Group by month for per-month pickle files
    matches_df['_month'] = matches_df['on_start'].dt.strftime('%m_%Y')

    post_dir = output_dir / "run_post" / f"house_{house_id}" / "matches"
    post_dir.mkdir(parents=True, exist_ok=True)

    for month_tag, month_df in matches_df.groupby('_month'):
        month_df = month_df.drop(columns=['_month'])
        out_path = post_dir / f"matches_{house_id}_{month_tag}.pkl"
        month_df.to_pickle(out_path)
        logger.info(f"Wave recovery: saved {len(month_df)} matches to {out_path}")


def _save_updated_remaining(
    output_dir: Path,
    house_id: str,
    monthly_data: list,
    updated_remaining: Dict[str, pd.Series],
    logger: logging.Logger,
):
    """Save updated remaining (after wave extraction) to run_post/house_{id}/summarized/."""
    post_dir = output_dir / "run_post" / f"house_{house_id}" / "summarized"
    post_dir.mkdir(parents=True, exist_ok=True)

    for month_tag, original_df in monthly_data:
        df = original_df.copy()

        # Ensure timestamp for joining
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Update remaining columns from the extracted results
        for phase in PHASES:
            col = f'remaining_{phase}'
            if col in df.columns and phase in updated_remaining:
                updated = updated_remaining[phase]
                # Map updated values back by timestamp
                if 'timestamp' in df.columns:
                    ts_to_val = updated.to_dict()
                    df[col] = df['timestamp'].map(ts_to_val).fillna(df[col])

        out_path = post_dir / f"summarized_{house_id}_{month_tag}.pkl"
        df.to_pickle(out_path)

    logger.info(f"Wave recovery: saved updated remaining for {len(monthly_data)} months to {post_dir}")
