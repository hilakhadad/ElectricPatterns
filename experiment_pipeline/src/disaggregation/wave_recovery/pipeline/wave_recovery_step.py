"""
Wave recovery pipeline step — Post-M1 orchestration.

Runs AFTER all M1 iterations (rectangle matching) and BEFORE M2 (identification).

Two modes of operation:
  A) **New wave detection** — scans the final remaining power for wave-shaped
     patterns (sharp rise → gradual decay) that M1 missed entirely.
  B) **Hole repair** — finds rectangle matches whose tags suggest a wave was
     extracted as a rectangle (APPROX/LOOSE + EXTENDED), detects the "hole"
     (remaining ≈ 0 in the event region), restores the approximate rectangle
     power, re-detects the wave, and re-extracts it properly.

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


# ============================================================================
# Hole repair — fix rectangle matches that extracted waves as rectangles
# ============================================================================

# Tags that suggest a wave was forced into a rectangle:
# ON magnitude ≠ OFF magnitude (APPROX/LOOSE) + long duration (EXTENDED)
_HOLE_CANDIDATE_TAG_PATTERN = r'(?:APPROX|LOOSE).*EXTENDED'

# Hole detection: remaining during event must drop significantly from edge levels.
# A real hole: remaining drops from edge level (before/after) to near-zero during event.
# A correctly-extracted flat device (e.g. boiler): remaining is similar before/during/after.
_HOLE_MIN_EDGE_WATTS = 100    # edges must have meaningful remaining (W)
_HOLE_DROP_FRACTION = 0.50    # event median must be < 50% of edge level to count as hole


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
        {phase: pd.Series} — will be modified in-place for repaired matches.
    config : ExperimentConfig
        Wave detection config.
    logger : logging.Logger

    Returns
    -------
    (wave_match_records, repaired_ids) : tuple
        wave_match_records: list of new WAVE match dicts to save
        repaired_ids: list of (run_number, on_event_id) tuples of replaced matches
    """
    # Load all rectangle matches
    all_matches = _load_rectangle_matches(output_dir, house_id, threshold_schedule, logger)
    if all_matches.empty:
        return [], []

    # Filter candidates by tag:
    # 1. APPROX/LOOSE + EXTENDED: ON ≠ OFF magnitude in long events (wave signature)
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
        # A real hole: remaining drops from meaningful edge level to near-zero.
        # A correctly-extracted flat device (e.g. boiler): remaining is similar
        # before, during, and after — no drop, no hole, no repair needed.
        event_mask = (remaining.index > on_end) & (remaining.index < off_start)
        event_region = remaining[event_mask]

        if len(event_region) < 3:
            continue

        pre_mask = (remaining.index >= on_start - pd.Timedelta(minutes=5)) & \
                   (remaining.index < on_start)
        post_mask = (remaining.index > off_end) & \
                    (remaining.index <= off_end + pd.Timedelta(minutes=5))
        pre_level = float(remaining[pre_mask].median()) if pre_mask.any() else 0.0
        post_level = float(remaining[post_mask].median()) if post_mask.any() else 0.0
        edge_level = (pre_level + post_level) / 2 if (pre_mask.any() and post_mask.any()) \
            else max(pre_level, post_level)
        event_median = float(event_region.median())

        if edge_level < _HOLE_MIN_EDGE_WATTS or event_median >= edge_level * _HOLE_DROP_FRACTION:
            logger.info(
                f"  Hole repair: no hole in {match.get('on_event_id', '?')} on {phase} "
                f"(edge={edge_level:.0f}W, event_median={event_median:.0f}W) — skipping"
            )
            continue

        # Found a hole! Build wave profile directly from match metadata.
        on_mag = abs(match['on_magnitude'])
        off_mag = abs(match.get('off_magnitude', 0))
        logger.info(
            f"  Hole repair: found hole in {match['on_event_id']} on {phase} "
            f"({on_start} -> {off_end}, edge={edge_level:.0f}W, event_median={event_median:.0f}W, "
            f"on_mag={on_mag:.0f}W, off_mag={off_mag:.0f}W)"
        )

        # Step 1: Restore the rectangle extraction (add constant power back)
        restored = remaining.copy()
        on_mask = (restored.index >= on_start) & (restored.index <= on_end)
        full_event_mask = (restored.index >= on_start) & (restored.index <= off_end)
        restored.loc[full_event_mask] += on_mag

        # Step 2: Estimate baseline from the minutes just before the match
        pre_match_mask = (remaining.index >= on_start - pd.Timedelta(minutes=5)) & \
                         (remaining.index < on_start)
        pre_match = remaining[pre_match_mask]
        baseline_power = float(pre_match.mean()) if len(pre_match) > 0 else 0.0

        # Step 3: Build a linear-decay wave profile from on_mag down to off_mag
        # The wave rises sharply (ON event) and decays linearly over the event duration
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

        # Step 5: Update remaining — replace with wave-extracted version
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


def _load_rectangle_matches(
    output_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load all rectangle matches (not wave) from M1 run directories."""
    all_dfs = []

    for run_number, threshold in enumerate(threshold_schedule):
        # Try both naming conventions
        for pattern in [f"run_{run_number}", f"run_{run_number}_{threshold}w"]:
            run_dir = output_dir / pattern
            if run_dir.is_dir():
                break
        else:
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
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def _remove_repaired_matches(
    output_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
    repaired_ids: List[tuple],
    logger: logging.Logger,
):
    """
    Remove original rectangle matches that were replaced by wave repairs.

    For each (run_number, on_event_id), finds the match file and removes that row.
    """
    # Group by run_number for efficiency
    by_run = {}
    for run_number, on_event_id in repaired_ids:
        by_run.setdefault(run_number, set()).add(on_event_id)

    for run_number, event_ids in by_run.items():
        threshold = threshold_schedule[run_number] if run_number < len(threshold_schedule) else 0

        # Find run directory
        run_dir = None
        for pattern in [f"run_{run_number}", f"run_{run_number}_{threshold}w"]:
            candidate = output_dir / pattern
            if candidate.is_dir():
                run_dir = candidate
                break

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

                original_len = len(df)
                mask = df['on_event_id'].isin(event_ids)
                if mask.any():
                    removed = mask.sum()
                    df = df[~mask]
                    df.to_pickle(pkl_file)
                    logger.info(
                        f"  Hole repair: removed {removed} rectangle match(es) from {pkl_file.name}"
                    )
            except Exception as exc:
                logger.warning(f"Failed to update {pkl_file}: {exc}")


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
