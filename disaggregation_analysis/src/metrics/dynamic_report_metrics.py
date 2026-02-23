"""
Metrics calculation for dynamic threshold experiment reports.

Computes a three-bucket power decomposition per phase:
  - Segregated: power successfully matched to device events
  - Background: estimated constant baseload (5th percentile)
  - Improvable: remaining targetable power not yet segregated

Also computes detection efficiency, per-threshold contributions,
remaining power classification, and device summary.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

from metrics.classification import (
    _load_threshold_schedule,
    _find_run_dir,
    _load_pkl_files,
    DEVICE_TYPES,
)
from metrics.remaining_events import detect_remaining_events

logger = logging.getLogger(__name__)


def calculate_dynamic_report_metrics(
    experiment_dir: Path,
    house_id: str,
) -> Dict[str, Any]:
    """
    Calculate all metrics for the dynamic threshold report.

    Loads summarized pkl files (run_0 baseline + last run remaining),
    dynamic_evaluation_summary CSV, and device_activations JSON.

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID string

    Returns:
        Dict with keys: house_id, phases, totals, per_threshold,
        remaining_classification, devices, data_period
    """
    experiment_dir = Path(experiment_dir)
    timing = {}

    t0 = time.time()
    threshold_schedule = _load_threshold_schedule(experiment_dir)
    timing['config'] = time.time() - t0

    # Load baseline (run_0) and final run summarized data
    t0 = time.time()
    baseline_data, final_data = _load_baseline_and_final(
        experiment_dir, house_id, threshold_schedule
    )
    timing['load_pkl'] = time.time() - t0

    if baseline_data is None or final_data is None:
        logger.error(f"Could not load summarized data for house {house_id}")
        return _empty_metrics(house_id)

    # Per-phase power decomposition
    min_threshold = threshold_schedule[-1] if threshold_schedule else 800

    # Calculate expected minutes from time range (true denominator)
    ts = baseline_data['timestamp']
    expected_minutes = int((ts.max() - ts.min()).total_seconds() / 60) + 1

    t0 = time.time()
    phases = {}
    for phase in ['w1', 'w2', 'w3']:
        phases[phase] = _compute_phase_decomposition(
            baseline_data, final_data, phase,
            min_threshold=min_threshold,
            expected_minutes=expected_minutes,
        )

    # Totals across all phases
    totals = _compute_totals(phases)
    timing['decomposition'] = time.time() - t0

    # Per-threshold contribution from dynamic_evaluation_summary
    t0 = time.time()
    per_threshold = _load_per_threshold_contribution(
        experiment_dir, house_id, threshold_schedule
    )
    timing['per_threshold'] = time.time() - t0

    # Remaining power classification
    t0 = time.time()
    remaining_classification = _classify_remaining_power(
        baseline_data, final_data, phases
    )
    timing['remaining'] = time.time() - t0

    # Remaining event detection (false-negative analysis)
    t0 = time.time()
    remaining_events = detect_remaining_events(
        baseline_data, final_data, phases,
        min_threshold=min_threshold,
    )
    timing['remaining_events'] = time.time() - t0

    # Device summary from device_activations JSON
    t0 = time.time()
    devices = _load_device_summary(experiment_dir, house_id)
    timing['devices'] = time.time() - t0

    # Data period info
    data_period = _get_data_period(baseline_data)

    return {
        'house_id': house_id,
        'threshold_schedule': threshold_schedule,
        'phases': phases,
        'totals': totals,
        'per_threshold': per_threshold,
        'remaining_classification': remaining_classification,
        'remaining_events': remaining_events,
        'devices': devices,
        'data_period': data_period,
        '_baseline': baseline_data,
        '_final': final_data,
        '_timing': timing,
    }


def _load_baseline_and_final(
    experiment_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
) -> tuple:
    """Load run_0 (baseline) and last run (final) summarized data."""
    run_0_dir = _find_run_dir(experiment_dir, 0, threshold_schedule[0])
    last_run = len(threshold_schedule) - 1
    last_run_dir = _find_run_dir(experiment_dir, last_run, threshold_schedule[last_run])

    if run_0_dir is None or last_run_dir is None:
        return None, None

    baseline_summ_dir = run_0_dir / f"house_{house_id}" / "summarized"
    final_summ_dir = last_run_dir / f"house_{house_id}" / "summarized"

    if not baseline_summ_dir.exists() or not final_summ_dir.exists():
        return None, None

    baseline = _load_pkl_files(baseline_summ_dir, f"summarized_{house_id}_*.pkl")
    final = _load_pkl_files(final_summ_dir, f"summarized_{house_id}_*.pkl")

    if baseline.empty or final.empty:
        return None, None

    # Normalize timestamps for alignment
    for df in [baseline, final]:
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Align by timestamp
    baseline_dedup = baseline.drop_duplicates(subset=['timestamp'], keep='first')
    final_dedup = final.drop_duplicates(subset=['timestamp'], keep='first')

    return baseline_dedup, final_dedup


def _compute_phase_decomposition(
    baseline: pd.DataFrame,
    final: pd.DataFrame,
    phase: str,
    min_threshold: int = 800,
    expected_minutes: int = None,
) -> Dict[str, Any]:
    """
    Compute 3-bucket power decomposition for a single phase.

    - total_power: sum of original (clipped >= 0)
    - background: percentile_5(original) * minutes
    - segregated: sum(original - final_remaining), clipped >= 0
    - improvable: total - background - segregated, clipped >= 0
    - efficiency: segregated / (total - background) * 100
    """
    orig_col = f'original_{phase}'
    remain_col = f'remaining_{phase}'

    if orig_col not in baseline.columns:
        return _empty_phase()

    # Merge on timestamp for alignment
    merged = baseline[['timestamp', orig_col]].merge(
        final[['timestamp', remain_col]],
        on='timestamp',
        how='inner',
    )

    # Exclude NaN minutes — only compute on real measurements
    merged_minutes = len(merged)  # rows in pkl (subset of expected)
    all_minutes = expected_minutes if expected_minutes else merged_minutes
    valid_mask = merged[orig_col].notna() & merged[remain_col].notna()
    nan_minutes = all_minutes - int(valid_mask.sum())  # includes dropped + in-pkl NaN
    original = merged.loc[valid_mask, orig_col].clip(lower=0)
    remaining = merged.loc[valid_mask, remain_col].clip(lower=0)

    total_power = original.sum()
    minutes = len(original)  # only valid minutes

    # Background: 5th percentile of original power * valid minutes
    p5 = np.percentile(original, 5) if minutes > 0 else 0
    background_power = p5 * minutes

    # Segregated: original - remaining (per-minute, clipped >= 0)
    segregated_power = (original - remaining).clip(lower=0).sum()

    # Improvable: what's left after subtracting background from remaining
    improvable_power = max(0, remaining.sum() - background_power)

    # Split improvable into above-threshold (detectable) and sub-threshold
    above_base = (remaining - p5).clip(lower=0)
    above_th_mask = above_base > min_threshold
    above_th_power = float(above_base[above_th_mask].sum())
    sub_threshold_power = float(above_base[~above_th_mask].sum())

    # Coverage: fraction of total period with real measurements
    coverage = minutes / all_minutes if all_minutes > 0 else 1.0
    no_data_pct = (1 - coverage) * 100

    # Percentages of measured power (internal, for efficiency calc)
    segregated_pct_measured = (segregated_power / total_power * 100) if total_power > 0 else 0
    background_pct_measured = (background_power / total_power * 100) if total_power > 0 else 0
    improvable_pct_measured = (improvable_power / total_power * 100) if total_power > 0 else 0
    above_th_pct_measured = (above_th_power / total_power * 100) if total_power > 0 else 0
    sub_threshold_pct_measured = (sub_threshold_power / total_power * 100) if total_power > 0 else 0

    # Display percentages: scaled by coverage so all categories sum to 100%
    segregated_pct = segregated_pct_measured * coverage
    background_pct = background_pct_measured * coverage
    improvable_pct = improvable_pct_measured * coverage
    above_th_pct = above_th_pct_measured * coverage
    sub_threshold_pct = sub_threshold_pct_measured * coverage

    # Detection efficiency: segregated / (segregated + above_th) — only detectable power
    detectable = segregated_power + above_th_power
    efficiency = (segregated_power / detectable * 100) if detectable > 0 else 100

    # kWh conversions (watts * minutes / 60 / 1000)
    to_kwh = lambda w: round(w / 60 / 1000, 2)

    # Negative remaining minutes (sanity check)
    neg_minutes = int((remaining < 0).sum())

    return {
        'total_power': round(total_power, 1),
        'total_kwh': to_kwh(total_power),
        'segregated_power': round(segregated_power, 1),
        'segregated_kwh': to_kwh(segregated_power),
        'segregated_pct': round(segregated_pct, 1),
        'background_power': round(background_power, 1),
        'background_kwh': to_kwh(background_power),
        'background_pct': round(background_pct, 1),
        'background_per_minute': round(p5, 1),
        'improvable_power': round(improvable_power, 1),
        'improvable_kwh': to_kwh(improvable_power),
        'improvable_pct': round(improvable_pct, 1),
        'above_th_power': round(above_th_power, 1),
        'above_th_kwh': to_kwh(above_th_power),
        'above_th_pct': round(above_th_pct, 1),
        'sub_threshold_power': round(sub_threshold_power, 1),
        'sub_threshold_kwh': to_kwh(sub_threshold_power),
        'sub_threshold_pct': round(sub_threshold_pct, 1),
        'no_data_pct': round(no_data_pct, 1),
        'efficiency': round(efficiency, 1),
        'minutes': minutes,
        'nan_minutes': nan_minutes,
        'all_minutes': all_minutes,
        'coverage': round(coverage, 4),
        'negative_minutes': neg_minutes,
    }


def _compute_totals(phases: Dict[str, Dict]) -> Dict[str, Any]:
    """Aggregate metrics across all 3 phases."""
    total_power = sum(p.get('total_power', 0) for p in phases.values())
    segregated_power = sum(p.get('segregated_power', 0) for p in phases.values())
    background_power = sum(p.get('background_power', 0) for p in phases.values())
    improvable_power = sum(p.get('improvable_power', 0) for p in phases.values())
    above_th_power = sum(p.get('above_th_power', 0) for p in phases.values())
    sub_threshold_power = sum(p.get('sub_threshold_power', 0) for p in phases.values())

    total_minutes = sum(p.get('minutes', 0) for p in phases.values())
    total_all_minutes = sum(p.get('all_minutes', 0) for p in phases.values())
    coverage = total_minutes / total_all_minutes if total_all_minutes > 0 else 1.0
    no_data_pct = (1 - coverage) * 100

    # Internal percentages (of measured power)
    segregated_pct_m = (segregated_power / total_power * 100) if total_power > 0 else 0
    background_pct_m = (background_power / total_power * 100) if total_power > 0 else 0
    improvable_pct_m = (improvable_power / total_power * 100) if total_power > 0 else 0
    above_th_pct_m = (above_th_power / total_power * 100) if total_power > 0 else 0
    sub_threshold_pct_m = (sub_threshold_power / total_power * 100) if total_power > 0 else 0

    # Display percentages: scaled by coverage
    segregated_pct = segregated_pct_m * coverage
    background_pct = background_pct_m * coverage
    improvable_pct = improvable_pct_m * coverage
    above_th_pct = above_th_pct_m * coverage
    sub_threshold_pct = sub_threshold_pct_m * coverage

    # Efficiency: segregated / (segregated + above_threshold) — only detectable power
    detectable = segregated_power + above_th_power
    efficiency = (segregated_power / detectable * 100) if detectable > 0 else 100

    to_kwh = lambda w: round(w / 60 / 1000, 2)

    return {
        'total_power': round(total_power, 1),
        'total_kwh': to_kwh(total_power),
        'segregated_power': round(segregated_power, 1),
        'segregated_kwh': to_kwh(segregated_power),
        'segregated_pct': round(segregated_pct, 1),
        'background_power': round(background_power, 1),
        'background_kwh': to_kwh(background_power),
        'background_pct': round(background_pct, 1),
        'improvable_power': round(improvable_power, 1),
        'improvable_kwh': to_kwh(improvable_power),
        'improvable_pct': round(improvable_pct, 1),
        'above_th_power': round(above_th_power, 1),
        'above_th_kwh': to_kwh(above_th_power),
        'above_th_pct': round(above_th_pct, 1),
        'sub_threshold_power': round(sub_threshold_power, 1),
        'sub_threshold_kwh': to_kwh(sub_threshold_power),
        'sub_threshold_pct': round(sub_threshold_pct, 1),
        'no_data_pct': round(no_data_pct, 1),
        'efficiency': round(efficiency, 1),
        'coverage': round(coverage, 4),
    }


def _load_per_threshold_contribution(
    experiment_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
) -> List[Dict[str, Any]]:
    """
    Load per-threshold contribution from dynamic_evaluation_summary CSV.

    Falls back to computing from summarized pkls if CSV is not available.
    """
    csv_path = experiment_dir / "evaluation_summaries" / f"dynamic_evaluation_summary_{house_id}.csv"
    # Fallback: check legacy location (experiment root)
    if not csv_path.exists():
        csv_path = experiment_dir / f"dynamic_evaluation_summary_{house_id}.csv"

    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            contributions = []
            for th in threshold_schedule:
                th_rows = df[df['threshold'] == th]
                if th_rows.empty:
                    contributions.append({
                        'threshold': th,
                        'segregated_pct': 0,
                        'segregated_power': 0,
                    })
                    continue

                iter_explained = th_rows['iteration_explained'].sum()
                iter_pct = th_rows['iteration_explained_pct'].mean()
                cum_pct = th_rows['cumulative_explained_pct'].mean()

                contributions.append({
                    'threshold': th,
                    'segregated_power': round(iter_explained, 1),
                    'segregated_pct': round(iter_pct, 1),
                    'cumulative_pct': round(cum_pct, 1),
                })
            return contributions
        except Exception as e:
            logger.warning(f"Failed to read evaluation summary CSV: {e}")

    # Fallback: compute from summarized pkls
    return _compute_per_threshold_from_pkls(
        experiment_dir, house_id, threshold_schedule
    )


def _compute_per_threshold_from_pkls(
    experiment_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
) -> List[Dict[str, Any]]:
    """Compute per-threshold contribution from summarized pkls."""
    contributions = []
    prev_remaining_total = None

    for run_number, threshold in enumerate(threshold_schedule):
        run_dir = _find_run_dir(experiment_dir, run_number, threshold)
        if run_dir is None:
            contributions.append({
                'threshold': threshold,
                'segregated_pct': 0,
                'segregated_power': 0,
            })
            continue

        summ_dir = run_dir / f"house_{house_id}" / "summarized"
        if not summ_dir.exists():
            contributions.append({
                'threshold': threshold,
                'segregated_pct': 0,
                'segregated_power': 0,
            })
            continue

        data = _load_pkl_files(summ_dir, f"summarized_{house_id}_*.pkl")
        if data.empty:
            contributions.append({
                'threshold': threshold,
                'segregated_pct': 0,
                'segregated_power': 0,
            })
            continue

        # Sum remaining across phases
        remaining_total = 0
        original_total = 0
        for phase in ['w1', 'w2', 'w3']:
            remain_col = f'remaining_{phase}'
            orig_col = f'original_{phase}'
            if remain_col in data.columns:
                remaining_total += data[remain_col].clip(lower=0).sum()
            if orig_col in data.columns:
                original_total += data[orig_col].clip(lower=0).sum()

        if run_number == 0:
            iter_explained = original_total - remaining_total
            base_total = original_total
        else:
            iter_explained = max(0, (prev_remaining_total or 0) - remaining_total)
            # Use run_0 original as base for percentage
            run0_dir = _find_run_dir(experiment_dir, 0, threshold_schedule[0])
            if run0_dir:
                run0_summ = run0_dir / f"house_{house_id}" / "summarized"
                run0_data = _load_pkl_files(run0_summ, f"summarized_{house_id}_*.pkl")
                base_total = sum(
                    run0_data[f'original_{p}'].clip(lower=0).sum()
                    for p in ['w1', 'w2', 'w3']
                    if f'original_{p}' in run0_data.columns
                ) if not run0_data.empty else original_total
            else:
                base_total = original_total

        iter_pct = (iter_explained / base_total * 100) if base_total > 0 else 0

        contributions.append({
            'threshold': threshold,
            'segregated_power': round(iter_explained, 1),
            'segregated_pct': round(iter_pct, 1),
        })

        prev_remaining_total = remaining_total

    return contributions


def _classify_remaining_power(
    baseline: pd.DataFrame,
    final: pd.DataFrame,
    phases: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Classify remaining power into categories:
    - noise: < 200W above baseload
    - small_events: 200-800W above baseload
    - large_unmatched: > 800W above baseload
    """
    noise_minutes = 0
    small_event_minutes = 0
    large_unmatched_minutes = 0
    total_minutes = 0

    for phase in ['w1', 'w2', 'w3']:
        remain_col = f'remaining_{phase}'
        if remain_col not in final.columns:
            continue

        remaining = final[remain_col].clip(lower=0)
        baseload = phases[phase].get('background_per_minute', 0)
        above_base = remaining - baseload

        total_minutes += len(remaining)
        noise_minutes += (above_base <= 200).sum()
        small_event_minutes += ((above_base > 200) & (above_base <= 800)).sum()
        large_unmatched_minutes += (above_base > 800).sum()

    return {
        'noise_minutes': int(noise_minutes),
        'small_event_minutes': int(small_event_minutes),
        'large_unmatched_minutes': int(large_unmatched_minutes),
        'total_minutes': int(total_minutes),
        'noise_pct': round(noise_minutes / total_minutes * 100, 1) if total_minutes > 0 else 0,
        'small_event_pct': round(small_event_minutes / total_minutes * 100, 1) if total_minutes > 0 else 0,
        'large_unmatched_pct': round(large_unmatched_minutes / total_minutes * 100, 1) if total_minutes > 0 else 0,
    }


def _load_device_summary(
    experiment_dir: Path,
    house_id: str,
) -> Dict[str, Any]:
    """Load device summary from device_activations JSON."""
    json_path = experiment_dir / "device_activations" / f"device_activations_{house_id}.json"
    # Fallback: check legacy location (experiment root)
    if not json_path.exists():
        json_path = experiment_dir / f"device_activations_{house_id}.json"

    if not json_path.exists():
        return {'available': False, 'types': {}}

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load device activations JSON: {e}")
        return {'available': False, 'types': {}}

    activations = data.get('activations', [])
    matched = [a for a in activations if a.get('match_type') == 'matched']

    if not matched:
        return {
            'available': True,
            'total_activations': len(activations),
            'total_matched': 0,
            'total_unmatched': len(activations),
            'types': {},
        }

    # Group by device_type
    types = {}
    total_energy = 0

    for dtype in DEVICE_TYPES:
        subset = [a for a in matched if a.get('device_type') == dtype]
        if not subset:
            continue

        count = len(subset)
        magnitudes = [abs(a.get('on_magnitude', 0) or 0) for a in subset]
        durations = [a.get('duration', 0) or 0 for a in subset]
        avg_mag = round(np.mean(magnitudes)) if magnitudes else 0
        avg_dur = round(np.mean(durations), 1) if durations else 0

        # Energy proxy: sum(magnitude * duration)
        energy = sum(m * d for m, d in zip(magnitudes, durations))
        total_energy += energy

        types[dtype] = {
            'count': count,
            'avg_magnitude': avg_mag,
            'avg_duration': avg_dur,
            'energy': round(energy, 1),
        }

    # Compute percentages
    if total_energy > 0:
        for dtype in types:
            types[dtype]['energy_pct'] = round(
                types[dtype]['energy'] / total_energy * 100, 1
            )
    else:
        for dtype in types:
            types[dtype]['energy_pct'] = 0

    unmatched_on = len([a for a in activations if a.get('match_type') == 'unmatched_on'])
    unmatched_off = len([a for a in activations if a.get('match_type') == 'unmatched_off'])

    return {
        'available': True,
        'total_activations': len(activations),
        'total_matched': len(matched),
        'total_unmatched': unmatched_on + unmatched_off,
        'unmatched_on': unmatched_on,
        'unmatched_off': unmatched_off,
        'types': types,
    }


def _get_data_period(baseline: pd.DataFrame) -> Dict[str, str]:
    """Extract data period from baseline timestamps."""
    if 'timestamp' not in baseline.columns or baseline.empty:
        return {'start': 'N/A', 'end': 'N/A', 'days': 0}

    ts = pd.to_datetime(baseline['timestamp'])
    start = ts.min()
    end = ts.max()
    days = (end - start).days + 1

    return {
        'start': start.strftime('%Y-%m-%d'),
        'end': end.strftime('%Y-%m-%d'),
        'days': days,
    }


def _empty_phase() -> Dict[str, Any]:
    """Return empty phase metrics."""
    return {
        'total_power': 0, 'total_kwh': 0,
        'segregated_power': 0, 'segregated_kwh': 0, 'segregated_pct': 0,
        'background_power': 0, 'background_kwh': 0, 'background_pct': 0,
        'background_per_minute': 0,
        'improvable_power': 0, 'improvable_kwh': 0, 'improvable_pct': 0,
        'above_th_power': 0, 'above_th_kwh': 0, 'above_th_pct': 0,
        'sub_threshold_power': 0, 'sub_threshold_kwh': 0, 'sub_threshold_pct': 0,
        'no_data_pct': 0,
        'efficiency': 0, 'minutes': 0, 'nan_minutes': 0, 'all_minutes': 0,
        'coverage': 1.0, 'negative_minutes': 0,
    }


def _empty_metrics(house_id: str) -> Dict[str, Any]:
    """Return empty metrics structure."""
    empty_phase = _empty_phase()
    return {
        'house_id': house_id,
        'threshold_schedule': [],
        'phases': {p: empty_phase for p in ['w1', 'w2', 'w3']},
        'totals': {
            'total_power': 0, 'total_kwh': 0,
            'segregated_power': 0, 'segregated_kwh': 0, 'segregated_pct': 0,
            'background_power': 0, 'background_kwh': 0, 'background_pct': 0,
            'improvable_power': 0, 'improvable_kwh': 0, 'improvable_pct': 0,
            'above_th_power': 0, 'above_th_kwh': 0, 'above_th_pct': 0,
            'sub_threshold_power': 0, 'sub_threshold_kwh': 0, 'sub_threshold_pct': 0,
            'no_data_pct': 0, 'efficiency': 0, 'coverage': 1.0,
        },
        'per_threshold': [],
        'remaining_classification': {
            'noise_minutes': 0, 'small_event_minutes': 0,
            'large_unmatched_minutes': 0, 'total_minutes': 0,
            'noise_pct': 0, 'small_event_pct': 0, 'large_unmatched_pct': 0,
        },
        'devices': {'available': False, 'types': {}},
        'data_period': {'start': 'N/A', 'end': 'N/A', 'days': 0},
    }
