"""
Metrics calculation for dynamic threshold experiment reports.

Computes a three-bucket power decomposition per phase:
  - Explained: power successfully matched to device events
  - Background: estimated constant baseload (5th percentile)
  - Improvable: remaining targetable power not yet explained

Also computes detection efficiency, per-threshold contributions,
remaining power classification, and device summary.
"""
import json
import logging
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
    threshold_schedule = _load_threshold_schedule(experiment_dir)

    # Load baseline (run_0) and final run summarized data
    baseline_data, final_data = _load_baseline_and_final(
        experiment_dir, house_id, threshold_schedule
    )

    if baseline_data is None or final_data is None:
        logger.error(f"Could not load summarized data for house {house_id}")
        return _empty_metrics(house_id)

    # Per-phase power decomposition
    phases = {}
    for phase in ['w1', 'w2', 'w3']:
        phases[phase] = _compute_phase_decomposition(
            baseline_data, final_data, phase
        )

    # Totals across all phases
    totals = _compute_totals(phases)

    # Per-threshold contribution from dynamic_evaluation_summary
    per_threshold = _load_per_threshold_contribution(
        experiment_dir, house_id, threshold_schedule
    )

    # Remaining power classification
    remaining_classification = _classify_remaining_power(
        baseline_data, final_data, phases
    )

    # Device summary from device_activations JSON
    devices = _load_device_summary(experiment_dir, house_id)

    # Data period info
    data_period = _get_data_period(baseline_data)

    return {
        'house_id': house_id,
        'threshold_schedule': threshold_schedule,
        'phases': phases,
        'totals': totals,
        'per_threshold': per_threshold,
        'remaining_classification': remaining_classification,
        'devices': devices,
        'data_period': data_period,
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
) -> Dict[str, Any]:
    """
    Compute 3-bucket power decomposition for a single phase.

    - total_power: sum of original (clipped >= 0)
    - background: percentile_5(original) * minutes
    - explained: sum(original - final_remaining), clipped >= 0
    - improvable: total - background - explained, clipped >= 0
    - efficiency: explained / (total - background) * 100
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

    # NaN = no consumption reading → treat as 0W (background)
    nan_minutes = int((merged[orig_col].isna() | merged[remain_col].isna()).sum())
    original = merged[orig_col].fillna(0).clip(lower=0)
    remaining = merged[remain_col].fillna(0).clip(lower=0)

    total_power = original.sum()
    minutes = len(original)

    # Background: 5th percentile of original power * minutes
    p5 = np.percentile(original, 5) if minutes > 0 else 0
    background_power = p5 * minutes

    # Explained: original - remaining (per-minute, clipped >= 0)
    explained_power = (original - remaining).clip(lower=0).sum()

    # Improvable: what's left after subtracting background from remaining
    improvable_power = max(0, remaining.sum() - background_power)

    # Percentages
    total_pct = 100.0
    explained_pct = (explained_power / total_power * 100) if total_power > 0 else 0
    background_pct = (background_power / total_power * 100) if total_power > 0 else 0
    improvable_pct = (improvable_power / total_power * 100) if total_power > 0 else 0

    # Detection efficiency: explained / targetable
    targetable = total_power - background_power
    efficiency = (explained_power / targetable * 100) if targetable > 0 else 0

    # kWh conversions (watts * minutes / 60 / 1000)
    to_kwh = lambda w: round(w / 60 / 1000, 2)

    # Negative remaining minutes (sanity check)
    neg_minutes = int((remaining < 0).sum())

    return {
        'total_power': round(total_power, 1),
        'total_kwh': to_kwh(total_power),
        'explained_power': round(explained_power, 1),
        'explained_kwh': to_kwh(explained_power),
        'explained_pct': round(explained_pct, 1),
        'background_power': round(background_power, 1),
        'background_kwh': to_kwh(background_power),
        'background_pct': round(background_pct, 1),
        'background_per_minute': round(p5, 1),
        'improvable_power': round(improvable_power, 1),
        'improvable_kwh': to_kwh(improvable_power),
        'improvable_pct': round(improvable_pct, 1),
        'efficiency': round(efficiency, 1),
        'minutes': minutes,
        'nan_minutes': nan_minutes,
        'negative_minutes': neg_minutes,
    }


def _compute_totals(phases: Dict[str, Dict]) -> Dict[str, Any]:
    """Aggregate metrics across all 3 phases."""
    total_power = sum(p.get('total_power', 0) for p in phases.values())
    explained_power = sum(p.get('explained_power', 0) for p in phases.values())
    background_power = sum(p.get('background_power', 0) for p in phases.values())
    improvable_power = sum(p.get('improvable_power', 0) for p in phases.values())

    explained_pct = (explained_power / total_power * 100) if total_power > 0 else 0
    background_pct = (background_power / total_power * 100) if total_power > 0 else 0
    improvable_pct = (improvable_power / total_power * 100) if total_power > 0 else 0

    targetable = total_power - background_power
    efficiency = (explained_power / targetable * 100) if targetable > 0 else 0

    to_kwh = lambda w: round(w / 60 / 1000, 2)

    return {
        'total_power': round(total_power, 1),
        'total_kwh': to_kwh(total_power),
        'explained_power': round(explained_power, 1),
        'explained_kwh': to_kwh(explained_power),
        'explained_pct': round(explained_pct, 1),
        'background_power': round(background_power, 1),
        'background_kwh': to_kwh(background_power),
        'background_pct': round(background_pct, 1),
        'improvable_power': round(improvable_power, 1),
        'improvable_kwh': to_kwh(improvable_power),
        'improvable_pct': round(improvable_pct, 1),
        'efficiency': round(efficiency, 1),
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
                        'explained_pct': 0,
                        'explained_power': 0,
                    })
                    continue

                iter_explained = th_rows['iteration_explained'].sum()
                iter_pct = th_rows['iteration_explained_pct'].mean()
                cum_pct = th_rows['cumulative_explained_pct'].mean()

                contributions.append({
                    'threshold': th,
                    'explained_power': round(iter_explained, 1),
                    'explained_pct': round(iter_pct, 1),
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
                'explained_pct': 0,
                'explained_power': 0,
            })
            continue

        summ_dir = run_dir / f"house_{house_id}" / "summarized"
        if not summ_dir.exists():
            contributions.append({
                'threshold': threshold,
                'explained_pct': 0,
                'explained_power': 0,
            })
            continue

        data = _load_pkl_files(summ_dir, f"summarized_{house_id}_*.pkl")
        if data.empty:
            contributions.append({
                'threshold': threshold,
                'explained_pct': 0,
                'explained_power': 0,
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
            'explained_power': round(iter_explained, 1),
            'explained_pct': round(iter_pct, 1),
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
        'explained_power': 0, 'explained_kwh': 0, 'explained_pct': 0,
        'background_power': 0, 'background_kwh': 0, 'background_pct': 0,
        'background_per_minute': 0,
        'improvable_power': 0, 'improvable_kwh': 0, 'improvable_pct': 0,
        'efficiency': 0, 'minutes': 0, 'negative_minutes': 0,
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
            'explained_power': 0, 'explained_kwh': 0, 'explained_pct': 0,
            'background_power': 0, 'background_kwh': 0, 'background_pct': 0,
            'improvable_power': 0, 'improvable_kwh': 0, 'improvable_pct': 0,
            'efficiency': 0,
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
