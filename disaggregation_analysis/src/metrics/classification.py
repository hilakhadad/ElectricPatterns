"""
Classification metrics for dynamic threshold experiments.

Computes metrics based on device classification (boiler/central_ac/regular_ac/unclassified)
rather than traditional threshold-based evaluation.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

logger = logging.getLogger(__name__)

DEVICE_TYPES = ['boiler', 'central_ac', 'regular_ac', 'recurring_pattern', 'unclassified']


def calculate_classification_metrics(
    experiment_dir: Path,
    house_id: str,
    preloaded: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate device classification metrics across all iterations.

    Loads classification pkl files from all run directories and computes:
    - classified_rate: fraction of matches assigned to a known device type
    - device_power_pct: fraction of total power segregated by classified devices
    - total_segregated_power_pct: threshold-independent total segregated power
    - per-device and per-iteration breakdowns

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID
        preloaded: Optional preloaded data dict (for efficiency)

    Returns:
        Dict with all classification metrics
    """
    experiment_dir = Path(experiment_dir)

    # Load threshold_schedule from metadata
    threshold_schedule = _load_threshold_schedule(experiment_dir)

    # Collect classification data from all iterations
    all_classified = []
    per_iteration = []

    for run_number, threshold in enumerate(threshold_schedule):
        run_dir = _find_run_dir(experiment_dir, run_number, threshold)
        if run_dir is None:
            continue

        classification_dir = run_dir / f"house_{house_id}" / "classification"
        if not classification_dir.exists():
            continue

        classified = _load_pkl_files(classification_dir, f"classification_{house_id}_*.pkl")
        if classified.empty:
            continue

        # Per-iteration metrics
        iter_metrics = _compute_device_metrics(classified)
        iter_metrics['run_number'] = run_number
        iter_metrics['threshold'] = threshold
        per_iteration.append(iter_metrics)

        all_classified.append(classified)

    if not all_classified:
        return _empty_metrics(house_id)

    # Combined metrics across all iterations
    combined = pd.concat(all_classified, ignore_index=True)
    overall = _compute_device_metrics(combined)

    # Total segregated power (threshold-independent)
    total_segregated_pct = _compute_total_segregated_power(experiment_dir, house_id, threshold_schedule)

    # Per-phase breakdown
    per_phase = {}
    for phase in ['w1', 'w2', 'w3']:
        phase_data = combined[combined['phase'] == phase] if 'phase' in combined.columns else pd.DataFrame()
        if not phase_data.empty:
            per_phase[phase] = _compute_device_metrics(phase_data)

    return {
        'house_id': house_id,
        'total_matches': len(combined),
        'classified_count': overall['classified_count'],
        'unclassified_count': overall['unclassified_count'],
        'classified_rate': overall['classified_rate'],
        'unclassified_rate': overall['unclassified_rate'],
        'device_power_pct': overall.get('device_power_pct', 0),
        'boiler_power_pct': overall.get('boiler_power_pct', 0),
        'ac_power_pct': overall.get('ac_power_pct', 0),
        'total_segregated_power_pct': total_segregated_pct,
        'device_breakdown': overall['device_breakdown'],
        'per_iteration': per_iteration,
        'per_phase': per_phase,
    }


def _compute_device_metrics(classified: pd.DataFrame) -> Dict[str, Any]:
    """Compute metrics for a set of classified matches."""
    total = len(classified)
    if total == 0:
        return {
            'classified_count': 0, 'unclassified_count': 0,
            'classified_rate': 0, 'unclassified_rate': 0,
            'device_breakdown': {dt: {'count': 0, 'power_pct': 0, 'avg_duration': 0, 'avg_magnitude': 0}
                                 for dt in DEVICE_TYPES},
        }

    breakdown = {}
    for dtype in DEVICE_TYPES:
        subset = classified[classified['device_type'] == dtype]
        count = len(subset)

        avg_mag = round(subset['on_magnitude'].abs().mean()) if not subset.empty else 0
        avg_dur = round(subset['duration'].mean(), 1) if not subset.empty else 0

        # Power contribution: sum(magnitude * duration) as proxy
        if not subset.empty and 'duration' in subset.columns:
            power_contribution = (subset['on_magnitude'].abs() * subset['duration']).sum()
        else:
            power_contribution = 0

        breakdown[dtype] = {
            'count': count,
            'avg_magnitude': avg_mag,
            'avg_duration': avg_dur,
            'power_contribution': round(power_contribution, 1),
        }

    # Calculate power percentages
    total_power = sum(b['power_contribution'] for b in breakdown.values())
    if total_power > 0:
        for dtype in DEVICE_TYPES:
            breakdown[dtype]['power_pct'] = round(
                breakdown[dtype]['power_contribution'] / total_power, 3
            )
    else:
        for dtype in DEVICE_TYPES:
            breakdown[dtype]['power_pct'] = 0

    unclassified_count = breakdown['unclassified']['count']
    classified_count = total - unclassified_count

    boiler_pct = breakdown['boiler']['power_pct']
    ac_pct = breakdown['central_ac']['power_pct'] + breakdown['regular_ac']['power_pct']

    return {
        'classified_count': classified_count,
        'unclassified_count': unclassified_count,
        'classified_rate': round(classified_count / total, 3) if total > 0 else 0,
        'unclassified_rate': round(unclassified_count / total, 3) if total > 0 else 0,
        'device_power_pct': round(1 - breakdown['unclassified']['power_pct'], 3),
        'boiler_power_pct': boiler_pct,
        'ac_power_pct': ac_pct,
        'device_breakdown': breakdown,
    }


def _compute_total_segregated_power(
    experiment_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
) -> float:
    """
    Compute total segregated power as a threshold-independent metric.

    Looks at the LAST iteration's summarized output and compares
    remaining power to original power from run_0.

    Returns:
        Fraction of total power segregated (0.0 - 1.0)
    """
    # Find the last run's summarized data
    last_run = len(threshold_schedule) - 1
    last_threshold = threshold_schedule[last_run]
    last_run_dir = _find_run_dir(experiment_dir, last_run, last_threshold)

    # Find run_0's summarized (has original power)
    first_run_dir = _find_run_dir(experiment_dir, 0, threshold_schedule[0])

    if last_run_dir is None or first_run_dir is None:
        return 0.0

    # Load summarized data
    last_summarized_dir = last_run_dir / f"house_{house_id}" / "summarized"
    first_summarized_dir = first_run_dir / f"house_{house_id}" / "summarized"

    if not last_summarized_dir.exists() or not first_summarized_dir.exists():
        return 0.0

    last_data = _load_pkl_files(last_summarized_dir, f"summarized_{house_id}_*.pkl")
    first_data = _load_pkl_files(first_summarized_dir, f"summarized_{house_id}_*.pkl")

    if last_data.empty or first_data.empty:
        return 0.0

    total_original = 0
    total_remaining = 0

    for phase in ['w1', 'w2', 'w3']:
        orig_col = f"original_{phase}"
        remain_col = f"remaining_{phase}"

        if orig_col in first_data.columns:
            total_original += first_data[orig_col].clip(lower=0).sum()

        if remain_col in last_data.columns:
            total_remaining += last_data[remain_col].clip(lower=0).sum()

    if total_original == 0:
        return 0.0

    segregated = total_original - total_remaining
    return round(max(0, segregated / total_original), 4)


def _load_threshold_schedule(experiment_dir: Path) -> List[int]:
    """Load threshold schedule from experiment metadata."""
    metadata_path = experiment_dir / "experiment_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        schedule = metadata.get('experiment', {}).get('threshold_schedule')
        if schedule:
            return schedule

    # Fallback: detect from directory names
    schedule = []
    for d in sorted(experiment_dir.glob("run_*_th*")):
        if d.is_dir():
            # Extract threshold from run_0_th2000
            parts = d.name.split('_th')
            if len(parts) == 2:
                try:
                    schedule.append(int(parts[1]))
                except ValueError:
                    pass

    return schedule or [2000, 1500, 1100, 800]


def _find_run_dir(experiment_dir: Path, run_number: int, threshold: int) -> Optional[Path]:
    """Find run directory supporting both run_N and run_N_thXXXX naming."""
    # Try dynamic naming first
    dynamic = experiment_dir / f"run_{run_number}_th{threshold}"
    if dynamic.exists():
        return dynamic

    # Try plain naming
    plain = experiment_dir / f"run_{run_number}"
    if plain.exists():
        return plain

    # Try glob
    for d in experiment_dir.glob(f"run_{run_number}_th*"):
        if d.is_dir():
            return d

    return None


def _load_pkl_files(directory: Path, pattern: str) -> pd.DataFrame:
    """Load all pkl files matching pattern into one DataFrame."""
    files = sorted(directory.glob(pattern))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_pickle(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    return pd.concat(dfs, ignore_index=False) if dfs else pd.DataFrame()


def _empty_metrics(house_id: str) -> Dict[str, Any]:
    """Return empty metrics structure."""
    return {
        'house_id': house_id,
        'total_matches': 0,
        'classified_count': 0,
        'unclassified_count': 0,
        'classified_rate': 0,
        'unclassified_rate': 0,
        'device_power_pct': 0,
        'boiler_power_pct': 0,
        'ac_power_pct': 0,
        'total_segregated_power_pct': 0,
        'device_breakdown': {dt: {'count': 0, 'power_pct': 0, 'avg_duration': 0, 'avg_magnitude': 0}
                             for dt in DEVICE_TYPES},
        'per_iteration': [],
        'per_phase': {},
    }
