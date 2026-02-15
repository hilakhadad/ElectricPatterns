"""
Build unified device activations JSON from pipeline output.

Combines:
- Classification files (matched ON/OFF pairs with device_type)
- Unmatched ON/OFF events
- Per-device power profiles captured during segmentation

Output: device_activations_{house_id}.json
"""
import json
import math
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def build_device_activations_json(
    experiment_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
    device_profiles: Optional[Dict[int, Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Build unified device activations JSON for a house.

    Args:
        experiment_dir: Root experiment output directory
        house_id: House ID
        threshold_schedule: List of thresholds [2000, 1500, 1100, 800]
        device_profiles: Dict mapping run_number -> {on_event_id -> {timestamps, values}}
                         If None, activations will not have per-minute values.
        output_path: Where to save the JSON. Default: experiment_dir/device_activations_{house_id}.json

    Returns:
        Path to the saved JSON file
    """
    if output_path is None:
        output_path = experiment_dir / f"device_activations_{house_id}.json"

    activations = []
    matched_count = 0
    unmatched_on_count = 0
    unmatched_off_count = 0

    for run_number, threshold in enumerate(threshold_schedule):
        logger.info(f"Building activations for run {run_number} (threshold={threshold})")

        # Get profiles for this run
        run_profiles = device_profiles.get(run_number, {}) if device_profiles else {}

        # Load classification files (matched events with device_type)
        try:
            classification_df = _load_classification_files(experiment_dir, house_id, run_number, threshold)
            if not classification_df.empty:
                for _, row in classification_df.iterrows():
                    activation = _build_matched_activation(row, run_number, threshold, run_profiles)
                    activations.append(activation)
                    matched_count += 1
        except FileNotFoundError:
            logger.warning(f"No classification files for run {run_number}")

        # Load unmatched ON events
        try:
            unmatched_on_df = _load_unmatched_files(experiment_dir, house_id, run_number, threshold, 'on')
            if not unmatched_on_df.empty:
                for _, row in unmatched_on_df.iterrows():
                    activation = _build_unmatched_activation(row, 'on', run_number, threshold)
                    activations.append(activation)
                    unmatched_on_count += 1
        except FileNotFoundError:
            logger.warning(f"No unmatched_on files for run {run_number}")

        # Load unmatched OFF events
        try:
            unmatched_off_df = _load_unmatched_files(experiment_dir, house_id, run_number, threshold, 'off')
            if not unmatched_off_df.empty:
                for _, row in unmatched_off_df.iterrows():
                    activation = _build_unmatched_activation(row, 'off', run_number, threshold)
                    activations.append(activation)
                    unmatched_off_count += 1
        except FileNotFoundError:
            logger.warning(f"No unmatched_off files for run {run_number}")

    # Sort by start time (on_start for matched/unmatched_on, off_start for unmatched_off)
    activations.sort(key=lambda a: a.get('on_start') or a.get('off_start') or '9999-99-99')

    # Build final JSON structure
    output = {
        'house_id': house_id,
        'generated_at': datetime.now().isoformat(),
        'threshold_schedule': threshold_schedule,
        'total_activations': len(activations),
        'total_matched': matched_count,
        'total_unmatched_on': unmatched_on_count,
        'total_unmatched_off': unmatched_off_count,
        'activations': activations
    }

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_serializer)

    logger.info(f"Saved {len(activations)} activations to {output_path}")
    return output_path


def _build_matched_activation(
    row: pd.Series,
    run_number: int,
    threshold: int,
    profiles: Dict[str, Any],
) -> dict:
    """
    Build a single matched activation entry.

    Fields:
        phase, on_start, on_end, off_start, off_end,
        on_duration, off_duration, on_magnitude, off_magnitude,
        duration, tag, iteration, threshold, device_type,
        values (list of floats or None)
    """
    on_event_id = row.get('on_event_id')
    profile = profiles.get(on_event_id, {}) if profiles else {}

    # Calculate durations
    on_duration = None
    off_duration = None
    if pd.notna(row.get('on_start')) and pd.notna(row.get('on_end')):
        on_duration = (pd.Timestamp(row['on_end']) - pd.Timestamp(row['on_start'])).total_seconds() / 60
    if pd.notna(row.get('off_start')) and pd.notna(row.get('off_end')):
        off_duration = (pd.Timestamp(row['off_end']) - pd.Timestamp(row['off_start'])).total_seconds() / 60

    return {
        'phase': row.get('phase'),
        'on_start': _serialize_timestamp(row.get('on_start')),
        'on_end': _serialize_timestamp(row.get('on_end')),
        'off_start': _serialize_timestamp(row.get('off_start')),
        'off_end': _serialize_timestamp(row.get('off_end')),
        'on_duration': _serialize_value(on_duration),
        'off_duration': _serialize_value(off_duration),
        'on_magnitude': _serialize_value(row.get('on_magnitude')),
        'off_magnitude': _serialize_value(row.get('off_magnitude')),
        'duration': _serialize_value(row.get('duration')),
        'tag': row.get('tag'),
        'iteration': run_number,
        'threshold': threshold,
        'device_type': row.get('device_type'),
        'values': [_serialize_value(v) for v in profile.get('values', [])] if profile.get('values') else None,
        'match_type': 'matched'
    }


def _build_unmatched_activation(
    row: pd.Series,
    event_type: str,  # 'on' or 'off'
    run_number: int,
    threshold: int,
) -> dict:
    """
    Build an unmatched event entry.

    For unmatched ON: on_* fields filled, off_* fields null
    For unmatched OFF: off_* fields filled, on_* fields null
    """
    if event_type == 'on':
        # Calculate on_duration
        on_duration = None
        if pd.notna(row.get('start')) and pd.notna(row.get('end')):
            on_duration = (pd.Timestamp(row['end']) - pd.Timestamp(row['start'])).total_seconds() / 60

        return {
            'phase': row.get('phase'),
            'on_start': _serialize_timestamp(row.get('start')),
            'on_end': _serialize_timestamp(row.get('end')),
            'off_start': None,
            'off_end': None,
            'on_duration': _serialize_value(on_duration),
            'off_duration': None,
            'on_magnitude': _serialize_value(row.get('magnitude')),
            'off_magnitude': None,
            'duration': None,
            'tag': None,
            'iteration': run_number,
            'threshold': threshold,
            'device_type': None,
            'values': None,
            'match_type': 'unmatched_on'
        }
    else:  # event_type == 'off'
        # Calculate off_duration
        off_duration = None
        if pd.notna(row.get('start')) and pd.notna(row.get('end')):
            off_duration = (pd.Timestamp(row['end']) - pd.Timestamp(row['start'])).total_seconds() / 60

        return {
            'phase': row.get('phase'),
            'on_start': None,
            'on_end': None,
            'off_start': _serialize_timestamp(row.get('start')),
            'off_end': _serialize_timestamp(row.get('end')),
            'on_duration': None,
            'off_duration': _serialize_value(off_duration),
            'on_magnitude': None,
            'off_magnitude': _serialize_value(row.get('magnitude')),
            'duration': None,
            'tag': None,
            'iteration': run_number,
            'threshold': threshold,
            'device_type': None,
            'values': None,
            'match_type': 'unmatched_off'
        }


def _serialize_value(v) -> Any:
    """Convert a value for JSON serialization. NaN -> None, numpy types -> Python types."""
    if v is None:
        return None
    if isinstance(v, (np.integer, np.floating)):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (int, float)):
        return round(v, 2) if isinstance(v, float) else v
    return v


def _serialize_timestamp(ts) -> Optional[str]:
    """Convert timestamp to ISO string, or None if NaT/None."""
    if ts is None or pd.isna(ts):
        return None
    if isinstance(ts, str):
        return ts
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    return str(ts)


def _json_serializer(obj):
    """JSON serializer for objects not serializable by default json module."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_classification_files(
    experiment_dir: Path,
    house_id: str,
    run_number: int,
    threshold: int,
) -> pd.DataFrame:
    """Load classification pkl files for a given run."""
    # Find run directory (supports both run_N and run_N_thXXXX naming)
    run_dir = _find_run_dir(experiment_dir, run_number)
    if run_dir is None:
        raise FileNotFoundError(f"Run directory for run_{run_number} not found")

    classification_dir = run_dir / f"house_{house_id}" / "classification"
    if not classification_dir.is_dir():
        raise FileNotFoundError(f"Classification directory not found: {classification_dir}")

    classification_files = sorted(classification_dir.glob(f"classification_{house_id}_*.pkl"))
    if not classification_files:
        raise FileNotFoundError(f"No classification files found in {classification_dir}")

    # Load and concatenate all monthly files
    dfs = [pd.read_pickle(f) for f in classification_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _load_unmatched_files(
    experiment_dir: Path,
    house_id: str,
    run_number: int,
    threshold: int,
    event_type: str,  # 'on' or 'off'
) -> pd.DataFrame:
    """Load unmatched_on or unmatched_off pkl files for a given run."""
    # Find run directory
    run_dir = _find_run_dir(experiment_dir, run_number)
    if run_dir is None:
        raise FileNotFoundError(f"Run directory for run_{run_number} not found")

    unmatched_dir = run_dir / f"house_{house_id}" / f"unmatched_{event_type}"
    if not unmatched_dir.is_dir():
        raise FileNotFoundError(f"Unmatched directory not found: {unmatched_dir}")

    unmatched_files = sorted(unmatched_dir.glob(f"unmatched_{event_type}_{house_id}_*.pkl"))
    if not unmatched_files:
        raise FileNotFoundError(f"No unmatched_{event_type} files found in {unmatched_dir}")

    # Load and concatenate all monthly files
    dfs = [pd.read_pickle(f) for f in unmatched_files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _find_run_dir(experiment_dir: Path, run_number: int) -> Optional[Path]:
    """
    Find run directory, supporting both run_N and run_N_thXXXX naming.

    Args:
        experiment_dir: Root experiment directory
        run_number: Run number to find

    Returns:
        Path to run directory, or None if not found
    """
    # Try exact match first
    exact = experiment_dir / f"run_{run_number}"
    if exact.exists():
        return exact

    # Try glob for dynamic threshold naming (run_0_th2000, etc.)
    for d in experiment_dir.glob(f"run_{run_number}_th*"):
        if d.is_dir():
            return d

    return None
