"""
Data loading utilities for user plot requests.

Handles loading data from experiment outputs or INPUT directory.
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd


# Project paths
_MODULE_DIR = Path(__file__).parent.parent.absolute()
_PROJECT_ROOT = _MODULE_DIR.parent
_EXPERIMENT_OUTPUT = _PROJECT_ROOT / "experiment_pipeline" / "OUTPUT" / "experiments"
_INPUT_DIR = _PROJECT_ROOT / "INPUT" / "HouseholdData"


def _sort_monthly_files(files: List[Path]) -> List[Path]:
    """
    Sort monthly files chronologically by parsing MM_YYYY from filename.

    Handles filenames like: summarized_10_01_2021.pkl (house 10, Jan 2021)
    """
    def get_date_key(f: Path):
        # Filename: summarized_{house_id}_{MM}_{YYYY}.pkl or {house_id}_{MM}_{YYYY}.pkl
        parts = f.stem.split('_')
        try:
            # Try to find MM_YYYY pattern (last two parts)
            if len(parts) >= 2:
                month = int(parts[-2])
                year = int(parts[-1])
                return (year, month)
        except (ValueError, IndexError):
            pass
        return (0, 0)  # Fallback for non-standard names

    return sorted(files, key=get_date_key)


def get_available_houses() -> List[str]:
    """
    Get list of all available house IDs from INPUT directory.

    Supports both single CSV files (house_id.pkl) and monthly folder structure
    (house_id/house_id_MM_YYYY.pkl).

    Returns:
        List of house IDs (sorted numerically)
    """
    if not _INPUT_DIR.exists():
        return []

    houses = set()

    # Check for single CSV files (old structure)
    for f in _INPUT_DIR.glob("*.pkl"):
        house_id = f.stem
        # Only include numeric house IDs (not house_id_MM_YYYY format)
        if house_id.isdigit():
            houses.add(house_id)

    # Check for folders with monthly files (new structure)
    for d in _INPUT_DIR.iterdir():
        if d.is_dir() and d.name.isdigit():
            # Verify folder contains CSV files
            if list(d.glob("*.pkl")):
                houses.add(d.name)

    return sorted(houses, key=lambda x: int(x))


def get_latest_experiment() -> Optional[Path]:
    """
    Find the most recent experiment output directory.

    Searches in both OUTPUT/ and OUTPUT/experiments/.

    Returns:
        Path to latest experiment or None if not found
    """
    # Search in both OUTPUT and OUTPUT/experiments
    output_dir = _PROJECT_ROOT / "experiment_pipeline" / "OUTPUT"
    search_dirs = [output_dir]
    if _EXPERIMENT_OUTPUT.exists():
        search_dirs.append(_EXPERIMENT_OUTPUT)

    experiments = []
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for item in search_dir.iterdir():
            if item.is_dir():
                # Check if it looks like an experiment directory
                # New structure: experiment_dir/run_0/house_X/
                has_new_structure = any(
                    (item / f"run_0" / d.name).exists()
                    for d in item.iterdir()
                    if d.is_dir() and d.name.startswith("house_")
                ) if (item / "run_0").exists() else False

                # Old structure: experiment_dir/house_X/run_0/
                has_old_structure = any(
                    (item / d.name / "run_0").exists()
                    for d in item.iterdir()
                    if d.is_dir() and d.name.startswith("house_")
                )

                if has_new_structure or has_old_structure:
                    experiments.append(item)

    if not experiments:
        return None

    # Sort by modification time (most recent first)
    experiments.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return experiments[0]


def get_house_date_range(house_id: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Get the date range available for a house.

    Reads first and last timestamps from the house's data file.
    Supports both new (run_N/house_X/) and old (house_X/run_N/house_X/) structures.

    Args:
        house_id: House ID to check

    Returns:
        Tuple of (start_date, end_date) or None if not found
    """
    # Try experiment output first
    experiment = get_latest_experiment()
    if experiment:
        # Try new structure: experiment/run_0/house_{id}/summarized/
        run_dir = experiment / "run_0" / f"house_{house_id}"
        if run_dir.exists():
            summarized_dir = run_dir / "summarized"
            if summarized_dir.exists():
                summarized_files = _sort_monthly_files(list(summarized_dir.glob(f"summarized_{house_id}_*.pkl")))
                if summarized_files:
                    # Get date range from first and last monthly file
                    first_range = _read_date_range(summarized_files[0])
                    last_range = _read_date_range(summarized_files[-1])
                    if first_range and last_range:
                        return (first_range[0], last_range[1])

        # Fall back to old structure: experiment/house_{id}/run_0/house_{id}/
        house_dir = experiment / f"house_{house_id}"
        if house_dir.exists():
            run_dir = house_dir / "run_0" / f"house_{house_id}"
            if run_dir.exists():
                # Check summarized/ subfolder first
                summarized_subdir = run_dir / "summarized"
                if summarized_subdir.exists():
                    summarized_files = _sort_monthly_files(list(summarized_subdir.glob(f"summarized_{house_id}_*.pkl")))
                    if summarized_files:
                        first_range = _read_date_range(summarized_files[0])
                        last_range = _read_date_range(summarized_files[-1])
                        if first_range and last_range:
                            return (first_range[0], last_range[1])
                # Check for files directly in run_dir
                summarized_files = list(run_dir.glob("summarized_*.pkl"))
                if summarized_files:
                    return _read_date_range(summarized_files[0])

    # Fall back to INPUT directory
    input_file = _INPUT_DIR / f"{house_id}.pkl"
    if input_file.exists():
        return _read_date_range(input_file)

    # Try monthly folder in INPUT
    input_folder = _INPUT_DIR / house_id
    if input_folder.exists():
        monthly_files = _sort_monthly_files(list(input_folder.glob("*.pkl")))
        if monthly_files:
            first_range = _read_date_range(monthly_files[0])
            last_range = _read_date_range(monthly_files[-1])
            if first_range and last_range:
                return (first_range[0], last_range[1])

    return None


def _read_date_range(file_path: Path) -> Optional[Tuple[datetime, datetime]]:
    """Read first and last timestamps from a pkl file."""
    try:
        df = pd.read_pickle(file_path)
        first_ts = df['timestamp'].iloc[0]
        last_ts = df['timestamp'].iloc[-1]

        # Ensure timestamps are datetime
        if not isinstance(first_ts, (datetime, pd.Timestamp)):
            first_ts = pd.to_datetime(first_ts, format='mixed', dayfirst=True)
        if not isinstance(last_ts, (datetime, pd.Timestamp)):
            last_ts = pd.to_datetime(last_ts, format='mixed', dayfirst=True)

        return (pd.Timestamp(first_ts).to_pydatetime(), pd.Timestamp(last_ts).to_pydatetime())
    except Exception:
        return None


def load_house_data(house_id: str, run_number: int = 0) -> Optional[pd.DataFrame]:
    """
    Load summarized data for a house.

    First tries experiment output, then falls back to INPUT.
    Supports both new (run_N/house_X/) and old (house_X/run_N/house_X/) structures.

    Args:
        house_id: House ID to load
        run_number: Run number (iteration) to load from

    Returns:
        DataFrame with timestamp and power columns, or None
    """
    # Try experiment output first
    experiment = get_latest_experiment()
    if experiment:
        # Try new structure: experiment/run_X/house_{id}/summarized/
        for run_num in [run_number, 0]:
            run_dir = experiment / f"run_{run_num}" / f"house_{house_id}"
            if run_dir.exists():
                summarized_dir = run_dir / "summarized"
                if summarized_dir.exists():
                    summarized_files = _sort_monthly_files(list(summarized_dir.glob(f"summarized_{house_id}_*.pkl")))
                    if summarized_files:
                        # Concatenate all monthly files
                        dfs = [_load_pkl(f) for f in summarized_files]
                        return pd.concat(dfs, ignore_index=True)

        # Fall back to old structure: experiment/house_{id}/run_X/house_{id}/
        house_dir = experiment / f"house_{house_id}"
        if house_dir.exists():
            for run_dir in [house_dir / f"run_{run_number}", house_dir / "run_0"]:
                if not run_dir.exists():
                    continue
                inner_house_dir = run_dir / f"house_{house_id}"
                if inner_house_dir.exists():
                    # Check for summarized/ subfolder (new format)
                    summarized_subdir = inner_house_dir / "summarized"
                    if summarized_subdir.exists():
                        summarized_files = _sort_monthly_files(list(summarized_subdir.glob(f"summarized_{house_id}_*.pkl")))
                        if summarized_files:
                            dfs = [_load_pkl(f) for f in summarized_files]
                            return pd.concat(dfs, ignore_index=True)
                    # Check for summarized_{id}.pkl directly
                    summarized_file = inner_house_dir / f"summarized_{house_id}.pkl"
                    if summarized_file.exists():
                        return _load_pkl(summarized_file)
                    # Also search for any summarized_*.pkl file
                    summarized_files = list(inner_house_dir.glob("summarized_*.pkl"))
                    if summarized_files:
                        return _load_pkl(summarized_files[0])

    # Fall back to INPUT directory (raw data, no segmentation)
    input_file = _INPUT_DIR / f"{house_id}.pkl"
    if input_file.exists():
        return _load_pkl(input_file)

    # Try monthly folder in INPUT
    input_folder = _INPUT_DIR / house_id
    if input_folder.exists():
        monthly_files = _sort_monthly_files(list(input_folder.glob("*.pkl")))
        if monthly_files:
            dfs = [_load_pkl(f) for f in monthly_files]
            return pd.concat(dfs, ignore_index=True)

    return None


def load_events_data(house_id: str, run_number: int = 0) -> Optional[pd.DataFrame]:
    """
    Load on/off events data for a house.

    Supports both new (run_N/house_X/on_off/) and old structures.

    Args:
        house_id: House ID to load
        run_number: Run number (iteration) to load from

    Returns:
        DataFrame with event data, or None
    """
    experiment = get_latest_experiment()
    if not experiment:
        return None

    # Try new structure: experiment/run_X/house_{id}/on_off/
    for run_num in [run_number, 0]:
        run_dir = experiment / f"run_{run_num}" / f"house_{house_id}"
        if run_dir.exists():
            on_off_dir = run_dir / "on_off"
            if on_off_dir.exists():
                on_off_files = sorted(on_off_dir.glob("on_off_*.pkl"))
                if on_off_files:
                    dfs = [pd.read_pickle(f) for f in on_off_files]
                    df = pd.concat(dfs, ignore_index=True)
                    df['start'] = pd.to_datetime(df['start'], format='%d/%m/%Y %H:%M', errors='coerce')
                    df['end'] = pd.to_datetime(df['end'], format='%d/%m/%Y %H:%M', errors='coerce')
                    return df

    # Fall back to old structure: experiment/house_{id}/run_X/house_{id}/
    house_dir = experiment / f"house_{house_id}"
    if not house_dir.exists():
        return None

    for run_dir in [house_dir / f"run_{run_number}", house_dir / "run_0"]:
        if not run_dir.exists():
            continue
        inner_house_dir = run_dir / f"house_{house_id}"
        if inner_house_dir.exists():
            # Check for on_off/ subfolder first
            on_off_subdir = inner_house_dir / "on_off"
            if on_off_subdir.exists():
                on_off_files = _sort_monthly_files(list(on_off_subdir.glob("on_off_*.pkl")))
                if on_off_files:
                    dfs = [pd.read_pickle(f) for f in on_off_files]
                    df = pd.concat(dfs, ignore_index=True)
                    df['start'] = pd.to_datetime(df['start'], format='mixed', dayfirst=True)
                    df['end'] = pd.to_datetime(df['end'], format='mixed', dayfirst=True)
                    return df
            # Check for files directly in inner_house_dir
            on_off_files = list(inner_house_dir.glob("on_off_*.pkl"))
            if on_off_files:
                df = pd.read_pickle(on_off_files[0])
                df['start'] = pd.to_datetime(df['start'], format='mixed', dayfirst=True)
                df['end'] = pd.to_datetime(df['end'], format='mixed', dayfirst=True)
                return df

    return None


def _load_pkl(file_path: Path) -> pd.DataFrame:
    """Load pkl file."""
    df = pd.read_pickle(file_path)
    # Ensure timestamp is datetime (pkl should preserve this, but just in case)
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
    return df


def filter_data_by_window(
    df: pd.DataFrame,
    center_time: datetime,
    window_type: str = 'day'
) -> pd.DataFrame:
    """
    Filter data to a 12-hour window.

    Args:
        df: DataFrame with 'timestamp' column
        center_time: Center time for the window
        window_type: 'day' (06:00-18:00) or 'night' (18:00-06:00)

    Returns:
        Filtered DataFrame
    """
    if window_type == 'day':
        # 06:00 to 18:00
        start = center_time.replace(hour=6, minute=0, second=0, microsecond=0)
        end = center_time.replace(hour=18, minute=0, second=0, microsecond=0)
    else:
        # 18:00 to 06:00 next day
        start = center_time.replace(hour=18, minute=0, second=0, microsecond=0)
        end = (center_time + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)

    return df[(df['timestamp'] >= start) & (df['timestamp'] <= end)].copy()


def get_houses_info() -> List[Dict]:
    """
    Get information about all available houses.

    Supports both new (run_N/house_X/) and old (house_X/) structures.

    Returns:
        List of dicts with house_id, start_date, end_date, has_experiment
    """
    houses = get_available_houses()
    experiment = get_latest_experiment()

    info = []
    for house_id in houses:
        date_range = get_house_date_range(house_id)
        has_exp = False

        if experiment:
            # Check new structure: experiment/run_0/house_{id}/summarized/
            new_house_dir = experiment / "run_0" / f"house_{house_id}"
            if new_house_dir.exists():
                summarized_dir = new_house_dir / "summarized"
                if summarized_dir.exists() and list(summarized_dir.glob(f"summarized_{house_id}_*.pkl")):
                    has_exp = True

            # Check old structure: experiment/house_{id}/run_0/house_{id}/
            if not has_exp:
                old_house_dir = experiment / f"house_{house_id}"
                if old_house_dir.exists():
                    run_dir = old_house_dir / "run_0" / f"house_{house_id}"
                    if run_dir.exists():
                        # Check summarized/ subfolder first
                        summarized_subdir = run_dir / "summarized"
                        if summarized_subdir.exists() and list(summarized_subdir.glob("summarized_*.pkl")):
                            has_exp = True
                        # Also check for files directly in run_dir
                        elif list(run_dir.glob("summarized_*.pkl")):
                            has_exp = True

        if date_range:
            info.append({
                'house_id': house_id,
                'start_date': date_range[0].strftime('%Y-%m-%d'),
                'end_date': date_range[1].strftime('%Y-%m-%d'),
                'has_experiment': has_exp
            })

    return info
