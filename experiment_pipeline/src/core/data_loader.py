"""
Data loading utilities for experiment pipeline.

Provides consistent data loading with proper column naming.
Supports both single pkl files and folders with monthly files.
"""
import pandas as pd
from pathlib import Path
from typing import Union


def load_power_data(filepath: Union[str, Path], parse_dates: bool = True) -> pd.DataFrame:
    """
    Load power data with standardized column names.

    Supports three input formats:
    1. Single pkl file: loads directly
    2. Folder with monthly files: loads and concatenates all pkl files in folder
    3. Summarized pkl file: extracts remaining_w1/w2/w3 as w1/w2/w3

    Automatically renames columns from raw format ('1', '2', '3')
    to phase format ('w1', 'w2', 'w3') and drops 'sum' column if present.

    Args:
        filepath: Path to pkl file OR folder containing monthly pkl files
        parse_dates: Whether to parse timestamp column as datetime (ignored for pkl)

    Returns:
        DataFrame with columns: timestamp, w1, w2, w3 (sorted by timestamp)
    """
    path = Path(filepath)

    # Check if it's a folder (monthly files structure)
    if path.is_dir():
        data = _load_from_folder(path, parse_dates)
    else:
        data = _load_single_file(path, parse_dates)

    # Handle summarized format (remaining_w1 → w1)
    if 'remaining_w1' in data.columns:
        result = data[['timestamp', 'remaining_w1', 'remaining_w2', 'remaining_w3']].copy()
        result.rename(columns={
            'remaining_w1': 'w1', 'remaining_w2': 'w2', 'remaining_w3': 'w3'
        }, inplace=True)
        return result

    # Rename columns if in raw format
    if '1' in data.columns:
        data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'}, inplace=True)

    # Drop sum column if present
    if 'sum' in data.columns:
        data.drop(columns=['sum'], inplace=True)

    return data


def _load_single_file(filepath: Path, parse_dates: bool) -> pd.DataFrame:
    """Load a single pkl file."""
    return pd.read_pickle(filepath).reset_index(drop=True)


def _load_from_folder(folder: Path, parse_dates: bool) -> pd.DataFrame:
    """Load and concatenate all pkl files from a folder."""
    pkl_files = sorted(folder.glob("*.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No pkl files found in {folder}")

    dfs = []
    for pkl_file in pkl_files:
        df = _load_single_file(pkl_file, parse_dates)
        dfs.append(df)

    # Concatenate all monthly files
    data = pd.concat(dfs, ignore_index=True)

    # Sort by timestamp to ensure chronological order
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp').reset_index(drop=True)

    return data


def get_monthly_files(house_folder: Union[str, Path]) -> list:
    """
    Get list of monthly files for a house folder.

    Args:
        house_folder: Path to house folder (e.g., INPUT/HouseholdData/1001/)

    Returns:
        List of Path objects sorted by date
    """
    folder = Path(house_folder)
    if not folder.is_dir():
        return []

    return sorted(folder.glob("*.pkl"))


def find_house_data_path(base_dir: Union[str, Path], house_id: str) -> Path:
    """
    Find the data path for a house - either folder (new) or file (old).

    Args:
        base_dir: Base directory containing house data
        house_id: House identifier

    Returns:
        Path to folder (if exists) or pkl file

    Raises:
        FileNotFoundError: If neither folder nor file exists
    """
    base = Path(base_dir)

    # Try folder first (new structure)
    folder_path = base / house_id
    if folder_path.is_dir():
        return folder_path

    # Fall back to single file (old structure)
    file_path = base / f"{house_id}.pkl"
    if file_path.is_file():
        return file_path

    raise FileNotFoundError(f"House data not found: neither {folder_path} nor {file_path} exists")


def load_single_month(house_folder: Union[str, Path], month: int, year: int) -> pd.DataFrame:
    """
    Load data for a specific month.

    Args:
        house_folder: Path to house folder
        month: Month number (1-12)
        year: Year (e.g., 2020)

    Returns:
        DataFrame for that month
    """
    folder = Path(house_folder)
    house_id = folder.name

    monthly_file = folder / f"{house_id}_{month:02d}_{year}.pkl"

    if not monthly_file.exists():
        raise FileNotFoundError(f"Monthly file not found: {monthly_file}")

    return load_power_data(monthly_file)


def find_previous_run_summarized(output_base: Union[str, Path], house_id: str, run_number: int) -> Path:
    """
    Find summarized directory from the previous run to use as input for the current run.

    Instead of writing separate HouseholdData files for each iteration,
    the pipeline reads remaining power directly from the previous run's
    summarized output (remaining_w1/w2/w3 columns).

    Args:
        output_base: Base output path (OUTPUT_BASE_PATH)
        house_id: House identifier
        run_number: Current run number (will look for run_number - 1)

    Returns:
        Path to the summarized directory from the previous run

    Raises:
        FileNotFoundError: If no summarized data exists from the previous run
    """
    prev_run = run_number - 1
    path = Path(output_base) / f"run_{prev_run}" / f"house_{house_id}" / "summarized"
    if path.is_dir() and any(path.glob(f"summarized_{house_id}_*.pkl")):
        return path
    raise FileNotFoundError(
        f"No summarized data from run {prev_run} for house {house_id} at {path}"
    )


def build_data_files_dict(data_path: Union[str, Path]) -> dict:
    """
    Build a {key: filepath} dict for monthly data files.

    Normalizes keys for both HouseholdData format ({house_id}_MM_YYYY)
    and summarized format (summarized_{house_id}_MM_YYYY → {house_id}_MM_YYYY).

    Args:
        data_path: Path to directory containing pkl files

    Returns:
        Dict mapping normalized stem → Path
    """
    data_path = Path(data_path)
    data_files = {}
    for f in data_path.glob("*.pkl"):
        key = f.stem
        # Strip 'summarized_' prefix for consistent key format
        if key.startswith('summarized_'):
            key = key[len('summarized_'):]
        data_files[key] = f
    return data_files
