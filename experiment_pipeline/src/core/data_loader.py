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

    Supports two input formats:
    1. Single pkl file: loads directly
    2. Folder with monthly files: loads and concatenates all pkl files in folder

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
