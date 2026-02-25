"""
Helper functions for saving and loading output files by month.

Output files are split by month for easier visualization:
- on_off/on_off_{threshold}_{MM}_{YYYY}.pkl
- matches/matches_{house_id}_{MM}_{YYYY}.pkl
- summarized/summarized_{house_id}_{MM}_{YYYY}.pkl
"""
import logging
import os

import pandas as pd
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_dataframe_by_month(
    df: pd.DataFrame,
    output_dir: str,
    folder_name: str,
    file_prefix: str,
    timestamp_col: str = 'start',
    date_format: str = '%d/%m/%Y %H:%M',
    show_progress: bool = False
) -> int:
    """
    Save DataFrame split by month into a folder.

    Args:
        df: DataFrame to save
        output_dir: Base output directory (e.g., run_0/house_140)
        folder_name: Subfolder name (e.g., 'on_off', 'matches')
        file_prefix: File name prefix (e.g., 'on_off_1500', 'matches_140')
        timestamp_col: Column to use for month grouping
        date_format: Date format (unused, kept for compatibility)
        show_progress: Show tqdm progress bar

    Returns:
        Number of monthly files saved
    """
    if df.empty:
        return 0

    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Ensure timestamp column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], dayfirst=True)

    # Group by year-month
    df = df.copy()
    df['_year_month'] = df[timestamp_col].dt.to_period('M')

    groups = list(df.groupby('_year_month'))
    iterator = tqdm(groups, desc=f"Saving {folder_name}", leave=False) if show_progress else groups

    months_saved = 0
    for period, group in iterator:
        month = period.month
        year = period.year
        group = group.drop(columns=['_year_month'])

        monthly_file = os.path.join(folder_path, f"{file_prefix}_{month:02d}_{year}.pkl")
        group.to_pickle(monthly_file)
        months_saved += 1

    logger.debug(f"Saved {len(df)} rows to {folder_name}/ ({months_saved} monthly files)")
    return months_saved


def load_dataframe_from_folder(
    folder_path: str,
    file_pattern: str = "*.pkl",
    parse_dates: Optional[list] = None,
    dayfirst: bool = True,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Load and concatenate all pkl files from a folder.

    Args:
        folder_path: Path to folder containing pkl files
        file_pattern: Glob pattern for files (default: *.pkl)
        parse_dates: Columns to parse as dates (unused, kept for compatibility)
        dayfirst: Whether dates are day-first format (unused, kept for compatibility)
        show_progress: Show tqdm progress bar

    Returns:
        Concatenated DataFrame from all files
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = sorted(folder.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {file_pattern} in {folder_path}")

    iterator = tqdm(files, desc=f"Loading {folder.name}", leave=False) if show_progress else files

    dfs = []
    for f in iterator:
        df = pd.read_pickle(f)
        dfs.append(df)

    logger.debug(f"Loading {len(files)} files from {folder.name}/")
    return pd.concat(dfs, ignore_index=True)


def find_output_path(
    output_dir: str,
    folder_name: str,
    file_prefix: str
) -> Union[str, Path]:
    """
    Find output path - returns folder if exists, otherwise single file path.

    Supports both new (folder with monthly files) and old (single file) structures.

    Args:
        output_dir: Base output directory
        folder_name: Subfolder name
        file_prefix: File name prefix

    Returns:
        Path to folder (if exists) or single file

    Raises:
        FileNotFoundError: If neither folder nor file exists
    """
    folder_path = Path(output_dir) / folder_name
    if folder_path.is_dir():
        return folder_path

    # Fall back to single file (old structure)
    file_path = Path(output_dir) / f"{file_prefix}.pkl"
    if file_path.is_file():
        return file_path

    raise FileNotFoundError(f"Output not found: neither {folder_path} nor {file_path} exists")
