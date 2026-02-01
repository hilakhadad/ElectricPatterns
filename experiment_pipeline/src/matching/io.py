"""
I/O functions for matching results.
"""
import os
import pandas as pd
from pathlib import Path
from typing import List
from tqdm import tqdm


def _save_by_month(df: pd.DataFrame, output_dir: str, folder_name: str, file_prefix: str,
                   timestamp_col: str = 'on_start', show_progress: bool = False) -> int:
    """Save DataFrame split by month into a folder."""
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

        monthly_file = os.path.join(folder_path, f"{file_prefix}_{month:02d}_{year}.csv")
        group.to_csv(monthly_file, index=False, date_format='%d/%m/%Y %H:%M')
        months_saved += 1

    return months_saved


def save_events(matches: List[dict], unmatched_on: List[dict], unmatched_off: List[dict],
                output_directory: str, house_id: str) -> None:
    """
    Save matched and unmatched events to CSV files, organized by month.

    Args:
        matches: List of matched event dicts
        unmatched_on: List of unmatched ON events
        unmatched_off: List of unmatched OFF events
        output_directory: Directory to save files
        house_id: House identifier for filename
    """
    os.makedirs(output_directory, exist_ok=True)

    matches_df = pd.DataFrame(matches)
    unmatched_on_df = pd.DataFrame(unmatched_on)
    unmatched_off_df = pd.DataFrame(unmatched_off)

    # Save matches by month (use on_start for timestamp)
    if not matches_df.empty:
        months = _save_by_month(matches_df, output_directory, 'matches', f'matches_{house_id}', 'on_start')
        print(f"Saved {len(matches_df)} matches to {output_directory}/matches/ ({months} monthly files)")

    # Save unmatched_on by month (use start for timestamp)
    if not unmatched_on_df.empty:
        months = _save_by_month(unmatched_on_df, output_directory, 'unmatched_on', f'unmatched_on_{house_id}', 'start')
        print(f"Saved {len(unmatched_on_df)} unmatched ON to {output_directory}/unmatched_on/ ({months} monthly files)")

    # Save unmatched_off by month (use start for timestamp)
    if not unmatched_off_df.empty:
        months = _save_by_month(unmatched_off_df, output_directory, 'unmatched_off', f'unmatched_off_{house_id}', 'start')
        print(f"Saved {len(unmatched_off_df)} unmatched OFF to {output_directory}/unmatched_off/ ({months} monthly files)")


def save_remainder_events(remainder_events: List[dict], output_directory: str, house_id: str) -> None:
    """
    Save remainder events from partial matching to CSV for next iteration.

    Remainder events are created when Stage 3 partial matching finds a match
    between events with different magnitudes. The remainder represents the
    unmatched portion that needs to find a match in the next iteration.

    Args:
        remainder_events: List of remainder event dicts
        output_directory: Directory to save file
        house_id: House identifier for filename
    """
    if not remainder_events:
        return

    os.makedirs(output_directory, exist_ok=True)

    df = pd.DataFrame(remainder_events)
    df.to_csv(
        os.path.join(output_directory, f"remainder_{house_id}.csv"),
        index=False,
        date_format='%d/%m/%Y %H:%M'
    )
    print(f"Saved {len(remainder_events)} remainder events to {output_directory}/remainder_{house_id}.csv")
