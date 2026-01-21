"""
I/O functions for matching results.
"""
import os
import pandas as pd
from typing import List


def save_events(matches: List[dict], unmatched_on: List[dict], unmatched_off: List[dict],
                output_directory: str, house_id: str) -> None:
    """
    Save matched and unmatched events to CSV files.

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

    if not matches_df.empty:
        matches_df.to_csv(
            os.path.join(output_directory, f"matches_{house_id}.csv"),
            index=False,
            date_format='%d/%m/%Y %H:%M'
        )
    if not unmatched_on_df.empty:
        unmatched_on_df.to_csv(
            os.path.join(output_directory, f"unmatched_on_{house_id}.csv"),
            index=False,
            date_format='%d/%m/%Y %H:%M'
        )
    if not unmatched_off_df.empty:
        unmatched_off_df.to_csv(
            os.path.join(output_directory, f"unmatched_off_{house_id}.csv"),
            index=False,
            date_format='%d/%m/%Y %H:%M'
        )

    print(f"Results saved to {output_directory}")
