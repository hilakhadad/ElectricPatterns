"""
Utility functions for visualization.
"""
import os
import re
import pandas as pd
from datetime import timedelta
from typing import List, Tuple
from tqdm import tqdm


def simplify_event_id(event_id: str) -> str:
    """
    Simplify event ID for display.

    'on_w1_42' -> 'on_42'
    """
    match = re.match(r"(on|off)_w\d+_(\d+)", event_id)
    if match:
        event_type, unique_id = match.groups()
        return f"{event_type}_{unique_id}"
    return event_id


def create_title(
    first_timestamp: pd.Series,
    columns: List[str],
    value: str,
    house_id: str
) -> str:
    """Create a title for the plot based on timestamp and columns."""
    house_id = house_id.split('_')[1] if "summarized" in house_id else house_id
    date = first_timestamp.iloc[0].date()
    first_time = first_timestamp.iloc[0].time().strftime("%H_%M")
    columns_str = '_'.join([col for col in columns if col != 'timestamp'])
    return f'{house_id}_{date}_{first_time}_{columns_str}_{value}'


def get_plot_directory(filepath: str) -> str:
    """Get or create the plot directory for a house."""
    house_id = os.path.splitext(os.path.basename(filepath))[0]
    plot_dir = os.path.join(os.path.dirname(os.path.dirname(filepath)), "plots")
    house_plot_dir = os.path.join(plot_dir, house_id)
    os.makedirs(house_plot_dir, exist_ok=True)
    return house_plot_dir


def get_save_path(title: str, filepath: str) -> str:
    """Generate full save path for a plot."""
    plot_dir = get_plot_directory(filepath)
    return os.path.join(plot_dir, title)


def split_by_day_night(
    filepath: str,
    sessions_df: pd.DataFrame,
    chunksize: int = 10000
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Split data into 12-hour day/night segments.

    Day: 06:00-18:00
    Night: 18:00-06:00

    Only returns segments that contain relevant sessions.

    Args:
        filepath: Path to CSV file
        sessions_df: DataFrame with session start/end times
        chunksize: Rows per chunk for reading large files

    Returns:
        Tuple of (segments_list, points_by_segment_list)
    """
    total_rows = sum(1 for _ in open(filepath)) - 1
    total_chunks = (total_rows // chunksize) + (1 if total_rows % chunksize > 0 else 0)

    segments = []
    points_by_segment = []
    segment_start = None

    for chunk in tqdm(pd.read_csv(filepath, chunksize=chunksize, dayfirst=True), total=total_chunks):
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='mixed', dayfirst=True)
        chunk = chunk.sort_values('timestamp').reset_index(drop=True)

        if segment_start is None:
            segment_start = _get_initial_segment_start(chunk.iloc[0]['timestamp'])

        while segment_start <= chunk['timestamp'].max():
            segment_end = segment_start + timedelta(hours=12)
            segment = chunk[(chunk['timestamp'] >= segment_start) & (chunk['timestamp'] < segment_end)]

            relevant_sessions = sessions_df[
                (sessions_df['start'] < segment_end) &
                (sessions_df['end'] > segment_start)
            ]

            if not segment.empty and not relevant_sessions.empty:
                segments.append(segment)
                points_by_segment.append(relevant_sessions)

            segment_start = segment_end

    return segments, points_by_segment


def _get_initial_segment_start(first_time: pd.Timestamp) -> pd.Timestamp:
    """Calculate the initial segment start time."""
    if 6 <= first_time.hour < 18:
        segment_start = first_time.replace(hour=6, minute=0, second=0)
    else:
        segment_start = first_time.replace(hour=18, minute=0, second=0)
        if first_time.hour < 6:
            segment_start -= timedelta(days=1)
        if segment_start < first_time:
            segment_start += timedelta(hours=12)
    return segment_start
