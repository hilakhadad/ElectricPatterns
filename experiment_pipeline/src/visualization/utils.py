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
    # Support both CSV and PKL file formats
    is_pkl = filepath.endswith('.pkl')

    segments = []
    points_by_segment = []
    segment_start = None

    # Compute session start/end columns once (handle both matched and unmatched events)
    if 'start' in sessions_df.columns and 'on_start' in sessions_df.columns:
        session_start_col = sessions_df['on_start'].fillna(sessions_df['start'])
        session_end_col = sessions_df.get('off_end', sessions_df['end']).fillna(sessions_df['end'])
    elif 'on_start' in sessions_df.columns:
        session_start_col = sessions_df['on_start']
        session_end_col = sessions_df['off_end'] if 'off_end' in sessions_df.columns else sessions_df['on_end']
    else:
        session_start_col = sessions_df['start']
        session_end_col = sessions_df['end']

    if is_pkl:
        # PKL: already in memory, process as single DataFrame (no chunking)
        all_data = pd.read_pickle(filepath)
        all_data['timestamp'] = pd.to_datetime(all_data['timestamp'], format='mixed', dayfirst=True)
        all_data = all_data.sort_values('timestamp').reset_index(drop=True)

        segment_start = _get_initial_segment_start(all_data.iloc[0]['timestamp'])

        while segment_start <= all_data['timestamp'].max():
            segment_end = segment_start + timedelta(hours=12)
            segment = all_data[(all_data['timestamp'] >= segment_start) & (all_data['timestamp'] < segment_end)]

            relevant_sessions = sessions_df[
                (session_start_col < segment_end) &
                (session_end_col > segment_start)
            ]

            if not segment.empty and not relevant_sessions.empty:
                segments.append(segment)
                points_by_segment.append(relevant_sessions)

            segment_start = segment_end
    else:
        total_rows = sum(1 for _ in open(filepath)) - 1
        total_chunks = (total_rows // chunksize) + (1 if total_rows % chunksize > 0 else 0)
        chunks = pd.read_csv(filepath, chunksize=chunksize, dayfirst=True)

        for chunk in tqdm(chunks, total=total_chunks):
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='mixed', dayfirst=True)
            chunk = chunk.sort_values('timestamp').reset_index(drop=True)

            if segment_start is None:
                segment_start = _get_initial_segment_start(chunk.iloc[0]['timestamp'])

            while segment_start <= chunk['timestamp'].max():
                segment_end = segment_start + timedelta(hours=12)
                segment = chunk[(chunk['timestamp'] >= segment_start) & (chunk['timestamp'] < segment_end)]

                relevant_sessions = sessions_df[
                    (session_start_col < segment_end) &
                    (session_end_col > segment_start)
                ]

                if not segment.empty and not relevant_sessions.empty:
                    segments.append(segment)
                    points_by_segment.append(relevant_sessions)

                segment_start = segment_end

    return segments, points_by_segment


def _get_initial_segment_start(first_time: pd.Timestamp) -> pd.Timestamp:
    """Calculate the initial segment start time (06:00 or 18:00 boundary)."""
    if 6 <= first_time.hour < 18:
        # Day segment: starts at 06:00 same day
        return first_time.replace(hour=6, minute=0, second=0)
    elif first_time.hour >= 18:
        # Night segment: starts at 18:00 same day
        return first_time.replace(hour=18, minute=0, second=0)
    else:
        # Before 06:00: night segment started at 18:00 previous day
        return (first_time - timedelta(days=1)).replace(hour=18, minute=0, second=0)
