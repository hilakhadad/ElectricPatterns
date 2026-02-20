"""
Visualization pipeline step.

Creates interactive plots of segmentation results.
"""
import pandas as pd
import os
from tqdm import tqdm

import core
from core import setup_logging, DEFAULT_THRESHOLD
from visualization import plot_interactive, split_by_day_night
from visualization.utils import get_plot_directory


def process_visualization(house_id: str, run_number: int, threshold: int = DEFAULT_THRESHOLD) -> None:
    """
    Create visualizations for segmentation results - processes monthly files.

    Args:
        house_id: House identifier
        run_number: Current run number
        threshold: Power threshold in watts
    """
    from pathlib import Path
    import pickle

    logger = setup_logging(house_id, run_number, core.LOGS_DIRECTORY)
    logger.info(f"Visualization for house {house_id}, run {run_number}")

    seg_dir = Path(f"{core.OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}")
    matches_dir = seg_dir / "matches"
    summarized_dir = seg_dir / "summarized"

    # Load both matches and unmatched individual events
    matches_files = sorted(matches_dir.glob(f"matches_{house_id}_*.pkl")) if matches_dir.exists() else []
    unmatched_on_dir = seg_dir / "unmatched_on"
    unmatched_off_dir = seg_dir / "unmatched_off"
    unmatched_on_files = sorted(unmatched_on_dir.glob(f"unmatched_on_{house_id}_*.pkl")) if unmatched_on_dir.exists() else []
    unmatched_off_files = sorted(unmatched_off_dir.glob(f"unmatched_off_{house_id}_*.pkl")) if unmatched_off_dir.exists() else []

    events_list = []

    # Load matches
    if matches_files:
        logger.info(f"Loading matches from {len(matches_files)} files")
        for matches_file in matches_files:
            try:
                with open(matches_file, 'rb') as f:
                    matches_df = pickle.load(f)
                    events_list.append(matches_df)
            except Exception as e:
                logger.warning(f"Could not load {matches_file}: {e}")

    # Load unmatched events from unmatched_on and unmatched_off directories
    all_unmatched_files = unmatched_on_files + unmatched_off_files
    if all_unmatched_files:
        logger.info(f"Loading unmatched events: {len(unmatched_on_files)} ON + {len(unmatched_off_files)} OFF files")
        for unmatched_file in all_unmatched_files:
            try:
                with open(unmatched_file, 'rb') as f:
                    unmatched_df = pickle.load(f)
                    if not unmatched_df.empty:
                        events_list.append(unmatched_df)
            except Exception as e:
                logger.warning(f"Could not load {unmatched_file}: {e}")

    if not events_list:
        logger.error("No event data could be loaded")
        return

    events = pd.concat(events_list, ignore_index=True)
    # Convert datetime columns
    for col in ['on_start', 'on_end', 'off_start', 'off_end', 'start', 'end']:
        if col in events.columns:
            events[col] = pd.to_datetime(events[col], errors='coerce')

    # Load summarized data from monthly files
    summarized_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.pkl"))
    if not summarized_files:
        logger.error(f"No summarized files found in {summarized_dir}")
        return

    try:
        phases = ['w1', 'w2', 'w3']
        # Process each monthly file separately
        for summarized_file in tqdm(summarized_files, desc=f"Visualizing {house_id}", leave=False):
            # Get events for this month
            parts = summarized_file.stem.split('_')
            if len(parts) >= 4:
                month, year = int(parts[-2]), int(parts[-1])
                # Filter events for this month - handle NaN values properly
                if 'on_start' in events.columns and 'start' in events.columns:
                    # For matched events: check on_start, for unmatched: check start
                    mask = (
                        ((pd.notna(events['on_start'])) &
                         (events['on_start'].dt.month == month) &
                         (events['on_start'].dt.year == year)) |
                        ((pd.notna(events['start'])) &
                         (events['start'].dt.month == month) &
                         (events['start'].dt.year == year))
                    )
                    month_events = events[mask].copy()
                elif 'on_start' in events.columns:
                    month_events = events[
                        (pd.notna(events['on_start'])) &
                        (events['on_start'].dt.month == month) &
                        (events['on_start'].dt.year == year)
                    ].copy()
                else:
                    month_events = events[
                        (pd.notna(events['start'])) &
                        (events['start'].dt.month == month) &
                        (events['start'].dt.year == year)
                    ].copy()
                _create_segment_plots(str(summarized_file), phases, logger, month_events)

        logger.info(f"Visualization completed for house {house_id}, run {run_number}")
    except Exception as e:
        logger.error(f"Error: {e}")


def _create_segment_plots(filepath: str, phases: list, logger, events: pd.DataFrame) -> None:
    """Create plots for each day/night segment."""
    logger.info(f"Splitting into segments: {filepath}")

    segments, points_by_segment = split_by_day_night(filepath, events)
    logger.info(f"Found {len(segments)} segments")

    get_plot_directory(filepath)

    for i, (segment, points) in tqdm(
        enumerate(zip(segments, points_by_segment)),
        desc=f"Plotting {os.path.basename(filepath)}",
        total=len(segments)
    ):
        try:
            if not points.empty:
                plot_interactive(segment, phases, filepath, logger, events=points)
        except Exception as e:
            logger.error(f"Error plotting segment {i + 1}: {e}")

    logger.info(f"Finished: {filepath}")
