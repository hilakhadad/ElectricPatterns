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

    logger = setup_logging(house_id, run_number, core.LOGS_DIRECTORY)
    logger.info(f"Visualization for house {house_id}, run {run_number}")

    seg_dir = Path(f"{core.OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}")
    on_off_dir = seg_dir / "on_off"
    summarized_dir = seg_dir / "summarized"

    # Load event data from monthly files
    on_off_files = sorted(on_off_dir.glob(f"on_off_{threshold}_*.csv"))
    if not on_off_files:
        logger.error(f"No on_off files found in {on_off_dir}")
        return

    events = pd.concat([pd.read_csv(f) for f in on_off_files], ignore_index=True)
    events['start'] = pd.to_datetime(events['start'], format='%d/%m/%Y %H:%M', errors='coerce')
    events['end'] = pd.to_datetime(events['end'], format='%d/%m/%Y %H:%M', errors='coerce')

    # Load summarized data from monthly files
    summarized_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.csv"))
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
                # Filter events for this month
                month_events = events[
                    (events['start'].dt.month == month) &
                    (events['start'].dt.year == year)
                ]
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
