"""
Visualization pipeline step.

Creates interactive plots of segmentation results.
"""
import pandas as pd
import os
from tqdm import tqdm

from core import setup_logging, OUTPUT_BASE_PATH, LOGS_DIRECTORY, DEFAULT_THRESHOLD
from visualization import plot_interactive, split_by_day_night
from visualization.utils import get_plot_directory


def process_visualization(house_id: str, run_number: int, threshold: int = DEFAULT_THRESHOLD) -> None:
    """
    Create visualizations for segmentation results.

    Args:
        house_id: House identifier
        run_number: Current run number
        threshold: Power threshold in watts
    """
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Visualization for house {house_id}, run {run_number}")

    seg_dir = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"

    # Load event data
    events_path = f"{seg_dir}/on_off_{threshold}.csv"
    events = pd.read_csv(events_path)
    events['start'] = pd.to_datetime(events['start'], format='mixed', dayfirst=True)
    events['end'] = pd.to_datetime(events['end'], format='mixed', dayfirst=True)

    # Load summarized data
    file_path = f"{seg_dir}/summarized_{house_id}.csv"
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    try:
        phases = ['w1', 'w2', 'w3']
        _create_segment_plots(file_path, phases, logger, events)
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
