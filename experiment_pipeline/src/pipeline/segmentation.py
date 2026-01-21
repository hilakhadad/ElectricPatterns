"""
Segmentation pipeline step.

Separates power consumption into device-specific time series.
"""
import pandas as pd
from tqdm import tqdm
import os

from core import (
    setup_logging, RAW_INPUT_DIRECTORY, INPUT_DIRECTORY,
    OUTPUT_BASE_PATH, ERRORS_DIRECTORY, LOGS_DIRECTORY
)
from segmentation import process_phase_segmentation, summarize_segmentation, log_negative_values


def process_segmentation(house_id: str, run_number: int) -> None:
    """
    Process segmentation for a house.

    Args:
        house_id: House identifier
        run_number: Current run number
    """
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Segmentation process for house {house_id}, run {run_number}")

    # Paths
    input_dir = f"{INPUT_DIRECTORY}/run_{run_number}/HouseholdData/" if run_number != 0 else RAW_INPUT_DIRECTORY
    events_dir = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    output_dir = events_dir

    os.makedirs(output_dir, exist_ok=True)

    data_path = f"{input_dir}/{house_id}.csv"
    events_path = f"{events_dir}/matches_{house_id}.csv"
    output_path = f"{output_dir}/segmented_{house_id}.csv"
    summarized_path = f"{output_dir}/summarized_{house_id}.csv"
    next_input_path = f"{INPUT_DIRECTORY}/run_{run_number + 1}/HouseholdData/{house_id}.csv"

    os.makedirs(os.path.dirname(next_input_path), exist_ok=True)

    # Skip if outputs exist
    if os.path.isfile(output_path) and os.path.isfile(summarized_path):
        logger.info(f"Files exist, skipping")
        return

    # Validate inputs
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    if not os.path.exists(events_path):
        logger.error(f"Events file not found: {events_path}")
        return

    # Load data
    try:
        data = pd.read_csv(data_path, parse_dates=['timestamp'])
        events = pd.read_csv(events_path, parse_dates=['on_start', 'on_end', 'off_start', 'off_end'], dayfirst=True)
    except Exception as e:
        logger.error(f"Failed to read files: {e}")
        return

    # Preprocess
    data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'}, inplace=True)
    data.drop(columns=['sum'], inplace=True, errors='ignore')
    data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)

    phases = ['w1', 'w2', 'w3']
    all_new_columns = {}

    # Process each phase
    for phase in tqdm(phases, desc=f"Segmentation for {house_id}"):
        data, new_cols, _ = process_phase_segmentation(data, events, phase, logger)
        all_new_columns.update(new_cols)

    # Save segmented data
    data = pd.concat([data, pd.DataFrame(all_new_columns)], axis=1)
    try:
        data.to_csv(output_path, index=False)
        logger.info(f"Segmented data saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save: {e}")
        return

    # Create summary
    summary_df = summarize_segmentation(data, phases)

    # Log negative values
    for source, cols in [
        ('original', [f'original_{p}' for p in phases]),
        ('remaining', [f'remaining_{p}' for p in phases]),
        ('short_duration', [f'short_duration_{p}' for p in phases]),
        ('medium_duration', [f'medium_duration_{p}' for p in phases]),
        ('long_duration', [f'long_duration_{p}' for p in phases]),
    ]:
        log_negative_values(house_id, run_number, summary_df, cols, source, ERRORS_DIRECTORY, logger)

    # Save summary
    try:
        summary_df.to_csv(summarized_path, index=False)
        logger.info(f"Summary saved to {summarized_path}")
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")
        return

    # Save next input
    try:
        next_input = data[['timestamp', 'remaining_power_w1', 'remaining_power_w2', 'remaining_power_w3']].copy()
        next_input.rename(columns={
            'remaining_power_w1': '1',
            'remaining_power_w2': '2',
            'remaining_power_w3': '3'
        }, inplace=True)
        next_input.to_csv(next_input_path, index=False)
        logger.info(f"Next input saved to {next_input_path}")
    except Exception as e:
        logger.error(f"Failed to save next input: {e}")
        return

    logger.info(f"Segmentation completed for house {house_id}, run {run_number}")
