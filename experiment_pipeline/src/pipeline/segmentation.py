"""
Segmentation pipeline step.

Separates power consumption into device-specific time series - processes month by month.
"""
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path

import core
from core import setup_logging, load_power_data, find_house_data_path
from segmentation import process_phase_segmentation, summarize_segmentation, log_negative_values


def process_segmentation(house_id: str, run_number: int, skip_large_file: bool = True) -> None:
    """
    Process segmentation for a house - processes month by month.

    Args:
        house_id: House identifier
        run_number: Current run number
        skip_large_file: If True, skip writing the large segmented_{id}.csv file
    """
    logger = setup_logging(house_id, run_number, core.LOGS_DIRECTORY)
    logger.info(f"Segmentation process for house {house_id}, run {run_number}")

    # Paths
    input_dir = f"{core.INPUT_DIRECTORY}/run_{run_number}/HouseholdData/" if run_number != 0 else core.RAW_INPUT_DIRECTORY
    output_dir = f"{core.OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    matches_dir = Path(output_dir) / "matches"
    summarized_dir = Path(output_dir) / "summarized"
    next_input_dir = Path(f"{core.INPUT_DIRECTORY}/run_{run_number + 1}/HouseholdData/{house_id}")

    os.makedirs(summarized_dir, exist_ok=True)
    os.makedirs(next_input_dir, exist_ok=True)

    # Find data path
    try:
        data_path = find_house_data_path(input_dir, house_id)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Get list of matches monthly files
    if not matches_dir.is_dir():
        logger.error(f"Matches folder not found: {matches_dir}")
        return

    matches_files = sorted(matches_dir.glob(f"matches_{house_id}_*.pkl"))
    if not matches_files:
        logger.error(f"No matches files found in {matches_dir}")
        return

    # Get list of data monthly files
    data_path = Path(data_path)
    if data_path.is_dir():
        data_files = {f.stem: f for f in data_path.glob("*.pkl")}
    else:
        data_files = {data_path.stem: data_path}

    phases = ['w1', 'w2', 'w3']

    # Process each monthly file
    for matches_file in tqdm(matches_files, desc=f"Segmentation {house_id}", leave=False):
        # Extract month/year from filename: matches_140_01_2023.csv
        parts = matches_file.stem.split('_')
        if len(parts) >= 4:
            month, year = int(parts[-2]), int(parts[-1])
        else:
            continue

        # Skip if summarized file already exists
        summary_file = summarized_dir / f"summarized_{house_id}_{month:02d}_{year}.pkl"
        if summary_file.exists():
            logger.info(f"Skipping {month:02d}/{year} - summarized file already exists")
            continue

        # Find corresponding data file
        data_file_key = f"{house_id}_{month:02d}_{year}"
        if data_file_key not in data_files:
            data_file_key = house_id
            if data_file_key not in data_files:
                logger.warning(f"Data file not found for {month:02d}/{year}")
                continue

        # Load matches for this month
        events = pd.read_pickle(matches_file)
        if events.empty:
            continue

        # Load power data for this month
        data = load_power_data(data_files[data_file_key])
        data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)

        if data.empty:
            continue

        all_new_columns = {}
        all_skipped_ids = []

        # Process each phase
        for phase in phases:
            data, new_cols, _, skipped_ids = process_phase_segmentation(data, events, phase, logger)
            all_new_columns.update(new_cols)
            all_skipped_ids.extend(skipped_ids)

        # Update matches file: remove events that were skipped during segmentation
        if all_skipped_ids:
            original_count = len(events)
            events = events[~events['on_event_id'].isin(all_skipped_ids)]
            removed_count = original_count - len(events)
            logger.info(f"Removed {removed_count} skipped events from matches file ({month:02d}/{year})")
            events.to_pickle(matches_file)

        # Add new columns (direct assignment avoids index misalignment with pd.concat)
        for col_name, col_values in all_new_columns.items():
            data[col_name] = col_values

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
            log_negative_values(house_id, run_number, summary_df, cols, source, core.ERRORS_DIRECTORY, logger)

        # Save summary for this month
        summary_file = summarized_dir / f"summarized_{house_id}_{month:02d}_{year}.pkl"
        summary_df.to_pickle(summary_file)

        # Save next input for this month
        next_input = data[['timestamp', 'remaining_power_w1', 'remaining_power_w2', 'remaining_power_w3']].copy()
        next_input.rename(columns={
            'remaining_power_w1': '1',
            'remaining_power_w2': '2',
            'remaining_power_w3': '3'
        }, inplace=True)

        next_input_file = next_input_dir / f"{house_id}_{month:02d}_{year}.pkl"
        next_input.to_pickle(next_input_file)

    logger.info(f"Segmentation completed for house {house_id}, run {run_number}")
