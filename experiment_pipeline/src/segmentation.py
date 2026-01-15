import numpy as np
from tqdm import tqdm
import sys
from data_util import *


def process_segmentation(house_id, run_number):

    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Segmentation process for house {house_id} in run {run_number} started.")

    input_directory = f"{INPUT_DIRECTORY}/run_{run_number}/HouseholdData/" if run_number != 0 else RAW_INPUT_DIRECTORY
    events_directory = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    output_directory = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    prev_output_directory = f"{OUTPUT_BASE_PATH}/run_{run_number-1}/house_{house_id}"

    os.makedirs(output_directory, exist_ok=True)

    data_path = f"{input_directory}/{house_id}.csv"
    events_path = f"{events_directory}/matches_{house_id}.csv"
    output_path = f"{output_directory}/segmented_{house_id}.csv"
    summarized_output_path = f"{output_directory}/summarized_{house_id}.csv"
    prev_summarized_output_path = f"{prev_output_directory}/summarized_{house_id}.csv"
    next_input_path = f"{INPUT_DIRECTORY}/run_{run_number + 1}/HouseholdData/{house_id}.csv"

    os.makedirs(os.path.dirname(next_input_path), exist_ok=True)

    # Skip if outputs already exist
    if os.path.isfile(output_path) and os.path.isfile(summarized_output_path):
        logger.info(f"{house_id} files already exist. Skipping.")
        return

    # Validate input files
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    if not os.path.exists(events_path):
        logger.error(f"Events file not found: {events_path}")
        return

    try:
        data = pd.read_csv(data_path, parse_dates=['timestamp'])
        events = pd.read_csv(events_path, parse_dates=['on_start', 'on_end', 'off_start', 'off_end'], dayfirst=True)
    except Exception as e:
        logger.error(f"Failed to read input files for house {house_id}: {e}")
        return

    logger.info(f"Processing segmentation for house {house_id}, run {run_number}.")

    # Preprocessing
    data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'}, inplace=True)
    data.drop(columns=['sum'], inplace=True, errors='ignore')
    data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)

    phases = ['w1', 'w2', 'w3']
    updated_data = data.copy()
    new_columns = {}
    errors_log = []

    for phase in tqdm(phases, desc=f"Segregation for house {house_id}"):
        unique_durations = np.sort(events[events['phase'] == phase]['duration'].unique())
        updated_data[phase] = data[phase].copy()
        diff = f'diff_{phase}'
        updated_data[diff] = updated_data[phase].diff()
        remaining_power = f'remaining_power_{phase}'
        updated_data[remaining_power] = updated_data[phase].copy()

        for duration in unique_durations:
            duration = int(duration)
            event_power = f'event_power_{duration}_m_{phase}'
            event_power_col = np.zeros(len(updated_data))

            smaller_events = events[
                (events['duration'] == duration) & (events['phase'] == phase)
            ].sort_values(by='on_start', ascending=True)

            for _, event in smaller_events.iterrows():
                on_range = (updated_data['timestamp'] >= event['on_start']) & (
                            updated_data['timestamp'] <= event['on_end'])
                event_range = (updated_data['timestamp'] > event['on_end']) & (
                            updated_data['timestamp'] < event['off_start'])
                off_range = (updated_data['timestamp'] >= event['off_start']) & (
                            updated_data['timestamp'] <= event['off_end'] - pd.Timedelta(minutes=1))
                magnitude = event['on_magnitude']
                is_noisy = event.get('tag', '') == 'NOISY'

                on_seg = updated_data.loc[on_range, diff].cumsum()
                on_remain = updated_data.loc[on_range, remaining_power] - on_seg
                if (on_remain < 0).any():
                    error_indices = on_remain[on_remain < 0].index
                    error_timestamps = updated_data.loc[error_indices.intersection(updated_data.index), 'timestamp']
                    errors_log.extend(error_timestamps.tolist())
                    logger.error(
                        f"Negative values detected in ON phase {phase}, duration {duration} at timestamps: {error_timestamps.tolist()}")

                # For NOISY events, take min(magnitude, remaining) to avoid negative values
                # For regular events, use cumsum to track actual power changes
                if is_noisy:
                    # Safe subtraction: don't take more than what's available
                    current_remaining = updated_data.loc[event_range, remaining_power]
                    event_seg = current_remaining.clip(upper=magnitude)  # Take at most magnitude
                else:
                    event_seg = magnitude + updated_data.loc[event_range, diff].cumsum()

                event_remain = updated_data.loc[event_range, remaining_power] - event_seg
                if (event_remain < 0).any():
                    error_indices = event_remain[event_remain < 0].index
                    error_timestamps = updated_data.loc[error_indices.intersection(updated_data.index), 'timestamp']
                    errors_log.extend(error_timestamps.tolist())
                    logger.error(
                        f"Negative values detected in EVENT phase {phase}, duration {duration} at timestamps: {error_timestamps.tolist()}")

                # For OFF segment:
                # - Regular events: use full cumsum (device power changes tracked normally)
                # - NOISY events: align remaining to the value at off_end (when device is fully off)
                #   This way remaining stays flat at the "after device turned off" level
                if is_noisy:
                    current_remaining_off = updated_data.loc[off_range, remaining_power]
                    if len(current_remaining_off) > 0:
                        # Target = remaining at off_end (the moment after device is fully off)
                        # Note: off_range excludes off_end, so we need to look it up separately
                        off_end_mask = updated_data['timestamp'] == event['off_end']
                        if off_end_mask.any():
                            target_remaining = updated_data.loc[off_end_mask, remaining_power].iloc[0]
                        else:
                            # Fallback to last row of off_range if off_end not found
                            target_remaining = current_remaining_off.iloc[-1]
                        # off_seg = what we need to subtract to reach target_remaining
                        # off_seg = current - target (but not negative)
                        off_seg = (current_remaining_off - target_remaining).clip(lower=0)
                        off_remain = current_remaining_off - off_seg  # Will be target_remaining for all rows
                    else:
                        off_seg = current_remaining_off
                        off_remain = current_remaining_off
                else:
                    off_seg = magnitude + updated_data.loc[off_range, diff].cumsum()
                    off_remain = updated_data.loc[off_range, remaining_power] - off_seg
                if (off_remain < 0).any():
                    error_indices = off_remain[off_remain < 0].index
                    error_timestamps = updated_data.loc[error_indices.intersection(updated_data.index), 'timestamp']
                    errors_log.extend(error_timestamps.tolist())
                    logger.error(
                        f"Negative values detected in OFF phase {phase}, duration {duration} at timestamps: {error_timestamps.tolist()}")

                event_power_col[on_range] = on_seg
                updated_data.loc[on_range, remaining_power] = on_remain
                event_power_col[event_range] = event_seg
                updated_data.loc[event_range, remaining_power] = event_remain
                event_power_col[off_range] = off_seg
                updated_data.loc[off_range, remaining_power] = off_remain

            logger.info(f"Finish record all events with {duration} min duration if phase {phase}.")
            new_columns[event_power] = event_power_col

        updated_data.drop(columns=diff, inplace=True)

    # Save segmented data
    updated_data = pd.concat([updated_data, pd.DataFrame(new_columns)], axis=1)
    try:
        updated_data.to_csv(output_path, index=False)
        logger.info(f"Segmented data saved to {output_path}.")
    except Exception as e:
        logger.error(f"Failed to save segmented data: {e}")
        return

    # Summarize data
    event_columns = [col for col in updated_data.columns if 'event_power' in col]
    summarized_data = {'timestamp': updated_data['timestamp']}

    for phase in tqdm(phases, desc=f"Summarization for house {house_id}"):
        summarized_data[f'original_{phase}'] = updated_data[phase].copy()
        summarized_data[f'remaining_{phase}'] = updated_data[f'remaining_power_{phase}']

        short_sum = np.zeros(len(updated_data))
        medium_sum = np.zeros(len(updated_data))
        long_sum = np.zeros(len(updated_data))

        phase_columns = [col for col in event_columns if col.endswith(f'_m_{phase}')]

        for col in phase_columns:
            duration = int(col.split('_')[2])
            if duration <= 2:
                short_sum += updated_data[col]
            elif 2 < duration <= 24:
                medium_sum += updated_data[col]
            else:  # duration >= 25
                long_sum += updated_data[col]

        summarized_data[f'short_duration_{phase}'] = short_sum
        summarized_data[f'medium_duration_{phase}'] = medium_sum
        summarized_data[f'long_duration_{phase}'] = long_sum

    final_df = pd.DataFrame(summarized_data)

    create_negative_values_log(house_id, logger, run_number, final_df, ['original_w1', 'original_w2', 'original_w3'], 'original')
    create_negative_values_log(house_id, logger, run_number, final_df, ['remaining_w1', 'remaining_w2', 'remaining_w3'], 'remaining')
    create_negative_values_log(house_id, logger, run_number, final_df, ['short_duration_w1', 'short_duration_w2', 'short_duration_w3'], 'short_duration')
    create_negative_values_log(house_id, logger, run_number, final_df, ['medium_duration_w1', 'medium_duration_w2', 'medium_duration_w3'], 'medium_duration')
    create_negative_values_log(house_id, logger, run_number, final_df, ['long_duration_w1', 'long_duration_w2', 'long_duration_w3'], 'long_duration')

    try:
        final_df.to_csv(summarized_output_path, index=False)
        logger.info(f"Summarized data saved to {summarized_output_path}.")
    except Exception as e:
        logger.error(f"Failed to save summarized data: {e}")
        return

    # Save next input
    try:
        next_input_data = updated_data[['timestamp', 'remaining_power_w1', 'remaining_power_w2', 'remaining_power_w3']].copy()
        next_input_data.rename(columns={
            'remaining_power_w1': '1',
            'remaining_power_w2': '2',
            'remaining_power_w3': '3'
        }, inplace=True)
        next_input_data.to_csv(next_input_path, index=False)
        logger.info(f"Next input data saved to {next_input_path}.")
    except Exception as e:
        logging.error(f"Failed to save next input data: {e}")
        return

    logger.info(f"Segmentation for house {house_id} completed successfully for run {run_number}.")


import os
import pandas as pd


def create_negative_values_log(house_id, logger, run_number, data, columns_arr, source):
    """
    Logs negative values from the dataset into a CSV file.

    - If the file exists, it appends new records without overwriting.
    - If the file does not exist, it creates a new one with headers.
    - Adds a "source" column to track where the data came from.

    Args:
        house_id (str): ID of the house.
        logger (logging.Logger): Logger for logging messages.
        run_number (int): The current run number.
        data (pd.DataFrame): The dataset to check.
        columns_arr (list): List of column names to check for negative values.
        source (str): A string indicating the source of the data (e.g., "updated_data" or "original_data").
    """
    os.makedirs(ERRORS_DIRECTORY, exist_ok=True)

    # Check if there are any negative values
    if data[columns_arr].any().any() < 0:
        # Melt the data to have 'phase' and 'power' columns
        error_df = data.melt(id_vars=['timestamp'],
                             value_vars=columns_arr,
                             var_name='phase',
                             value_name='power')
        # Keep only rows where power is negative
        error_df = error_df[error_df['power'] < 0].copy()
        error_df['house_id'] = house_id
        error_df['run_number'] = run_number
        error_df['source'] = source  # Indicate the data source

        error_file_path = os.path.join(ERRORS_DIRECTORY, f"errors_{house_id}_run{run_number}.csv")

        try:
            # Check if the file exists to determine whether to write headers
            file_exists = os.path.isfile(error_file_path)

            # Append data, but write headers only if the file does not exist
            error_df.to_csv(error_file_path, index=False, mode='a', header=not file_exists)
            logger.info(f"Errors saved to {error_file_path} from source: {source}")

        except Exception as e:
            logger.error(f"Failed to save error log: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        logging.error("Usage: python segmentation.py <house_id> <run_number>")
    else:
        house_id = sys.argv[1]
        run_number = int(sys.argv[2])
        process_segmentation(house_id, run_number)
