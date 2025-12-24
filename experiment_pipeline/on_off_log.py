from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
from data_util import *


def expand_event(event, data, event_type, diff):
    start, end, magnitude = event['start'], event['end'], event['magnitude']

    before_start = start - pd.Timedelta(minutes=1)
    after_end = end + pd.Timedelta(minutes=1)

    before_data = data[(data['timestamp'] >= before_start) & (data['timestamp'] < start)]
    after_data = data[(data['timestamp'] > end) & (data['timestamp'] <= after_end)]

    before_magnitude = before_data[diff].sum()
    after_magnitude = after_data[diff].sum()

    threshold_factor = 0.05 * magnitude

    if event_type == 'on':
        if before_magnitude > threshold_factor:
            start = before_start
            magnitude += before_magnitude
        if after_magnitude > threshold_factor:
            end = after_end
            magnitude += after_magnitude

    elif event_type == 'off':
        if before_magnitude < -threshold_factor:
            start = before_start
            magnitude += before_magnitude
        if after_magnitude < -threshold_factor:
            end = after_end
            magnitude += after_magnitude

    return pd.Series({'start': start, 'end': end, 'magnitude': magnitude})

def process_house(house_id, run_number, threshold=DEFAULT_THRESHOLD):
    """
    Process on/off events for a given house and run number, with an optional threshold.
    """
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Detection On and Off events process for house {house_id} for run {run_number} with threshold {threshold}.")

    # Determine input directory based on run number
    input_directory = RAW_INPUT_DIRECTORY if run_number < 1 else f"{INPUT_DIRECTORY}/run_{run_number}/HouseholdData"
    file_path = f"{input_directory}/{house_id}.csv"

    if not os.path.exists(file_path):
        logger.error(f"File {file_path} does not exist.")
        return

    try:
        data = pd.read_csv(file_path, parse_dates=['timestamp'])
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return

    data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'}, inplace=True)
    data.drop(columns=['sum'], inplace=True, errors='ignore')
    data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)

    phases = ['w1', 'w2', 'w3']
    res = pd.DataFrame()

    for phase in tqdm(phases):
        on_col = f"{phase}_{threshold}_on"
        off_col = f"{phase}_{threshold}_off"
        diff = f'{phase}_diff'

        data[diff] = data[phase].diff()

        data[on_col] = np.where(data[diff] >= threshold, data[diff], 0)
        data[off_col] = np.where(data[diff] <= -threshold, data[diff], 0)

        is_on = f'{on_col}_magnitude'
        is_off = f'{off_col}_magnitude'

        data[is_on] = (data[on_col] == 0).cumsum() * (data[on_col] != 0)
        data[is_off] = (data[off_col] == 0).cumsum() * (data[off_col] != 0)

        results_on = (
            data[data[on_col] != 0]
                .groupby(is_on, as_index=False)
                .agg(start=('timestamp', 'min'),
                     end=('timestamp', 'max'),
                     magnitude=(on_col, 'sum'))
        )

        results_off = (
            data[data[off_col] != 0]
                .groupby(is_off, as_index=False)
                .agg(start=('timestamp', 'min'),
                     end=('timestamp', 'max'),
                     magnitude=(off_col, 'sum'))
        )

        results_on = results_on.apply(lambda x: expand_event(x, data, 'on', diff), axis=1)
        results_off = results_off.apply(lambda x: expand_event(x, data, 'off', diff), axis=1)

        results_on['duration'] = (results_on['end'] - results_on['start']).dt.total_seconds() / 60
        results_on['phase'] = phase
        results_on['event'] = 'on'

        results_off['duration'] = (results_off['end'] - results_off['start']).dt.total_seconds() / 60
        results_off['phase'] = phase
        results_off['event'] = 'off'

        on_events = pd.DataFrame(results_on)
        off_events = pd.DataFrame(results_off)
        res = pd.concat([res, on_events, off_events], ignore_index=True)

    res = res.sort_values(by='start')

    output_directory = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    os.makedirs(output_directory, exist_ok=True)

    output_path = f"{output_directory}/on_off_{threshold}.csv"
    mid_res = res[(res['magnitude'] >= threshold) | (res['magnitude'] <= -threshold)].copy()

    mid_res['event_counter'] = mid_res.groupby(['phase', 'event']).cumcount() + 1
    mid_res['event_id'] = mid_res.apply(lambda row: f"{row['event']}_{row['phase']}_{row['event_counter']}", axis=1)
    mid_res.drop(columns=['event_counter'], inplace=True)

    mid_res = mid_res.sort_values(by='start')

    try:
        mid_res.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save results to {output_path}: {e}")

    logger.info(f"Processing for house {house_id} completed successfully for run {run_number}.")

if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) < 2:
        print("Usage: python on_off_log.py <house_id> <run_number> [threshold]")
        sys.exit(1)

    house_id = args[0]
    run_number = int(args[1])

    threshold = int(args[2]) if len(args) > 2 else DEFAULT_THRESHOLD

    process_house(house_id, run_number, threshold)
