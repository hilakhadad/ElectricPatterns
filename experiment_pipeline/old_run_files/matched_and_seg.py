import pandas as pd
from tqdm import tqdm
import sys
from data_util import *

def fill_matches_in_data_if_uniform(data, on_event, matched_off, unmatched_on, off_events, event_power_col,
                                    remaining_power, matches, diff_cal_name, logger):
    phase = on_event['phase']
    on_id = on_event['event_id']
    off_id = matched_off['event_id']
    on_range = (data['timestamp'] >= on_event['start']) & (data['timestamp'] <= on_event['end'])
    event_range = (data['timestamp'] > on_event['end']) & (data['timestamp'] < matched_off['start'])
    off_range = (data['timestamp'] >= matched_off['start']) & (data['timestamp'] <= matched_off['end'] - pd.Timedelta(minutes=1))
    magnitude = on_event['magnitude']

    event_data = data.loc[event_range, phase]
    if event_data.max() - event_data.min() > 300:
        logger.warning(f"Skipping match for phase {phase}: Non-M shape detected between {on_id} and {off_id}.")
        unmatched_on.append(on_event)
        return unmatched_on, off_events

    try:
        on_seg = data.loc[on_range, f"{on_event['phase']}_diff"].cumsum()
        on_remain = data.loc[on_range, remaining_power] - on_seg
        event_seg = magnitude + data.loc[event_range, f"{on_event['phase']}_diff"].cumsum()
        event_remain = data.loc[event_range, remaining_power] - event_seg
        off_seg = magnitude + data.loc[off_range, f"{on_event['phase']}_diff"].cumsum()
        off_remain = data.loc[off_range, remaining_power] - off_seg
    except Exception as e:
        logger.error(f"Error adjusting data ranges of {on_id} and {off_id}: {e}")
        return unmatched_on, off_events

    prev_state = {
        "on_seg": data.loc[on_range, event_power_col].copy(),
        "on_remain": data.loc[on_range, remaining_power].copy(),
        "event_seg": data.loc[event_range, event_power_col].copy(),
        "event_remain": data.loc[event_range, remaining_power].copy(),
        "off_seg": data.loc[off_range, event_power_col].copy(),
        "off_remain": data.loc[off_range, remaining_power].copy()
    }

    data.loc[on_range, event_power_col] = on_seg
    data.loc[on_range, remaining_power] = on_remain
    data.loc[event_range, event_power_col] = event_seg
    data.loc[event_range, remaining_power] = event_remain
    data.loc[off_range, event_power_col] = off_seg
    data.loc[off_range, remaining_power] = off_remain

    data[diff_cal_name] = data[remaining_power].diff()

    if (data[event_power_col] < 0).any() or (data[remaining_power] < 0).any():
        logger.warning(f"Negative values detected between {on_id} and {off_id}, reverting changes.")
        data.loc[on_range, event_power_col] = prev_state["on_seg"]
        data.loc[on_range, remaining_power] = prev_state["on_remain"]
        data.loc[event_range, event_power_col] = prev_state["event_seg"]
        data.loc[event_range, remaining_power] = prev_state["event_remain"]
        data.loc[off_range, event_power_col] = prev_state["off_seg"]
        data.loc[off_range, remaining_power] = prev_state["off_remain"]
        unmatched_on.append(on_event)
        return unmatched_on, off_events

    matches.append({
        'on_event_id': on_event['event_id'],
        'off_event_id': matched_off['event_id'],
        'on_start': on_event['start'],
        'on_end': on_event['end'],
        'off_start': matched_off['start'],
        'off_end': matched_off['end'],
        'on_magnitude': on_event['magnitude'],
        'duration': (matched_off['end'] - on_event['start']).total_seconds() / 60,
        'off_magnitude': matched_off['magnitude'],
        'phase': on_event['phase']
    })
    off_events = off_events[off_events['start'] != matched_off['start']]
    logger.info(f"Events {on_id} and {off_id} matched together!")
    return unmatched_on, off_events

def process_matches(house_id, run_number):
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info("Starting processing matches.")

    input_directory_data = RAW_INPUT_DIRECTORY if run_number == 0 else f"{INPUT_DIRECTORY}/run_{run_number}/HouseholdData"
    input_directory_log = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    output_directory = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"

    os.makedirs(output_directory, exist_ok=True)

    file_path_data = f"{input_directory_data}/{house_id}.csv"
    file_path_log = f"{input_directory_log}/on_off_1600.csv"

    if not os.path.exists(file_path_data) or not os.path.exists(file_path_log):
        logger.error(f"Missing input files for house {house_id} in run {run_number}.")
        return

    try:
        data = pd.read_csv(file_path_data, parse_dates=['timestamp'])
        log = pd.read_csv(file_path_log, parse_dates=['start', 'end'])
    except Exception as e:
        logger.error(f"Error reading files for house {house_id}: {e}")
        return

    logger.info(f"Processing matches for house {house_id} in run {run_number}.")

    data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'}, inplace=True)
    data.drop(columns=['sum'], inplace=True, errors='ignore')
    data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)

    total_on_events = log[log['event'] == 'on'].shape[0]
    total_off_events = log[log['event'] == 'off'].shape[0]
    logger.info(f"Total ON events: {total_on_events}, Total OFF events: {total_off_events}")

    matches = []

    phases = ['w1', 'w2', 'w3']
    all_unmatched_on = []
    all_unmatched_off = pd.DataFrame()

    for phase in tqdm(phases, desc=f"Processing phases for house {house_id}"):
        diff_cal_name = f'{phase}_diff'
        data[diff_cal_name] = data[phase].diff()
        remaining_power = f'remaining_power_{phase}'
        data[remaining_power] = data[phase]
        event_power_col = f'event_power_col_{phase}'
        data[event_power_col] = 0

        on_events = log[(log['phase'] == phase) & (log['event'] == 'on')]
        off_events = log[(log['phase'] == phase) & (log['event'] == 'off')]
        on_events = on_events.sort_values("start")
        off_events = off_events.sort_values("start")
        unmatched_on = []
        unmatched_off = pd.DataFrame()

        for _, on_event in tqdm(on_events.iterrows(), desc=f"Primary matching for phase {phase}", total=len(on_events)):
            potential_matches = off_events[
                (off_events['phase'] == on_event['phase']) &
                ((off_events['start'] - on_event['start']) >= pd.Timedelta(0)) &
                ((off_events['start'] - on_event['end']) <= pd.Timedelta(hours=5)) &
                (abs(off_events['magnitude'] + on_event['magnitude']) < 300)
            ]

            if not potential_matches.empty:
                potential_matches = potential_matches.copy()
                potential_matches['time_diff'] = (potential_matches['start'] - on_event['end']).abs()
                matched_off = potential_matches.loc[potential_matches['time_diff'].idxmin()]
                unmatched_on, off_events = fill_matches_in_data_if_uniform(
                    data, on_event, matched_off, unmatched_on, off_events, event_power_col, remaining_power,
                    matches, diff_cal_name, logger
                )
            else:
                unmatched_on.append(on_event)

        unmatched_off = pd.concat([unmatched_off, off_events])
        unmatched_off = unmatched_off.drop_duplicates()

        for _, on_event in tqdm(pd.DataFrame(unmatched_on).iterrows(), desc=f"Secondary matching for phase {phase}", total=len(unmatched_on)):
            potential_matches = unmatched_off[
                (unmatched_off['phase'] == on_event['phase']) &
                ((unmatched_off['start'] - on_event['start']) >= pd.Timedelta(0)) &
                ((unmatched_off['start'] - on_event['end']) <= pd.Timedelta(hours=6)) &
                (abs(unmatched_off['magnitude'] + on_event['magnitude']) < 300)
            ]

            if not potential_matches.empty:
                potential_matches = potential_matches.copy()
                potential_matches['time_diff'] = (potential_matches['start'] - on_event['end']).abs()
                matched_off = potential_matches.loc[potential_matches['time_diff'].idxmin()]
                unmatched_on, unmatched_off = fill_matches_in_data_if_uniform(
                    data, on_event, matched_off, unmatched_on, unmatched_off, event_power_col, remaining_power,
                    matches, diff_cal_name, logger
                )

        all_unmatched_off = pd.concat([all_unmatched_off, unmatched_off])
        all_unmatched_on.extend(unmatched_on)

    pd.DataFrame(matches).to_csv(f"{output_directory}/matches_{house_id}.csv", index=False)
    all_unmatched_on = pd.DataFrame(all_unmatched_on)
    all_unmatched_on.drop_duplicates(inplace=True)
    all_unmatched_on.to_csv(f"{output_directory}/unmatched_on_{house_id}.csv", index=False)
    all_unmatched_off.to_csv(f"{output_directory}/unmatched_off_{house_id}.csv", index=False)

    total_matched_events = len(matches)
    total_unmatched_on_events = len(all_unmatched_on)
    total_unmatched_off_events = all_unmatched_off.shape[0]

    if (total_matched_events * 2 + total_unmatched_on_events + total_unmatched_off_events) != (total_on_events + total_off_events):
        logger.error(
            f"Mismatch in event counts! Total ON + OFF: {total_on_events + total_off_events}, "
            f"Matched: {total_matched_events * 2}, Unmatched ON: {total_unmatched_on_events}, "
            f"Unmatched OFF: {total_unmatched_off_events}"
        )
    else:
        logger.info("Event counts validated successfully."
                    f"Matched: {total_matched_events * 2}, Unmatched ON: {total_unmatched_on_events}, "
                    f"Unmatched OFF: {total_unmatched_off_events}")

    logger.info(f"Matching for house {house_id} completed successfully for run {run_number}.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logging.error("Usage: python matched_and_seg.py <house_id> <run_number>")
    else:
        house_id = sys.argv[1]
        run_number = int(sys.argv[2])
        process_matches(house_id, run_number)
