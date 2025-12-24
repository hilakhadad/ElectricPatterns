import pandas as pd
from tqdm import tqdm
import sys
from data_util import *


def save_events(matches, unmatched_on, unmatched_off, output_directory, house_id):
    os.makedirs(output_directory, exist_ok=True)

    matches_df = pd.DataFrame(matches)
    unmatched_on_df = pd.DataFrame(unmatched_on)
    unmatched_off_df = pd.DataFrame(unmatched_off)

    if not matches_df.empty:
        matches_df.to_csv(os.path.join(output_directory, f"matches_{house_id}.csv"), index=False,
                          date_format='%d/%m/%Y %H:%M')
    if not unmatched_on_df.empty:
        unmatched_on_df.to_csv(os.path.join(output_directory, f"unmatched_on_{house_id}.csv"), index=False,
                               date_format='%d/%m/%Y %H:%M')
    if not unmatched_off_df.empty:
        unmatched_off_df.to_csv(os.path.join(output_directory, f"unmatched_off_{house_id}.csv"), index=False,
                                date_format='%d/%m/%Y %H:%M')

    print(f"Results saved to {output_directory}")


def find_match(data, on_event, off_events, max_time_diff, max_magnitude_diff, logger):
    phase = on_event['phase']
    on_id = on_event['event_id']
    on_start = on_event['start']
    on_end = on_event['end']

    logger.info(f"Processing ON event: ID={on_id}, Phase={phase}, End Time={on_end}")

    candidates = off_events[
        (off_events['phase'] == phase) &
        (off_events['start'] > on_end) &
        (off_events['start'] - on_end <= pd.Timedelta(hours=max_time_diff))
    ]

    if candidates.empty:
        logger.info(f"No matching OFF events found for {on_id} within the allowed time window.")
        return None, None

    candidates = candidates.copy()
    candidates['time_diff'] = (candidates['start'] - on_end).abs()
    candidates = candidates.sort_values(by='time_diff')

    logger.info(f"Found {len(candidates)} potential OFF events for {on_id}. Sorting by closest time difference.")

    for _, off_event in candidates.iterrows():
        off_end = off_event['end']
        off_start = off_event['start']
        off_id = off_event['event_id']
        logger.info(f"Checking for {on_id} OFF event {off_id} at {off_start} for match.")

        if (off_end - on_start).total_seconds() / 60 <= 2:
            if not is_valid_event_removal(data, on_event, off_event, logger):
                logger.warning(
                    f"Skipping incorrect match for ON event {on_id} and OFF event {off_id} due to negative residuals.")
                continue
            else:
                logger.info(f"On {on_id} and off {off_id} - matched as SPIKE: ON at {on_end}, OFF at {off_start}")
                return off_event, "SPIKE"

        event_range = (data['timestamp'] > on_end) & (data['timestamp'] < off_start)
        phase_data = data.loc[event_range, phase]

        if phase_data.empty:
            logger.info(f"No data found in the time range between On {on_id} and off {off_id} events.")
            continue

        baseline_power = phase_data.iloc[0]  # Assume the first recorded power after ON is the baseline
        max_deviation = (phase_data - baseline_power).max()
        min_deviation = (phase_data - baseline_power).min()

        logger.info(f"Between On {on_id} and off {off_id} events - Max deviation: {max_deviation}, Min deviation: {min_deviation}, Magnitude: {max_magnitude_diff}")

        if abs(max_deviation) <= max_magnitude_diff and abs(min_deviation) <= max_magnitude_diff:
            if not is_valid_event_removal(data, on_event, off_event, logger):
                logger.warning(
                    f"Skipping incorrect match for ON event {on_id} and OFF event {off_id} due to negative residuals.")
                continue
            else:
                logger.info(
                    f"On {on_id} and off {off_id} events - matched as NON-M: ON at {on_end}, OFF at {off_start}")
                return off_event, "NON-M"

    logger.info(f"No suitable OFF event found matching the criteria of {on_id}.")
    return None, None

def is_valid_event_removal(data, on_event, off_event, logger):

    phase = on_event['phase']
    on_id = on_event['event_id']
    off_id = off_event['event_id']

    if on_id == 'on_w1_103':
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    power_col = phase
    diff_col = f"{phase}_diff"  #

    on_range = (data['timestamp'] >= on_event['start']) & (data['timestamp'] <= on_event['end'])
    event_range = (data['timestamp'] > on_event['end']) & (data['timestamp'] < off_event['start'])
    off_range = (data['timestamp'] >= off_event['start']) & (data['timestamp'] <= off_event['end'] - pd.Timedelta(minutes=1))

    magnitude = on_event['magnitude']

    try:
        on_seg = data.loc[on_range, diff_col].cumsum()
        on_remain = data.loc[on_range, power_col] - on_seg
        event_seg = magnitude + data.loc[event_range, diff_col].cumsum()
        event_remain = data.loc[event_range, power_col] - event_seg
        off_seg = magnitude + data.loc[off_range, diff_col].cumsum()
        off_remain = data.loc[off_range, power_col] - off_seg
        if on_id == 'on_w1_103':
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(f"on_seg:{on_seg}")
            print(f"on_remain:{on_remain}")
            print(f"event_seg:{event_seg}")
            print(f"event_remain:{event_remain}")
            print(f"off_seg:{off_seg}")
            print(f"off_remain:{off_remain}")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    except Exception as e:
        logger.error(f"Error adjusting data ranges of {on_id} and {off_id}: {e}")
        return False

    if (on_remain < 0).any():
        logger.warning(f"Negative values detected in ON period for event {on_id} (Phase {phase}). Skipping match with OFF event {off_id}.")
        return False

    if (event_remain < 0).any():
        logger.warning(f"Negative values detected in the intermediate period between ON event {on_id} and OFF event {off_id} (Phase {phase}). Skipping match.")
        return False

    if (off_remain < 0).any():
        logger.warning(f"Negative values detected in OFF period for event {off_id} (Phase {phase}). Skipping match with ON event {on_id}.")
        return False

    return True


def process_matches(house_id, run_number, threshold):
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Match process for house {house_id} in run {run_number}.")

    input_directory_data = RAW_INPUT_DIRECTORY if run_number == 0 else f"{INPUT_DIRECTORY}/run_{run_number}/HouseholdData"
    input_directory_log = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    output_directory = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"

    os.makedirs(output_directory, exist_ok=True)

    file_path_data = f"{input_directory_data}/{house_id}.csv"
    file_path_log = f"{input_directory_log}/on_off_{threshold}.csv"

    if not os.path.exists(file_path_data) or not os.path.exists(file_path_log):
        logger.error(f"Missing input files for house {house_id} in run {run_number}.")
        return

    data = pd.read_csv(file_path_data, parse_dates=['timestamp'])
    log = pd.read_csv(file_path_log, parse_dates=['start', 'end'])

    data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'}, inplace=True)
    data.drop(columns=['sum'], inplace=True, errors='ignore')

    phases = ['w1', 'w2', 'w3']
    matches = []
    unmatched_on = []
    unmatched_off = log[log['event'] == 'off'].to_dict('records')

    # Track all event IDs
    all_event_ids = set(log['event_id'])
    matched_event_ids = set()
    unmatched_on_event_ids = set()
    unmatched_off_event_ids = set(log[log['event'] == 'off']['event_id'])  # Initially, all OFF events are unmatched

    for phase in phases:
        data[f"{phase}_diff"] = data[phase].diff()

        on_events = log[(log['phase'] == phase) & (log['event'] == 'on')].to_dict('records')
        off_events = log[(log['phase'] == phase) & (log['event'] == 'off')]

        for on_event in tqdm(on_events, desc=f"Processing phase {phase}", total=len(on_events)):
            matched_off, tag = find_match(
                data,
                on_event,
                off_events,
                max_time_diff=6,
                max_magnitude_diff=350,
                logger=logger
            )

            on_id = on_event['event_id']
            if matched_off is not None:
                off_id = matched_off['event_id']
                off_events = off_events[off_events['event_id'] != off_id]
                duration = (matched_off['end'] - on_event['start']).total_seconds() / 60

                on_magnitude = on_event['magnitude']
                off_magnitude = matched_off['magnitude']

                matches.append({
                    'on_event_id': on_id,
                    'off_event_id': off_id,
                    'on_start': on_event['start'].strftime('%d/%m/%Y %H:%M'),
                    'on_end': on_event['end'].strftime('%d/%m/%Y %H:%M'),
                    'off_start': matched_off['start'].strftime('%d/%m/%Y %H:%M'),
                    'off_end': matched_off['end'].strftime('%d/%m/%Y %H:%M'),
                    'duration': duration,
                    'on_magnitude': on_magnitude,
                    'off_magnitude': off_magnitude,
                    'tag': tag,
                    'phase': phase
                })

                # Track matched event IDs
                matched_event_ids.add(on_id)
                matched_event_ids.add(off_id)

                # Remove matched OFF event from unmatched list
                unmatched_off = [e for e in unmatched_off if e['event_id'] != off_id]
                unmatched_off_event_ids.discard(off_id)

            else:
                unmatched_on.append(on_event)
                unmatched_on_event_ids.add(on_id)

    # Remove unmatched ON and OFF event IDs from tracking set
    all_event_ids -= matched_event_ids
    all_event_ids -= unmatched_on_event_ids
    all_event_ids -= unmatched_off_event_ids

    # Ensure no duplicate IDs exist between matched and unmatched sets
    intersection_check = matched_event_ids & (unmatched_on_event_ids | unmatched_off_event_ids)
    if intersection_check:
        logger.error(f"Error: Some event IDs appear in both matched and unmatched lists! {intersection_check}")

    # Final check: all processed event IDs should account for the full original set
    total_tracked = len(matched_event_ids) + len(unmatched_on_event_ids) + len(unmatched_off_event_ids)
    total_original = len(all_event_ids) + total_tracked

    if total_original != len(log):
        logger.error(f"Warning: Mismatch in event counts. Expected {len(log)}, but tracked {total_tracked}. Missing: {len(all_event_ids)} events.")

    save_events(matches, unmatched_on, unmatched_off, output_directory, house_id)
    logger.info(f"Matching process for house {house_id}, run {run_number} completed.")


if __name__ == "__main__":
    house_id = sys.argv[1]
    run_number = int(sys.argv[2])
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_THRESHOLD

    process_matches(house_id, run_number, threshold)

