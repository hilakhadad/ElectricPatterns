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


def find_noisy_match(data, on_event, off_events, max_time_diff, max_magnitude_diff, logger):
    """
    Stage 2 matcher: finds matches when there's noise (other devices) between ON and OFF.

    Criteria:
    1. Same phase
    2. OFF within time window (6 hours)
    3. ON and OFF magnitudes are similar (±max_magnitude_diff)
    4. Power never drops below baseline - 200W (device stays on)

    These matches get tag "NOISY" and will use clipped cumsum in segregation.
    """
    phase = on_event['phase']
    on_id = on_event['event_id']
    on_end = on_event['end']
    on_magnitude = abs(on_event['magnitude'])

    logger.info(f"[Stage 2] Processing unmatched ON event: ID={on_id}, Phase={phase}, Magnitude={on_magnitude}")

    # Find OFF events in same phase within time window
    candidates = off_events[
        (off_events['phase'] == phase) &
        (off_events['start'] > on_end) &
        (off_events['start'] - on_end <= pd.Timedelta(hours=max_time_diff))
    ]

    if candidates.empty:
        logger.info(f"[Stage 2] No OFF events found for {on_id} within time window.")
        return None, None

    # Filter by magnitude similarity
    candidates = candidates.copy()
    candidates['magnitude_diff'] = abs(abs(candidates['magnitude']) - on_magnitude)
    candidates = candidates[candidates['magnitude_diff'] <= max_magnitude_diff]

    if candidates.empty:
        logger.info(f"[Stage 2] No OFF events with similar magnitude found for {on_id}.")
        return None, None

    # Sort by magnitude similarity first, then by time
    candidates['time_diff'] = (candidates['start'] - on_end).abs()
    candidates = candidates.sort_values(by=['magnitude_diff', 'time_diff'])

    logger.info(f"[Stage 2] Found {len(candidates)} magnitude-compatible OFF events for {on_id}.")

    for _, off_event in candidates.iterrows():
        off_start = off_event['start']
        off_id = off_event['event_id']
        off_magnitude = abs(off_event['magnitude'])

        logger.info(f"[Stage 2] Checking {on_id} with {off_id}: ON={on_magnitude}W, OFF={off_magnitude}W")

        # Get data between ON and OFF
        event_range = (data['timestamp'] > on_end) & (data['timestamp'] < off_start)
        phase_data = data.loc[event_range, phase]

        if phase_data.empty:
            continue

        # Check that power never drops significantly below baseline
        # (device must stay on the whole time)
        baseline_power = phase_data.iloc[0]
        min_power = phase_data.min()
        min_allowed = baseline_power - 200  # Allow 200W tolerance

        if min_power < min_allowed:
            logger.info(f"[Stage 2] Rejected {on_id}-{off_id}: power dropped to {min_power}W (baseline={baseline_power}W)")
            continue

        # Validate that removal won't create negative values
        off_event_dict = off_event.to_dict()
        off_event_dict['phase'] = phase
        on_event_dict = {'start': on_event['start'], 'end': on_end, 'magnitude': on_magnitude, 'phase': phase, 'event_id': on_id}

        if not is_valid_event_removal(data, on_event_dict, off_event_dict, logger):
            logger.info(f"[Stage 2] Rejected {on_id}-{off_id}: would create negative values")
            continue

        logger.info(f"[Stage 2] Matched {on_id} with {off_id} as NOISY: ON={on_magnitude}W, OFF={off_magnitude}W")
        return off_event, "NOISY"

    logger.info(f"[Stage 2] No suitable noisy match found for {on_id}.")
    return None, None


def find_matches_stack_based(data, on_events_list, off_events_list, phase, max_time_diff, max_magnitude_diff, logger):
    """
    Stack-based matching: process events chronologically using LIFO approach.

    When an OFF event arrives, try to match it with the most recent ON event (LIFO)
    that has similar magnitude. This handles overlapping events better.

    Args:
        data: DataFrame with power data
        on_events_list: list of ON event dicts
        off_events_list: list of OFF event dicts
        phase: phase name (w1, w2, w3)
        max_time_diff: max hours between ON and OFF
        max_magnitude_diff: max watts difference for deviation check
        logger: logger instance

    Returns:
        matches: list of matched event dicts
        unmatched_on: list of unmatched ON events
        unmatched_off: list of unmatched OFF events
    """
    # Combine and sort all events chronologically
    all_events = []
    for e in on_events_list:
        all_events.append({**e, 'type': 'on', 'sort_time': e['end']})
    for e in off_events_list:
        all_events.append({**e, 'type': 'off', 'sort_time': e['start']})

    all_events.sort(key=lambda x: x['sort_time'])

    # Stack of active ON events (LIFO)
    on_stack = []
    matches = []
    unmatched_off = []

    diff_col = f"{phase}_diff"

    for event in all_events:
        if event['type'] == 'on':
            # Push ON event to stack
            on_stack.append(event)
            logger.info(f"[Stack] Pushed {event['event_id']} to stack (magnitude={event['magnitude']}W). Stack size: {len(on_stack)}")

        else:  # OFF event
            off_event = event
            off_id = off_event['event_id']
            off_magnitude = abs(off_event['magnitude'])
            off_start = off_event['start']

            logger.info(f"[Stack] Processing OFF {off_id} (magnitude={off_magnitude}W). Stack size: {len(on_stack)}")

            # Try to find matching ON from stack (LIFO order - check from end)
            matched = False
            for i in range(len(on_stack) - 1, -1, -1):
                on_event = on_stack[i]
                on_id = on_event['event_id']
                on_magnitude = abs(on_event['magnitude'])
                on_end = on_event['end']

                # Check time constraint
                time_diff_hours = (off_start - on_end).total_seconds() / 3600
                if time_diff_hours > max_time_diff or time_diff_hours < 0:
                    continue

                # Check magnitude similarity (±500W for stack matching)
                mag_diff = abs(on_magnitude - off_magnitude)
                if mag_diff > 500:
                    logger.info(f"[Stack] {on_id} magnitude mismatch with {off_id}: {on_magnitude}W vs {off_magnitude}W (diff={mag_diff}W)")
                    continue

                # Validate that removal won't create negative values
                if not is_valid_event_removal(data, on_event, off_event, logger):
                    logger.info(f"[Stack] {on_id}-{off_id} failed validation, trying next in stack")
                    continue

                # Determine tag (SPIKE or NON-M)
                duration_minutes = (off_event['end'] - on_event['start']).total_seconds() / 60
                if duration_minutes <= 2:
                    tag = "SPIKE"
                else:
                    tag = "NON-M"

                # Match found!
                matches.append({
                    'on_event_id': on_id,
                    'off_event_id': off_id,
                    'on_start': on_event['start'],
                    'on_end': on_event['end'],
                    'off_start': off_event['start'],
                    'off_end': off_event['end'],
                    'duration': duration_minutes,
                    'on_magnitude': on_event['magnitude'],
                    'off_magnitude': off_event['magnitude'],
                    'tag': tag,
                    'phase': phase
                })

                logger.info(f"[Stack] Matched {on_id} with {off_id} as {tag} (mag diff={mag_diff}W)")

                # Remove matched ON from stack
                on_stack.pop(i)
                matched = True
                break

            if not matched:
                logger.info(f"[Stack] No match found for OFF {off_id}, adding to unmatched")
                unmatched_off.append(off_event)

    # Remaining ON events in stack are unmatched
    unmatched_on = on_stack
    for on_event in unmatched_on:
        logger.info(f"[Stack] Unmatched ON: {on_event['event_id']}")

    logger.info(f"[Stack] Phase {phase} complete: {len(matches)} matches, {len(unmatched_on)} unmatched ON, {len(unmatched_off)} unmatched OFF")

    return matches, unmatched_on, unmatched_off


def is_valid_event_removal(data, on_event, off_event, logger):

    phase = on_event['phase']
    on_id = on_event['event_id']
    off_id = off_event['event_id']

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
    except Exception as e:
        logger.error(f"Error adjusting data ranges of {on_id} and {off_id}: {e}")
        return False

    # Check remaining power (what's left after removing event) - must not be negative
    if (on_remain < 0).any():
        logger.warning(f"Negative remaining in ON period for {on_id}. Skipping match with {off_id}.")
        return False

    if (event_remain < 0).any():
        logger.warning(f"Negative remaining in event period between {on_id} and {off_id}. Skipping match.")
        return False

    if (off_remain < 0).any():
        logger.warning(f"Negative remaining in OFF period for {off_id}. Skipping match with {on_id}.")
        return False

    # Check event power itself - must not be negative
    if (on_seg < 0).any():
        logger.warning(f"Negative event power in ON period for {on_id}. Skipping match with {off_id}.")
        return False

    if (event_seg < 0).any():
        logger.warning(f"Negative event power in event period between {on_id} and {off_id}. Skipping match.")
        return False

    if (off_seg < 0).any():
        logger.warning(f"Negative event power in OFF period for {off_id}. Skipping match with {on_id}.")
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

    # ============== STAGE 2: Noisy Matching ==============
    logger.info(f"Starting Stage 2 (Noisy Matching) with {len(unmatched_on)} unmatched ON events")

    # Convert unmatched_off back to DataFrame for stage 2
    remaining_off_df = pd.DataFrame(unmatched_off)
    if not remaining_off_df.empty:
        remaining_off_df['start'] = pd.to_datetime(remaining_off_df['start'])
        remaining_off_df['end'] = pd.to_datetime(remaining_off_df['end'])

    stage2_matched = []
    still_unmatched_on = []

    for on_event in tqdm(unmatched_on, desc="Stage 2: Noisy matching"):
        if remaining_off_df.empty:
            still_unmatched_on.append(on_event)
            continue

        matched_off, tag = find_noisy_match(
            data,
            on_event,
            remaining_off_df,
            max_time_diff=6,
            max_magnitude_diff=350,
            logger=logger
        )

        on_id = on_event['event_id']
        if matched_off is not None:
            off_id = matched_off['event_id']
            remaining_off_df = remaining_off_df[remaining_off_df['event_id'] != off_id]
            duration = (matched_off['end'] - on_event['start']).total_seconds() / 60

            matches.append({
                'on_event_id': on_id,
                'off_event_id': off_id,
                'on_start': on_event['start'].strftime('%d/%m/%Y %H:%M'),
                'on_end': on_event['end'].strftime('%d/%m/%Y %H:%M'),
                'off_start': matched_off['start'].strftime('%d/%m/%Y %H:%M'),
                'off_end': matched_off['end'].strftime('%d/%m/%Y %H:%M'),
                'duration': duration,
                'on_magnitude': on_event['magnitude'],
                'off_magnitude': matched_off['magnitude'],
                'tag': tag,  # "NOISY"
                'phase': on_event['phase']
            })

            matched_event_ids.add(on_id)
            matched_event_ids.add(off_id)
            unmatched_on_event_ids.discard(on_id)
            unmatched_off_event_ids.discard(off_id)
            stage2_matched.append(on_id)
        else:
            still_unmatched_on.append(on_event)

    # Update unmatched lists after stage 2
    unmatched_on = still_unmatched_on
    unmatched_off = remaining_off_df.to_dict('records') if not remaining_off_df.empty else []

    logger.info(f"Stage 2 completed: {len(stage2_matched)} additional matches found")
    # ============== END STAGE 2 ==============

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

    # Update on_off CSV with matched status column
    on_off_path = f"{input_directory_log}/on_off_{threshold}.csv"
    on_off_df = pd.read_csv(on_off_path)
    on_off_df['matched'] = on_off_df['event_id'].isin(matched_event_ids).astype(int)
    on_off_df.to_csv(on_off_path, index=False)
    logger.info(f"Updated {on_off_path} with matched status column ({matched_event_ids.__len__()} matched events)")

    logger.info(f"Matching process for house {house_id}, run {run_number} completed.")


def process_matches_stack_based(house_id, run_number, threshold):
    """
    Alternative matching using stack-based approach.
    Call this instead of process_matches to use LIFO matching.
    """
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"[Stack-based] Match process for house {house_id} in run {run_number}.")

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
    all_matches = []
    all_unmatched_on = []
    all_unmatched_off = []

    for phase in phases:
        data[f"{phase}_diff"] = data[phase].diff()

        on_events = log[(log['phase'] == phase) & (log['event'] == 'on')].to_dict('records')
        off_events = log[(log['phase'] == phase) & (log['event'] == 'off')].to_dict('records')

        matches, unmatched_on, unmatched_off = find_matches_stack_based(
            data, on_events, off_events, phase,
            max_time_diff=6,
            max_magnitude_diff=350,
            logger=logger
        )

        all_matches.extend(matches)
        all_unmatched_on.extend(unmatched_on)
        all_unmatched_off.extend(unmatched_off)

    # Format matches for saving (convert timestamps to strings)
    formatted_matches = []
    for m in all_matches:
        formatted_matches.append({
            'on_event_id': m['on_event_id'],
            'off_event_id': m['off_event_id'],
            'on_start': m['on_start'].strftime('%d/%m/%Y %H:%M'),
            'on_end': m['on_end'].strftime('%d/%m/%Y %H:%M'),
            'off_start': m['off_start'].strftime('%d/%m/%Y %H:%M'),
            'off_end': m['off_end'].strftime('%d/%m/%Y %H:%M'),
            'duration': m['duration'],
            'on_magnitude': m['on_magnitude'],
            'off_magnitude': m['off_magnitude'],
            'tag': m['tag'],
            'phase': m['phase']
        })

    save_events(formatted_matches, all_unmatched_on, all_unmatched_off, output_directory, house_id)

    # Track matched IDs
    matched_event_ids = set()
    for m in all_matches:
        matched_event_ids.add(m['on_event_id'])
        matched_event_ids.add(m['off_event_id'])

    # Update on_off CSV with matched status column
    on_off_path = f"{input_directory_log}/on_off_{threshold}.csv"
    on_off_df = pd.read_csv(on_off_path)
    on_off_df['matched'] = on_off_df['event_id'].isin(matched_event_ids).astype(int)
    on_off_df.to_csv(on_off_path, index=False)

    logger.info(f"[Stack-based] Matching completed: {len(all_matches)} matches, {len(matched_event_ids)} matched events")


if __name__ == "__main__":
    house_id = sys.argv[1]
    run_number = int(sys.argv[2])
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_THRESHOLD

    # Use --stack flag to use stack-based matching
    if len(sys.argv) > 4 and sys.argv[4] == '--stack':
        process_matches_stack_based(house_id, run_number, threshold)
    else:
        process_matches(house_id, run_number, threshold)

