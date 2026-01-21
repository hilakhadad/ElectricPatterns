"""
Matching pipeline step.

Matches ON events to OFF events.
"""
import pandas as pd
from tqdm import tqdm
import os

from core import (
    setup_logging, RAW_INPUT_DIRECTORY, INPUT_DIRECTORY,
    OUTPUT_BASE_PATH, DEFAULT_THRESHOLD, LOGS_DIRECTORY
)
from matching import find_match, find_noisy_match, save_events


def process_matching(house_id: str, run_number: int, threshold: int = DEFAULT_THRESHOLD) -> None:
    """
    Match ON events to OFF events for a house.

    Args:
        house_id: House identifier
        run_number: Current run number
        threshold: Detection threshold in watts
    """
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Matching process for house {house_id}, run {run_number}")

    # Paths
    data_dir = RAW_INPUT_DIRECTORY if run_number == 0 else f"{INPUT_DIRECTORY}/run_{run_number}/HouseholdData"
    log_dir = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    output_dir = log_dir

    os.makedirs(output_dir, exist_ok=True)

    data_path = f"{data_dir}/{house_id}.csv"
    log_path = f"{log_dir}/on_off_{threshold}.csv"

    if not os.path.exists(data_path) or not os.path.exists(log_path):
        logger.error(f"Missing input files for house {house_id}")
        return

    # Load data
    data = pd.read_csv(data_path, parse_dates=['timestamp'])
    log = pd.read_csv(log_path, parse_dates=['start', 'end'])

    data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'}, inplace=True)
    data.drop(columns=['sum'], inplace=True, errors='ignore')

    phases = ['w1', 'w2', 'w3']
    matches = []
    unmatched_on = []
    unmatched_off = log[log['event'] == 'off'].to_dict('records')
    matched_event_ids = set()

    # Stage 1: Clean matching
    for phase in phases:
        data[f"{phase}_diff"] = data[phase].diff()

        on_events = log[(log['phase'] == phase) & (log['event'] == 'on')].to_dict('records')
        off_events = log[(log['phase'] == phase) & (log['event'] == 'off')]

        for on_event in tqdm(on_events, desc=f"Phase {phase}", total=len(on_events)):
            matched_off, tag = find_match(data, on_event, off_events, max_time_diff=6, max_magnitude_diff=350, logger=logger)

            if matched_off is not None:
                off_id = matched_off['event_id']
                off_events = off_events[off_events['event_id'] != off_id]

                matches.append(_format_match(on_event, matched_off, tag, phase))
                matched_event_ids.add(on_event['event_id'])
                matched_event_ids.add(off_id)
                unmatched_off = [e for e in unmatched_off if e['event_id'] != off_id]
            else:
                unmatched_on.append(on_event)

    # Stage 2: Noisy matching
    logger.info(f"Stage 2: {len(unmatched_on)} unmatched ON events")

    remaining_off_df = pd.DataFrame(unmatched_off)
    if not remaining_off_df.empty:
        remaining_off_df['start'] = pd.to_datetime(remaining_off_df['start'])
        remaining_off_df['end'] = pd.to_datetime(remaining_off_df['end'])

    still_unmatched_on = []
    for on_event in tqdm(unmatched_on, desc="Noisy matching"):
        if remaining_off_df.empty:
            still_unmatched_on.append(on_event)
            continue

        matched_off, tag = find_noisy_match(data, on_event, remaining_off_df, max_time_diff=6, max_magnitude_diff=350, logger=logger)

        if matched_off is not None:
            off_id = matched_off['event_id']
            remaining_off_df = remaining_off_df[remaining_off_df['event_id'] != off_id]

            matches.append(_format_match(on_event, matched_off, tag, on_event['phase']))
            matched_event_ids.add(on_event['event_id'])
            matched_event_ids.add(off_id)
        else:
            still_unmatched_on.append(on_event)

    unmatched_on = still_unmatched_on
    unmatched_off = remaining_off_df.to_dict('records') if not remaining_off_df.empty else []

    logger.info(f"Matched: {len(matches)}, Unmatched ON: {len(unmatched_on)}, Unmatched OFF: {len(unmatched_off)}")

    # Save results
    save_events(matches, unmatched_on, unmatched_off, output_dir, house_id)

    # Update on_off CSV with matched status
    on_off_df = pd.read_csv(log_path)
    on_off_df['matched'] = on_off_df['event_id'].isin(matched_event_ids).astype(int)
    on_off_df.to_csv(log_path, index=False)
    logger.info(f"Updated {log_path} with matched status ({len(matched_event_ids)} matched)")

    logger.info(f"Matching completed for house {house_id}, run {run_number}")


def _format_match(on_event, off_event, tag, phase):
    """Format a matched event pair for saving."""
    duration = (off_event['end'] - on_event['start']).total_seconds() / 60
    return {
        'on_event_id': on_event['event_id'],
        'off_event_id': off_event['event_id'],
        'on_start': on_event['start'].strftime('%d/%m/%Y %H:%M'),
        'on_end': on_event['end'].strftime('%d/%m/%Y %H:%M'),
        'off_start': off_event['start'].strftime('%d/%m/%Y %H:%M'),
        'off_end': off_event['end'].strftime('%d/%m/%Y %H:%M'),
        'duration': duration,
        'on_magnitude': on_event['magnitude'],
        'off_magnitude': off_event['magnitude'],
        'tag': tag,
        'phase': phase
    }
