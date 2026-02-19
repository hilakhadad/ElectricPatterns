"""
Matching pipeline step.

Matches ON events to OFF events - processes month by month.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

import core
from core import setup_logging, DEFAULT_THRESHOLD, load_power_data, find_house_data_path, find_previous_run_summarized, build_data_files_dict
from matching import find_match, find_noisy_match, find_partial_match


def process_matching(house_id: str, run_number: int, threshold: int = DEFAULT_THRESHOLD) -> None:
    """
    Match ON events to OFF events for a house - processes month by month.

    Args:
        house_id: House identifier
        run_number: Current run number
        threshold: Detection threshold in watts
    """
    logger = setup_logging(house_id, run_number, core.LOGS_DIRECTORY)
    logger.info(f"Matching process for house {house_id}, run {run_number}")

    # Paths
    output_dir = f"{core.OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    on_off_dir = Path(output_dir) / "on_off"
    matches_dir = Path(output_dir) / "matches"
    unmatched_on_dir = Path(output_dir) / "unmatched_on"
    unmatched_off_dir = Path(output_dir) / "unmatched_off"

    os.makedirs(matches_dir, exist_ok=True)
    os.makedirs(unmatched_on_dir, exist_ok=True)
    os.makedirs(unmatched_off_dir, exist_ok=True)

    # Find data path: run 0 reads raw data, run N reads remaining from summarized of run N-1
    try:
        if run_number == 0:
            data_path = find_house_data_path(core.RAW_INPUT_DIRECTORY, house_id)
        else:
            data_path = find_previous_run_summarized(core.OUTPUT_BASE_PATH, house_id, run_number)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Get list of on_off monthly files
    if not on_off_dir.is_dir():
        logger.error(f"On/off folder not found: {on_off_dir}")
        return

    on_off_files = sorted(on_off_dir.glob(f"on_off_{threshold}_*.pkl"))
    if not on_off_files:
        logger.error(f"No on_off files found in {on_off_dir}")
        return

    # Get list of data monthly files (handles both HouseholdData and summarized naming)
    data_path = Path(data_path)
    if data_path.is_dir():
        data_files = build_data_files_dict(data_path)
    else:
        data_files = {data_path.stem: data_path}

    phases = ['w1', 'w2', 'w3']
    total_matches = 0
    total_unmatched_on = 0
    total_unmatched_off = 0

    # Process each monthly file
    for on_off_file in tqdm(on_off_files, desc=f"Matching {house_id}", leave=False):
        # Extract month/year from filename: on_off_1500_01_2023.csv
        parts = on_off_file.stem.split('_')
        if len(parts) >= 4:
            month, year = int(parts[-2]), int(parts[-1])
        else:
            continue

        # Skip if matches file already exists
        matches_file = matches_dir / f"matches_{house_id}_{month:02d}_{year}.pkl"
        if matches_file.exists():
            logger.info(f"Skipping {month:02d}/{year} - matches file already exists")
            continue

        # Find corresponding data file
        data_file_key = f"{house_id}_{month:02d}_{year}"
        if data_file_key not in data_files:
            data_file_key = house_id
            if data_file_key not in data_files:
                logger.warning(f"Data file not found for {month:02d}/{year}")
                continue

        # Load on_off events for this month
        on_off_df = pd.read_pickle(on_off_file)
        if on_off_df.empty:
            continue

        # Load power data for this month
        data = load_power_data(data_files[data_file_key])

        # Pre-compute numpy timestamps once — avoids repeated .astype('datetime64[ns]')
        # conversion inside validator (called ~43K times/house). The validator checks for
        # this column and uses it directly instead of re-converting each call.
        data['_np_ts'] = data['timestamp'].values.astype('datetime64[ns]')

        # Pre-compute diff columns
        for phase in phases:
            data[f"{phase}_diff"] = data[phase].diff()

        # Run matching for this month
        matches = []
        unmatched_on = []
        matched_event_ids = set()
        used_off_ids = set()

        # Stage 1: Clean matching
        for phase in phases:
            on_events = on_off_df[(on_off_df['phase'] == phase) & (on_off_df['event'] == 'on')].to_dict('records')
            off_events_df = on_off_df[(on_off_df['phase'] == phase) & (on_off_df['event'] == 'off')]

            for on_event in on_events:
                available_off = off_events_df[~off_events_df['event_id'].isin(used_off_ids)]
                matched_off, tag, correction = find_match(data, on_event, available_off, max_time_diff=6, max_magnitude_diff=350, logger=logger)

                if matched_off is not None:
                    off_id = matched_off['event_id']
                    used_off_ids.add(off_id)
                    matches.append(_format_match(on_event, matched_off, tag, phase, correction))
                    matched_event_ids.add(on_event['event_id'])
                    matched_event_ids.add(off_id)
                else:
                    unmatched_on.append(on_event)

        # Stage 2: Noisy matching
        all_off_events = on_off_df[on_off_df['event'] == 'off']
        still_unmatched_on = []

        for on_event in unmatched_on:
            available_off = all_off_events[~all_off_events['event_id'].isin(used_off_ids)]
            if available_off.empty:
                still_unmatched_on.append(on_event)
                continue

            matched_off, tag, correction = find_noisy_match(data, on_event, available_off, max_time_diff=6, max_magnitude_diff=350, logger=logger)

            if matched_off is not None:
                off_id = matched_off['event_id']
                used_off_ids.add(off_id)
                matches.append(_format_match(on_event, matched_off, tag, on_event['phase'], correction))
                matched_event_ids.add(on_event['event_id'])
                matched_event_ids.add(off_id)
            else:
                still_unmatched_on.append(on_event)

        unmatched_on = still_unmatched_on

        # Log ON events that failed Stage 1+2 (going to Stage 3)
        if unmatched_on:
            logger.info(f"  {len(unmatched_on)} ON events unmatched after Stage 1+2, trying Stage 3")

        # Stage 3: Partial matching
        final_unmatched_on = []
        remainder_on_events = []
        remainder_off_events = []

        for on_event in unmatched_on:
            available_off = all_off_events[~all_off_events['event_id'].isin(used_off_ids)]
            if available_off.empty:
                final_unmatched_on.append(on_event)
                continue

            matched_off, tag, remainder, correction = find_partial_match(
                data, on_event, available_off,
                max_time_diff=6, max_magnitude_diff=350, logger=logger
            )

            if matched_off is not None:
                off_id = matched_off['event_id']
                used_off_ids.add(off_id)
                matches.append(_format_partial_match(on_event, matched_off, tag, on_event['phase'], correction))
                matched_event_ids.add(on_event['event_id'])
                matched_event_ids.add(off_id)

                # Collect remainder events for next iteration
                if remainder is not None:
                    if remainder['event'] == 'on':
                        remainder_on_events.append(remainder)
                    else:
                        remainder_off_events.append(remainder)
            else:
                final_unmatched_on.append(on_event)

        # Log all ON events that remain unmatched after all 3 stages
        for on_event in final_unmatched_on:
            logger.info(f"UNMATCHED {on_event['event_id']}({abs(on_event['magnitude']):.0f}W, "
                        f"phase={on_event['phase']}, start={on_event['start']})")

        # Add remainder events to unmatched lists
        unmatched_on = final_unmatched_on + remainder_on_events
        final_unmatched_off = all_off_events[~all_off_events['event_id'].isin(used_off_ids)].to_dict('records') + remainder_off_events

        if remainder_on_events or remainder_off_events:
            logger.info(f"  Created {len(remainder_on_events)} remainder ON + {len(remainder_off_events)} remainder OFF events")

        # Update on_off with matched status
        on_off_df['matched'] = on_off_df['event_id'].isin(matched_event_ids).astype(int)

        # Save results for this month
        if matches:
            matches_df = pd.DataFrame(matches)
            matches_df.to_pickle(matches_dir / f"matches_{house_id}_{month:02d}_{year}.pkl")
            total_matches += len(matches)

        if unmatched_on:
            unmatched_on_df = pd.DataFrame(unmatched_on)
            unmatched_on_df.to_pickle(unmatched_on_dir / f"unmatched_on_{house_id}_{month:02d}_{year}.pkl")
            total_unmatched_on += len(unmatched_on)

        if final_unmatched_off:
            unmatched_off_df = pd.DataFrame(final_unmatched_off)
            unmatched_off_df.to_pickle(unmatched_off_dir / f"unmatched_off_{house_id}_{month:02d}_{year}.pkl")
            total_unmatched_off += len(final_unmatched_off)

        # Save updated on_off
        on_off_df.to_pickle(on_off_file)

    logger.info(f"Total: {total_matches} matches, {total_unmatched_on} unmatched ON, {total_unmatched_off} unmatched OFF")
    logger.info(f"Matching completed for house {house_id}, run {run_number}")


def _format_match(on_event, off_event, tag, phase, correction=0):
    """Format a matched event pair for saving.

    Args:
        on_event: ON event dict
        off_event: OFF event dict or Series
        tag: Match tag (SPIKE, STABLE, CLOSE-MAG, NOISY, CORRECTED)
        phase: Phase name
        correction: Magnitude correction applied (0 if none)
    """
    duration = (off_event['end'] - on_event['start']).total_seconds() / 60

    # Apply correction to magnitude if needed
    on_magnitude = on_event['magnitude']
    off_magnitude = off_event['magnitude']
    if correction > 0:
        # Reduce both magnitudes by correction amount
        on_magnitude = abs(on_magnitude) - correction
        off_magnitude = -(abs(off_magnitude) - correction)  # OFF events are negative

    return {
        'on_event_id': on_event['event_id'],
        'off_event_id': off_event['event_id'],
        'on_start': on_event['start'],
        'on_end': on_event['end'],
        'off_start': off_event['start'],
        'off_end': off_event['end'],
        'duration': duration,
        'on_magnitude': on_magnitude,
        'off_magnitude': off_magnitude,
        'correction': correction,
        'tag': tag,
        'phase': phase
    }


def _format_partial_match(on_event, off_event, tag, phase, correction=0):
    """Format a partial match - uses min magnitude for the match.

    Args:
        on_event: ON event dict
        off_event: OFF event dict or Series
        tag: Match tag (PARTIAL, CORRECTED)
        phase: Phase name
        correction: Magnitude correction applied (0 if none)
    """
    on_mag = abs(on_event['magnitude'])
    off_mag = abs(off_event['magnitude'])
    match_magnitude = min(on_mag, off_mag) - correction  # Apply correction

    duration = (off_event['end'] - on_event['start']).total_seconds() / 60
    return {
        'on_event_id': on_event['event_id'],
        'off_event_id': off_event['event_id'],
        'on_start': on_event['start'],
        'on_end': on_event['end'],
        'off_start': off_event['start'],
        'off_end': off_event['end'],
        'duration': duration,
        'on_magnitude': match_magnitude,
        'off_magnitude': match_magnitude,
        'original_on_magnitude': on_mag,
        'original_off_magnitude': off_mag,
        'correction': correction,
        'tag': tag,
        'phase': phase
    }
