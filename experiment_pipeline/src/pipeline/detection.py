"""
Detection pipeline step.

Detects ON/OFF events from power consumption data.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from core import (
    setup_logging, RAW_INPUT_DIRECTORY, INPUT_DIRECTORY,
    OUTPUT_BASE_PATH, DEFAULT_THRESHOLD, LOGS_DIRECTORY
)
from detection import merge_overlapping_events, expand_event
from detection.gradual import detect_gradual_events


def _calc_magnitude(df: pd.DataFrame, phase: str, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """
    Calculate event magnitude as the difference between power at end and power before start.
    """
    # Get value at end of event
    end_row = df[df['timestamp'] == end]
    value_end = end_row[phase].values[0] if len(end_row) > 0 else 0

    # Get value before start of event (1 minute before)
    before_start = start - pd.Timedelta(minutes=1)
    before_row = df[df['timestamp'] == before_start]
    value_before = before_row[phase].values[0] if len(before_row) > 0 else 0

    return value_end - value_before


def process_detection(house_id: str, run_number: int, threshold: int = DEFAULT_THRESHOLD, config=None) -> None:
    """
    Detect ON/OFF events for a house.

    Args:
        house_id: House identifier
        run_number: Current run number
        threshold: Detection threshold in watts
        config: Optional ExperimentConfig with advanced parameters
    """
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Detection process for house {house_id}, run {run_number}, threshold {threshold}W")

    # Extract config parameters
    if config:
        off_threshold_factor = config.off_threshold_factor
        use_gradual = config.use_gradual_detection
        gradual_window = config.gradual_window_minutes
        progressive_search = getattr(config, 'progressive_window_search', False)
        logger.info(f"Config: off_factor={off_threshold_factor}, gradual={use_gradual}, progressive={progressive_search}")
    else:
        off_threshold_factor = 0.8
        use_gradual = False
        gradual_window = 3
        progressive_search = False

    # Determine input path
    input_dir = RAW_INPUT_DIRECTORY if run_number < 1 else f"{INPUT_DIRECTORY}/run_{run_number}/HouseholdData"
    file_path = f"{input_dir}/{house_id}.csv"

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return

    # Load data
    try:
        data = pd.read_csv(file_path, parse_dates=['timestamp'])
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return

    data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'}, inplace=True)
    data.drop(columns=['sum'], inplace=True, errors='ignore')
    data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)

    # Sort by timestamp for efficient processing
    data = data.sort_values('timestamp').reset_index(drop=True)

    # Create indexed version for fast lookups
    data_indexed = data.set_index('timestamp')

    phases = ['w1', 'w2', 'w3']
    results = pd.DataFrame()

    for phase in tqdm(phases, desc=f"Detecting events for {house_id}"):
        on_events, off_events = _detect_phase_events(
            data, data_indexed, phase, threshold, off_threshold_factor,
            use_gradual, gradual_window, progressive_search, logger
        )

        on_events['phase'] = phase
        on_events['event'] = 'on'
        off_events['phase'] = phase
        off_events['event'] = 'off'

        results = pd.concat([results, on_events, off_events], ignore_index=True)

    results = results.sort_values(by='start')

    # Save results
    output_dir = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/on_off_{threshold}.csv"

    # Filter by minimum magnitude
    partial_threshold = threshold * 0.8
    filtered = results[abs(results['magnitude']) >= partial_threshold].copy()

    # Add event IDs
    filtered['event_counter'] = filtered.groupby(['phase', 'event']).cumcount() + 1
    filtered['event_id'] = filtered.apply(
        lambda row: f"{row['event']}_{row['phase']}_{row['event_counter']}", axis=1
    )
    filtered.drop(columns=['event_counter'], inplace=True)
    filtered = filtered.sort_values(by='start')

    try:
        filtered.to_csv(output_path, index=False)
        logger.info(f"Saved {len(filtered)} events to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save: {e}")

    logger.info(f"Detection completed for house {house_id}, run {run_number}")


def _detect_phase_events(data, data_indexed, phase, threshold, off_threshold_factor,
                         use_gradual, gradual_window, progressive_search, logger):
    """Detect ON and OFF events for a single phase.

    Args:
        data: DataFrame with timestamp column (for boolean indexing)
        data_indexed: Same DataFrame but with timestamp as index (for fast lookups)
    """
    diff_col = f'{phase}_diff'
    on_col = f"{phase}_{threshold}_on"
    off_col = f"{phase}_{threshold}_off"

    data[diff_col] = data[phase].diff()

    off_threshold = int(threshold * off_threshold_factor)
    data[on_col] = np.where(data[diff_col] >= threshold, data[diff_col], 0)
    data[off_col] = np.where(data[diff_col] <= -off_threshold, data[diff_col], 0)

    # Update indexed version with new columns
    data_indexed[diff_col] = data.set_index('timestamp')[diff_col]

    # Group consecutive events
    is_on = f'{on_col}_magnitude'
    is_off = f'{off_col}_magnitude'
    data[is_on] = (data[on_col] == 0).cumsum() * (data[on_col] != 0)
    data[is_off] = (data[off_col] == 0).cumsum() * (data[off_col] != 0)

    # Aggregate events
    results_on = (
        data[data[on_col] != 0]
        .groupby(is_on)
        .agg(start=('timestamp', 'min'), end=('timestamp', 'max'))
        .reset_index(drop=True)
    )
    results_off = (
        data[data[off_col] != 0]
        .groupby(is_off)
        .agg(start=('timestamp', 'min'), end=('timestamp', 'max'))
        .reset_index(drop=True)
    )

    # Add placeholder magnitude for expand_event (it recalculates from phase values)
    if len(results_on) > 0:
        results_on['magnitude'] = 0
    if len(results_off) > 0:
        results_off['magnitude'] = 0

    # Expand events - use indexed data for fast lookups
    results_on = results_on.apply(lambda x: expand_event(x, data_indexed, 'on', diff_col), axis=1)
    results_off = results_off.apply(lambda x: expand_event(x, data_indexed, 'off', diff_col), axis=1)

    # Add gradual detection
    if use_gradual:
        logger.info(f"  Detecting gradual events for {phase}...")
        gradual_on = detect_gradual_events(
            data, diff_col, threshold, event_type='on',
            window_minutes=gradual_window, progressive_search=progressive_search
        )
        gradual_off = detect_gradual_events(
            data, diff_col, threshold, event_type='off',
            window_minutes=gradual_window, progressive_search=progressive_search
        )

        if len(gradual_on) > 0:
            logger.info(f"    Found {len(gradual_on)} gradual ON events")
            results_on = pd.concat([results_on, gradual_on], ignore_index=True)
        if len(gradual_off) > 0:
            logger.info(f"    Found {len(gradual_off)} gradual OFF events")
            results_off = pd.concat([results_off, gradual_off], ignore_index=True)

        # Merge overlapping (pass indexed data for magnitude recalculation)
        before_on, before_off = len(results_on), len(results_off)
        results_on = merge_overlapping_events(results_on, max_gap_minutes=0, data=data_indexed, phase=phase)
        results_off = merge_overlapping_events(results_off, max_gap_minutes=0, data=data_indexed, phase=phase)
        if len(results_on) < before_on:
            logger.info(f"    Merged {before_on - len(results_on)} overlapping ON events")
        if len(results_off) < before_off:
            logger.info(f"    Merged {before_off - len(results_off)} overlapping OFF events")

    # Add duration
    results_on['duration'] = (results_on['end'] - results_on['start']).dt.total_seconds() / 60
    results_off['duration'] = (results_off['end'] - results_off['start']).dt.total_seconds() / 60

    return results_on, results_off
