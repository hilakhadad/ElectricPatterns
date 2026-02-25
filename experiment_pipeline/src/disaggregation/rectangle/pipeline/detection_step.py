"""
Detection pipeline step.

Detects ON/OFF events from power consumption data.

Implementation is split across:
  - detection_step.py (this file) -- main process_detection() orchestration
  - detection_config.py -- config extraction/defaults logic
  - detection_postprocess.py -- post-processing: settling, near-threshold, tail extension
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

from pathlib import Path
import core
from core import setup_logging, DEFAULT_THRESHOLD, load_power_data, find_house_data_path, find_previous_run_summarized
from disaggregation.rectangle.detection import merge_overlapping_events, merge_consecutive_on_events, merge_consecutive_off_events, expand_event
from disaggregation.rectangle.detection.gradual import detect_gradual_events

from .detection_config import extract_detection_params, format_config_log
from .detection_postprocess import (
    apply_near_threshold, apply_tail_extension,
    apply_split_off_merger, apply_settling_extension,
)


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


def _safe_lookup(data_indexed, phase, timestamp):
    """Look up power value at timestamp, return NaN if not found."""
    try:
        return float(data_indexed.loc[timestamp, phase])
    except KeyError:
        return float('nan')


def _add_nearby_value(events, data_indexed, phase, event_type):
    """Add nearby_value column: power 1min before ON, power 1min after OFF."""
    if len(events) == 0:
        events['nearby_value'] = []
        return events
    if event_type == 'on':
        events['nearby_value'] = events['start'].apply(
            lambda ts: _safe_lookup(data_indexed, phase, ts - pd.Timedelta(minutes=1)))
    else:
        events['nearby_value'] = events['end'].apply(
            lambda ts: _safe_lookup(data_indexed, phase, ts + pd.Timedelta(minutes=1)))
    return events


def process_detection(house_id: str, run_number: int, threshold: int = DEFAULT_THRESHOLD, config=None, input_file: str = None, month_filter: str = None) -> None:
    """
    Detect ON/OFF events for a house - processes month by month.

    Args:
        house_id: House identifier
        run_number: Current run number
        threshold: Detection threshold in watts
        config: Optional ExperimentConfig with advanced parameters
        input_file: Optional specific file to process (processes all files if None)
        month_filter: Optional month to process, e.g. '07_2021' (processes all if None)
    """
    logger = setup_logging(house_id, run_number, core.LOGS_DIRECTORY)
    logger.info(f"Detection process for house {house_id}, run {run_number}, threshold {threshold}W")

    # Extract config parameters
    params = extract_detection_params(config)
    if config:
        logger.info(format_config_log(params))

    # Determine input path: run 0 reads raw data, run N reads remaining from summarized of run N-1
    try:
        if run_number < 1:
            data_path = find_house_data_path(core.RAW_INPUT_DIRECTORY, house_id)
        else:
            data_path = find_previous_run_summarized(core.OUTPUT_BASE_PATH, house_id, run_number)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Output directory
    output_dir = f"{core.OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    on_off_dir = f"{output_dir}/on_off"
    os.makedirs(on_off_dir, exist_ok=True)

    # Get list of monthly files (or single file)
    data_path = Path(data_path)
    if input_file:
        # Process only the specified file
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {input_file}")
            return
        monthly_files = [input_path]
    elif data_path.is_dir():
        monthly_files = sorted(data_path.glob("*.pkl"))
    else:
        monthly_files = [data_path]

    # Filter to a single month if requested (for month-level parallelism)
    if month_filter and not input_file:
        monthly_files = [f for f in monthly_files if f.stem.endswith(f"_{month_filter}")]

    phases = ['w1', 'w2', 'w3']
    partial_threshold = threshold
    total_events = 0
    event_counters = {(phase, event): 0 for phase in phases for event in ['on', 'off']}

    # Process each monthly file
    for monthly_file in tqdm(monthly_files, desc=f"Detection {house_id}", leave=False):
        # Check if output already exists (skip if so)
        if monthly_file.stem != house_id:
            parts = monthly_file.stem.split('_')
            if len(parts) >= 3:
                file_month, file_year = int(parts[-2]), int(parts[-1])
                output_file = Path(f"{on_off_dir}/on_off_{threshold}_{file_month:02d}_{file_year}.pkl")
                if output_file.exists():
                    logger.info(f"Skipping {file_month:02d}/{file_year} - on_off file already exists")
                    continue

        try:
            data = load_power_data(monthly_file)
        except Exception as e:
            logger.error(f"Failed to read {monthly_file}: {e}")
            continue

        data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)
        if data.empty:
            continue

        data = data.sort_values('timestamp').reset_index(drop=True)

        # NaN imputation -- fill short gaps to prevent false diff() jumps
        if params['use_nan_imputation']:
            from core.nan_imputation import impute_nan_gaps
            data = impute_nan_gaps(data, phase_cols=phases, logger=logger)

        data_indexed = data.set_index('timestamp')

        month_results = pd.DataFrame()

        for phase in phases:
            on_events, off_events = _detect_phase_events(
                data, data_indexed, phase, threshold, params, logger,
            )

            on_events['phase'] = phase
            on_events['event'] = 'on'
            off_events['phase'] = phase
            off_events['event'] = 'off'

            month_results = pd.concat([month_results, on_events, off_events], ignore_index=True)

        if month_results.empty:
            continue

        # Filter by minimum magnitude
        # For settling-extended events, use the original (pre-extension) magnitude
        # for the threshold check. The event was genuinely detected at this threshold
        # (its spike exceeded it), but extension corrected to steady-state value.
        if 'settling_original_magnitude' in month_results.columns:
            check_mag = abs(month_results['magnitude']).copy()
            has_original = month_results['settling_original_magnitude'].notna()
            check_mag[has_original] = abs(month_results.loc[has_original, 'settling_original_magnitude'])
            filtered = month_results[check_mag >= partial_threshold].copy()
        else:
            filtered = month_results[abs(month_results['magnitude']) >= partial_threshold].copy()
        if filtered.empty:
            continue

        filtered = filtered.sort_values(by='start')

        # Add event IDs with global counter
        event_ids = []
        for _, row in filtered.iterrows():
            key = (row['phase'], row['event'])
            event_counters[key] += 1
            event_ids.append(f"{row['event']}_it{run_number}_{row['phase']}_{event_counters[key]}")
        filtered['event_id'] = event_ids

        # Extract month/year from filename or data
        if monthly_file.stem != house_id:
            # Filename format: house_id_MM_YYYY.csv
            parts = monthly_file.stem.split('_')
            if len(parts) >= 3:
                month, year = int(parts[-2]), int(parts[-1])
            else:
                # Fall back to data
                month = filtered['start'].iloc[0].month
                year = filtered['start'].iloc[0].year
        else:
            month = filtered['start'].iloc[0].month
            year = filtered['start'].iloc[0].year

        # Save this month's results
        output_file = f"{on_off_dir}/on_off_{threshold}_{month:02d}_{year}.pkl"
        filtered.to_pickle(output_file)
        total_events += len(filtered)

    logger.info(f"Saved {total_events} events to {on_off_dir}/ ({len(monthly_files)} monthly files)")
    logger.info(f"Detection completed for house {house_id}, run {run_number}")


def _detect_phase_events(data, data_indexed, phase, threshold, params, logger):
    """Detect ON and OFF events for a single phase.

    Args:
        data: DataFrame with timestamp column (for boolean indexing)
        data_indexed: Same DataFrame but with timestamp as index (for fast lookups)
        phase: Phase column name (w1, w2, w3)
        threshold: Detection threshold in watts
        params: Detection parameters dict from extract_detection_params()
        logger: Logger instance
    """
    off_threshold_factor = params['off_threshold_factor']
    use_gradual = params['use_gradual']
    gradual_window = params['gradual_window']
    progressive_search = params['progressive_search']

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

    # Merge consecutive ON events (appliances turning on in stages, e.g., AC compressor)
    if len(results_on) > 1:
        before_merge = len(results_on)
        results_on = merge_consecutive_on_events(
            results_on, results_off, max_gap_minutes=2, data=data_indexed, phase=phase, logger=logger
        )
        if len(results_on) < before_merge:
            logger.info(f"  Merged {before_merge - len(results_on)} consecutive ON events for {phase}")

    # Merge consecutive OFF events (appliances turning off in stages)
    if len(results_off) > 1:
        before_merge = len(results_off)
        results_off = merge_consecutive_off_events(
            results_off, results_on, max_gap_minutes=2, data=data_indexed, phase=phase, logger=logger
        )
        if len(results_off) < before_merge:
            logger.info(f"  Merged {before_merge - len(results_off)} consecutive OFF events for {phase}")

    # Add gradual detection
    if use_gradual:
        logger.info(f"  Detecting gradual events for {phase}...")
        gradual_on = detect_gradual_events(
            data, diff_col, threshold, event_type='on',
            window_minutes=gradual_window, progressive_search=progressive_search,
            partial_factor=1.0, logger=logger
        )
        gradual_off = detect_gradual_events(
            data, diff_col, threshold, event_type='off',
            window_minutes=gradual_window, progressive_search=progressive_search,
            partial_factor=1.0, logger=logger
        )

        if len(gradual_on) > 0:
            logger.info(f"    Found {len(gradual_on)} gradual ON events")
            results_on = pd.concat([results_on, gradual_on], ignore_index=True)
        if len(gradual_off) > 0:
            logger.info(f"    Found {len(gradual_off)} gradual OFF events")
            results_off = pd.concat([results_off, gradual_off], ignore_index=True)

        # Merge overlapping (pass indexed data for magnitude recalculation)
        before_on, before_off = len(results_on), len(results_off)
        results_on = merge_overlapping_events(results_on, max_gap_minutes=0, data=data_indexed, phase=phase, logger=logger)
        results_off = merge_overlapping_events(results_off, max_gap_minutes=0, data=data_indexed, phase=phase, logger=logger)
        if len(results_on) < before_on:
            logger.info(f"    Merged {before_on - len(results_on)} overlapping ON events")
        if len(results_off) < before_off:
            logger.info(f"    Merged {before_off - len(results_off)} overlapping OFF events")

    # Post-processing steps (near-threshold, tail extension, split-off, settling)
    if params['use_near_threshold']:
        results_on, results_off = apply_near_threshold(
            results_on, results_off, data, data_indexed, diff_col,
            threshold, off_threshold, phase, logger,
            min_factor=params['near_threshold_min_factor'],
            max_extend_minutes=params['near_threshold_max_extend'],
        )

    if params['use_tail_extension']:
        results_off = apply_tail_extension(
            results_off, data_indexed, phase, logger,
            max_minutes=params['tail_max_minutes'],
            min_residual=params['tail_min_residual'],
            noise_tolerance=params['tail_noise_tolerance'],
            min_gain=params['tail_min_gain'],
            min_residual_fraction=params['tail_min_residual_fraction'],
        )

    if params['use_split_off_merger'] and len(results_off) > 1:
        results_off = apply_split_off_merger(
            results_off, results_on, data_indexed, phase, logger,
            max_gap_minutes=params['split_off_max_gap_minutes'],
        )

    if params['use_settling_extension']:
        results_on, results_off = apply_settling_extension(
            results_on, results_off, data_indexed, phase, logger,
            settling_factor=params['settling_factor'],
            max_settling_minutes=params['settling_max_minutes'],
        )

    # Add duration
    results_on['duration'] = (results_on['end'] - results_on['start']).dt.total_seconds() / 60
    results_off['duration'] = (results_off['end'] - results_off['start']).dt.total_seconds() / 60

    # Add nearby_value: power 1min before ON, power 1min after OFF
    results_on = _add_nearby_value(results_on, data_indexed, phase, 'on')
    results_off = _add_nearby_value(results_off, data_indexed, phase, 'off')

    return results_on, results_off
