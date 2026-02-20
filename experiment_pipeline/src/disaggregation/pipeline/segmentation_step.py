"""
Segmentation pipeline step.

Separates power consumption into device-specific time series - processes month by month.
"""
import shutil
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
from typing import Optional, Dict, Any

import core
from core import setup_logging, load_power_data, find_house_data_path, find_previous_run_summarized, build_data_files_dict
from disaggregation.segmentation import process_phase_segmentation, summarize_segmentation, log_negative_values
from disaggregation.segmentation.restore import restore_skipped_to_unmatched


def process_segmentation(
    house_id: str,
    run_number: int,
    skip_large_file: bool = True,
    capture_device_profiles: bool = False,
    use_nan_imputation: bool = False,
    month_filter: str = None,
) -> Optional[Dict[str, Any]]:
    """
    Process segmentation for a house - processes month by month.

    Args:
        house_id: House identifier
        run_number: Current run number
        skip_large_file: If True, skip writing the large segmented_{id}.csv file
        capture_device_profiles: If True, capture per-device power profiles (default: False)
        month_filter: Optional month to process, e.g. '07_2021' (processes all if None)

    Returns:
        Dict mapping on_event_id -> {timestamps, values} if capture_device_profiles=True, else None
    """
    logger = setup_logging(house_id, run_number, core.LOGS_DIRECTORY)
    logger.info(f"Segmentation process for house {house_id}, run {run_number}")

    # Initialize device profiles collection
    all_device_profiles = {} if capture_device_profiles else None

    # Paths
    output_dir = f"{core.OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    matches_dir = Path(output_dir) / "matches"
    summarized_dir = Path(output_dir) / "summarized"

    os.makedirs(summarized_dir, exist_ok=True)

    # Find data path: run 0 reads raw data, run N reads remaining from summarized of run N-1
    data_path = None
    try:
        if run_number == 0:
            data_path = find_house_data_path(core.RAW_INPUT_DIRECTORY, house_id)
        else:
            data_path = find_previous_run_summarized(core.OUTPUT_BASE_PATH, house_id, run_number)
    except FileNotFoundError as e:
        if run_number == 0:
            logger.error(str(e))
            return
        else:
            logger.warning(f"Previous summarized not found: {e} — will copy from earlier run")

    # Get list of matches monthly files (may be empty)
    if matches_dir.is_dir():
        matches_files = sorted(matches_dir.glob(f"matches_{house_id}_*.pkl"))
        if month_filter:
            matches_files = [f for f in matches_files if f.stem.endswith(f"_{month_filter}")]
        if not matches_files:
            logger.info(f"No matches files found - creating passthrough summaries")
    else:
        matches_files = []
        logger.info(f"No matches folder — will create passthrough summaries")

    # Get list of data monthly files (handles both HouseholdData and summarized naming)
    if data_path is not None:
        data_path = Path(data_path)
        if data_path.is_dir():
            data_files = build_data_files_dict(data_path)
        else:
            data_files = {data_path.stem: data_path}
    else:
        data_files = {}

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
            # No events matched: remaining = original (nothing to extract)
            data = load_power_data(data_files[data_file_key])
            data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)
            if not data.empty:
                _create_passthrough_summary(data, phases, summary_file, logger, month, year)
            continue

        # Load power data for this month
        data = load_power_data(data_files[data_file_key])
        data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)

        if data.empty:
            continue

        # NaN imputation — ensure remaining power doesn't have NaN gaps
        if use_nan_imputation:
            from core.nan_imputation import impute_nan_gaps
            data = impute_nan_gaps(data, phase_cols=phases, logger=logger)

        all_new_columns = {}
        all_skipped_ids = []

        # Process each phase
        for phase in phases:
            data, new_cols, _, skipped_ids, profiles = process_phase_segmentation(
                data, events, phase, logger,
                capture_device_profiles=capture_device_profiles
            )
            all_new_columns.update(new_cols)
            all_skipped_ids.extend(skipped_ids)

            # Collect device profiles
            if capture_device_profiles and profiles:
                for profile in profiles:
                    all_device_profiles[profile['on_event_id']] = {
                        'timestamps': profile['timestamps'],
                        'values': profile['values']
                    }

        # Update matches file: remove events that were skipped during segmentation
        # Bug #14 fix: restore skipped events to unmatched files so no events are lost
        if all_skipped_ids:
            skipped_matches = events[events['on_event_id'].isin(all_skipped_ids)]
            restore_skipped_to_unmatched(skipped_matches, output_dir, house_id, month, year, logger)

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
        # Next iteration reads remaining_w1/w2/w3 directly from this file
        summary_file = summarized_dir / f"summarized_{house_id}_{month:02d}_{year}.pkl"
        summary_df.to_pickle(summary_file)

    # Create passthrough summaries for data months without matches files
    # (remaining = input, nothing was extracted)
    for data_key, data_file_path in data_files.items():
        parts = data_key.split('_')
        if len(parts) >= 3:
            try:
                month, year = int(parts[-2]), int(parts[-1])
            except ValueError:
                continue
        else:
            continue

        # Skip months not matching month_filter (for month-level parallelism)
        if month_filter and f"{month:02d}_{year}" != month_filter:
            continue

        summary_file = summarized_dir / f"summarized_{house_id}_{month:02d}_{year}.pkl"
        if summary_file.exists():
            continue

        data = load_power_data(data_file_path)
        data.dropna(subset=['w1', 'w2', 'w3'], how='all', inplace=True)
        if not data.empty:
            _create_passthrough_summary(data, phases, summary_file, logger, month, year)

    # Final safeguard: for run N>0, search backwards through ALL previous runs
    # and copy any missing summarized months.  This guarantees every month that
    # ever had data keeps a summarized file in every iteration — even when
    # several consecutive iterations find zero events for a particular month.
    if run_number > 0:
        for prev_run in range(run_number - 1, -1, -1):
            prev_summarized_dir = (
                Path(core.OUTPUT_BASE_PATH)
                / f"run_{prev_run}"
                / f"house_{house_id}"
                / "summarized"
            )
            if not prev_summarized_dir.is_dir():
                continue
            for prev_file in prev_summarized_dir.glob(f"summarized_{house_id}_*.pkl"):
                # Skip months not matching month_filter
                if month_filter and not prev_file.stem.endswith(f"_{month_filter}"):
                    continue
                target_file = summarized_dir / prev_file.name
                if not target_file.exists():
                    shutil.copy2(prev_file, target_file)
                    logger.info(
                        f"Copied {prev_file.name} from run_{prev_run} "
                        f"(no events found in current iteration)"
                    )

    logger.info(f"Segmentation completed for house {house_id}, run {run_number}")
    return all_device_profiles


def _create_passthrough_summary(data, phases, summary_file, logger, month, year):
    """Create summarized file where remaining = original (no events to extract)."""
    for phase in phases:
        data[f'remaining_power_{phase}'] = data[phase]
    summary_df = summarize_segmentation(data, phases)
    summary_df.to_pickle(summary_file)
    logger.info(f"Created passthrough summary for {month:02d}/{year} (no events to extract)")
