from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
from data_util import *


def merge_overlapping_events(events_df, max_gap_minutes=0):
    """
    Merge events that overlap or touch each other.

    Only merges if:
    - Events overlap (gap < 0)
    - Events touch (next starts exactly when current ends, gap = 0)

    Does NOT merge events with a gap between them (e.g., one ends at 22:05, next starts at 22:06).

    Args:
        events_df: DataFrame with 'start', 'end', 'magnitude' columns
        max_gap_minutes: Maximum gap between events to still merge them (default 0 = only overlapping/touching)

    Returns:
        DataFrame with merged events
    """
    if len(events_df) <= 1:
        return events_df

    # Sort by start time
    df = events_df.sort_values('start').reset_index(drop=True)

    merged = []
    current = df.iloc[0].copy()

    for i in range(1, len(df)):
        next_event = df.iloc[i]

        # Check if events overlap or touch
        # gap < 0: overlap, gap = 0: touch, gap > 0: separate events
        gap = (next_event['start'] - current['end']).total_seconds() / 60

        if gap <= max_gap_minutes:
            # Merge: extend end time and add magnitudes
            current['end'] = max(current['end'], next_event['end'])
            current['magnitude'] = current['magnitude'] + next_event['magnitude']
        else:
            # No overlap - save current and start new
            merged.append(current)
            current = next_event.copy()

    # Don't forget the last event
    merged.append(current)

    return pd.DataFrame(merged).reset_index(drop=True)


def expand_event(event, data, event_type, diff):
    start, end, magnitude = event['start'], event['end'], event['magnitude']

    before_start = start - pd.Timedelta(minutes=1)
    after_end = end + pd.Timedelta(minutes=1)

    before_data = data[(data['timestamp'] >= before_start) & (data['timestamp'] < start)]
    after_data = data[(data['timestamp'] > end) & (data['timestamp'] <= after_end)]

    before_magnitude = before_data[diff].sum()
    after_magnitude = after_data[diff].sum()

    # Use absolute value for threshold calculation to work correctly with both on/off
    threshold_factor = 0.05 * abs(magnitude)

    if event_type == 'on':
        # For ON events: look for positive changes
        if before_magnitude > threshold_factor:
            start = before_start
            magnitude += before_magnitude
        if after_magnitude > threshold_factor:
            end = after_end
            magnitude += after_magnitude

    elif event_type == 'off':
        # For OFF events: look for negative changes (threshold_factor is positive)
        if before_magnitude < -threshold_factor:
            start = before_start
            magnitude += before_magnitude
        if after_magnitude < -threshold_factor:
            end = after_end
            magnitude += after_magnitude

    return pd.Series({'start': start, 'end': end, 'magnitude': magnitude})


def detect_smart_gradual_events(data, diff_col, threshold, event_type='on',
                                window_minutes=3, partial_factor=0.8, max_factor=1.3,
                                max_duration_minutes=3, progressive_search=False):
    """
    Smart gradual event detection with optional progressive window search.

    When progressive_search=True:
    - First try ±1 minute window
    - If no valid event found, try ±2 minutes
    - If still no valid event found, try ±3 minutes (up to window_minutes)
    - Stop as soon as a valid event is found (prefer shorter windows)

    This helps avoid including OFF events that happen right after ON events
    when using large windows.

    Standard behavior (progressive_search=False):
    1. Find "partial" significant changes (80%+ of threshold but below threshold)
    2. Check ±window_minutes to see if surrounding changes complete to full threshold
    3. Only merge if total is between 80%-130% (1280-2080W for 1600W threshold)
    4. Allow small opposite-direction changes (noise): max(50W, 3% of primary direction)
    5. Event duration (of significant changes only) cannot exceed 3 minutes

    Args:
        data: DataFrame with timestamp and diff columns
        diff_col: Column with power differences
        threshold: Full threshold (e.g., 1600W)
        event_type: 'on' or 'off'
        window_minutes: Maximum window to search (±N minutes)
        partial_factor: Minimum to be "interesting" (0.8 = 80%)
        max_factor: Maximum to avoid merging separate devices (1.3 = 130%)
        max_duration_minutes: Maximum duration for gradual event
        progressive_search: If True, try smaller windows first (1, 2, 3...)

    Returns:
        DataFrame with gradual events: start, end, magnitude
    """
    if len(data) < 2:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    df = data[['timestamp', diff_col]].copy().sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=[diff_col])

    if len(df) < 2:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    partial_threshold = threshold * partial_factor  # 1200W for 1500W
    max_threshold = threshold * max_factor  # 1950W for 1500W

    timestamps = df['timestamp'].values
    diffs = df[diff_col].values

    # OPTIMIZATION: Pre-filter to "significant" changes (>= 50% of threshold)
    min_significant = threshold * 0.5  # 750W for 1500W threshold

    if event_type == 'on':
        candidate_mask = (diffs >= min_significant)
    else:  # off
        candidate_mask = (diffs <= -min_significant)

    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    events = []
    used_indices = set()

    # Determine which windows to try
    if progressive_search:
        windows_to_try = list(range(1, window_minutes + 1))  # [1, 2, 3] for window_minutes=3
    else:
        windows_to_try = [window_minutes]  # Just the max window

    # Only loop over candidate indices
    for i in candidate_indices:
        if i in used_indices:
            continue

        # Try progressively larger windows until we find a valid event
        event_found = False
        for current_window in windows_to_try:
            if event_found:
                break

            # Try symmetric window first, then asymmetric windows if needed
            # window_configs: (before_minutes, after_minutes)
            window_configs = [
                (current_window, current_window),  # Symmetric: ±N minutes
                (current_window, 0),               # Only before: -N to 0 minutes
                (0, current_window),               # Only after: 0 to +N minutes
            ]

            for before_mins, after_mins in window_configs:
                if event_found:
                    break

                start_time = timestamps[i] - np.timedelta64(before_mins, 'm')
                end_time = timestamps[i] + np.timedelta64(after_mins, 'm')

                window_mask = (timestamps >= start_time) & (timestamps <= end_time)
                window_indices = np.where(window_mask)[0]

                if len(window_indices) < 1:
                    continue

                # Filter to only adjacent indices that are NOT already used
                # This prevents the same timestamps from being included in multiple events
                adjacent_indices = []
                for idx in sorted(window_indices):
                    if idx in used_indices:
                        # Skip indices that are already part of another event
                        continue
                    if not adjacent_indices:
                        # First index in the sequence
                        adjacent_indices.append(idx)
                    else:
                        # Check if this index is within 2 minutes of the last added index
                        time_diff = (timestamps[idx] - timestamps[adjacent_indices[-1]]) / np.timedelta64(1, 'm')
                        if time_diff <= 2:
                            adjacent_indices.append(idx)
                        else:
                            # Gap too large - stop here and use what we have
                            break

                if len(adjacent_indices) < 1:
                    continue

                adjacent_indices = np.array(adjacent_indices)
                window_diffs = diffs[adjacent_indices]
                window_sum = window_diffs.sum()

                # Determine primary direction and calculate noise threshold
                if event_type == 'on':
                    positive_sum = window_diffs[window_diffs > 0].sum()
                    negative_sum = abs(window_diffs[window_diffs < 0].sum())
                    primary_sum = positive_sum
                    opposite_sum = negative_sum
                else:
                    positive_sum = window_diffs[window_diffs > 0].sum()
                    negative_sum = abs(window_diffs[window_diffs < 0].sum())
                    primary_sum = negative_sum
                    opposite_sum = positive_sum

                # Allow opposite-direction changes if they're small enough
                noise_threshold = max(50, abs(primary_sum) * 0.03)

                if opposite_sum > noise_threshold:
                    # Too much opposite-direction change - try next window config
                    continue

                valid_indices = adjacent_indices
                if len(valid_indices) == 0:
                    continue

                # Check duration constraint
                min_significant_for_duration = threshold * 0.05
                significant_mask = np.abs(diffs[valid_indices]) >= min_significant_for_duration
                significant_indices = valid_indices[significant_mask]

                if len(significant_indices) == 0:
                    continue

                event_start = timestamps[significant_indices[0]]
                event_end = timestamps[significant_indices[-1]]
                # Duration = number of data points, not time difference
                # If we have points at 22:06 and 22:07, that's 2 minutes of data
                duration_minutes = len(significant_indices)

                if duration_minutes > max_duration_minutes:
                    continue

                # Check if it meets threshold but not too much (between 80%-130%)
                abs_sum = abs(window_sum)
                if partial_threshold <= abs_sum <= max_threshold:
                    # Valid gradual event found!
                    events.append({
                        'start': event_start,
                        'end': event_end,
                        'magnitude': window_sum
                    })
                    used_indices.update(valid_indices)
                    event_found = True  # Stop trying other window configs

    if not events:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    # Convert to DataFrame
    result_df = pd.DataFrame(events).sort_values('start').reset_index(drop=True)

    # Remove events with same start and end (keep only first occurrence)
    result_df = result_df.drop_duplicates(subset=['start', 'end'], keep='first').reset_index(drop=True)

    return result_df

def process_house(house_id, run_number, threshold=DEFAULT_THRESHOLD, config=None):
    """
    Process on/off events for a given house and run number.

    Args:
        house_id: House identifier
        run_number: Run number
        threshold: Detection threshold (default 1600W)
        config: Optional ExperimentConfig object with advanced parameters
    """
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Detection On and Off events process for house {house_id} for run {run_number} with threshold {threshold}.")

    # Extract config parameters if provided
    if config:
        off_threshold_factor = config.off_threshold_factor
        use_gradual = config.use_gradual_detection
        gradual_window = config.gradual_window_minutes
        progressive_search = getattr(config, 'progressive_window_search', False)
        logger.info(f"Using experiment config: off_factor={off_threshold_factor}, gradual={use_gradual}, progressive={progressive_search}")
    else:
        off_threshold_factor = 0.8
        use_gradual = False
        gradual_window = 3
        progressive_search = False

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

        # Detect on/off events with DIFFERENT thresholds
        # ON events: use full threshold
        # OFF events: use configurable threshold (default 80%)
        off_threshold = int(threshold * off_threshold_factor)

        data[on_col] = np.where(data[diff] >= threshold, data[diff], 0)
        data[off_col] = np.where(data[diff] <= -off_threshold, data[diff], 0)

        is_on = f'{on_col}_magnitude'
        is_off = f'{off_col}_magnitude'

        data[is_on] = (data[on_col] == 0).cumsum() * (data[on_col] != 0)
        data[is_off] = (data[off_col] == 0).cumsum() * (data[off_col] != 0)

        results_on = (
            data[data[on_col] != 0]
                .groupby(is_on)
                .agg(start=('timestamp', 'min'),
                     end=('timestamp', 'max'),
                     magnitude=(on_col, 'sum'))
                .reset_index(drop=True)
        )

        results_off = (
            data[data[off_col] != 0]
                .groupby(is_off)
                .agg(start=('timestamp', 'min'),
                     end=('timestamp', 'max'),
                     magnitude=(off_col, 'sum'))
                .reset_index(drop=True)
        )

        results_on = results_on.apply(lambda x: expand_event(x, data, 'on', diff), axis=1)
        results_off = results_off.apply(lambda x: expand_event(x, data, 'off', diff), axis=1)

        # Add gradual event detection if enabled
        if use_gradual:
            logger.info(f"  Detecting gradual events for {phase}...")
            gradual_on = detect_smart_gradual_events(
                data, diff, threshold, event_type='on',
                window_minutes=gradual_window,
                progressive_search=progressive_search
            )
            gradual_off = detect_smart_gradual_events(
                data, diff, threshold, event_type='off',
                window_minutes=gradual_window,
                progressive_search=progressive_search
            )

            # Merge gradual events with sharp events
            if len(gradual_on) > 0:
                logger.info(f"    Found {len(gradual_on)} gradual ON events")
                results_on = pd.concat([results_on, gradual_on], ignore_index=True)
            if len(gradual_off) > 0:
                logger.info(f"    Found {len(gradual_off)} gradual OFF events")
                results_off = pd.concat([results_off, gradual_off], ignore_index=True)

            # Merge overlapping or adjacent events (e.g., sharp + gradual that form one device activation)
            before_merge_on = len(results_on)
            before_merge_off = len(results_off)
            results_on = merge_overlapping_events(results_on, max_gap_minutes=0)
            results_off = merge_overlapping_events(results_off, max_gap_minutes=0)
            if len(results_on) < before_merge_on:
                logger.info(f"    Merged {before_merge_on - len(results_on)} overlapping ON events")
            if len(results_off) < before_merge_off:
                logger.info(f"    Merged {before_merge_off - len(results_off)} overlapping OFF events")

        # Duration = number of data points, not time difference
        # start=22:05, end=22:05 -> 1 minute (1 data point)
        # start=22:05, end=22:07 -> 3 minutes (3 data points)
        results_on['duration'] = ((results_on['end'] - results_on['start']).dt.total_seconds() / 60)
        results_on['phase'] = phase
        results_on['event'] = 'on'

        results_off['duration'] = ((results_off['end'] - results_off['start']).dt.total_seconds() / 60)
        results_off['phase'] = phase
        results_off['event'] = 'off'

        on_events = pd.DataFrame(results_on)
        off_events = pd.DataFrame(results_off)
        res = pd.concat([res, on_events, off_events], ignore_index=True)

    res = res.sort_values(by='start')

    output_directory = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    os.makedirs(output_directory, exist_ok=True)

    output_path = f"{output_directory}/on_off_{threshold}.csv"

    # Filter events: keep those with magnitude >= 80% of threshold (for gradual events)
    # This allows sub-threshold gradual events (1280-1600W) to be included
    partial_threshold = threshold * 0.8
    mid_res = res[abs(res['magnitude']) >= partial_threshold].copy()

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
