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
                                window_minutes=2, partial_factor=0.8, max_factor=1.3,
                                max_duration_minutes=3):
    """
    Smart gradual event detection:
    1. Find "partial" significant changes (80%+ of threshold but below threshold)
    2. Check ±2 minutes window to see if surrounding changes complete to full threshold
    3. Only merge if total is between 80%-130% (1280-2080W for 1600W threshold)
    4. Allow small opposite-direction changes (noise): max(50W, 3% of primary direction)
    5. Event duration (of significant changes only) cannot exceed 3 minutes

    Example:
        11:13 +1000W (80% of 1600) → check window
        11:14 +700W, -30W → total 1670W (within 1280-2080) → valid gradual event!
        (Opposite direction -30W < max(50, 1670*0.03=50) → allowed as noise)

    Args:
        data: DataFrame with timestamp and diff columns
        diff_col: Column with power differences
        threshold: Full threshold (e.g., 1600W)
        event_type: 'on' or 'off'
        window_minutes: Window to search before and after (±2 minutes)
        partial_factor: Minimum to be "interesting" (0.8 = 80%)
        max_factor: Maximum to avoid merging separate devices (1.3 = 130%)
        max_duration_minutes: Maximum duration for gradual event (3 minutes, counting only significant changes)

    Returns:
        DataFrame with gradual events: start, end, magnitude
    """
    if len(data) < 2:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    df = data[['timestamp', diff_col]].copy().sort_values('timestamp').reset_index(drop=True)
    df = df.dropna(subset=[diff_col])

    if len(df) < 2:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    partial_threshold = threshold * partial_factor  # 1280W
    max_threshold = threshold * max_factor  # 2080W

    timestamps = df['timestamp'].values
    diffs = df[diff_col].values

    # OPTIMIZATION: Pre-filter to "significant" changes (>= 50% of threshold)
    # This catches both partial events (80-100%) and multi-step gradual events
    # We search for changes >= 50% (800W for 1600W threshold) that might combine
    min_significant = threshold * 0.5  # 800W for 1600W threshold

    if event_type == 'on':
        candidate_mask = (diffs >= min_significant)
    else:  # off
        candidate_mask = (diffs <= -min_significant)

    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) == 0:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    events = []
    used_indices = set()

    # Only loop over candidate indices (much faster!)
    for i in candidate_indices:
        if i in used_indices:
            continue

        # Found a partial change! Look in ±window_minutes
        start_time = timestamps[i] - np.timedelta64(window_minutes, 'm')
        end_time = timestamps[i] + np.timedelta64(window_minutes, 'm')

        window_mask = (timestamps >= start_time) & (timestamps <= end_time)
        window_indices = np.where(window_mask)[0]

        if len(window_indices) < 2:
            continue

        # Filter: allow small opposite-direction changes (noise/fluctuations)
        # Strategy: include ALL changes, but later validate direction consistency
        window_diffs = diffs[window_indices]

        # First pass: calculate total sum including all directions
        window_sum = window_diffs.sum()

        # Determine primary direction and calculate noise threshold
        if event_type == 'on':
            # For ON events: expect mostly positive changes
            positive_sum = window_diffs[window_diffs > 0].sum()
            negative_sum = abs(window_diffs[window_diffs < 0].sum())
            primary_sum = positive_sum
            opposite_sum = negative_sum
        else:
            # For OFF events: expect mostly negative changes
            positive_sum = window_diffs[window_diffs > 0].sum()
            negative_sum = abs(window_diffs[window_diffs < 0].sum())
            primary_sum = negative_sum
            opposite_sum = positive_sum

        # Allow opposite-direction changes if they're small enough:
        # Either less than 50W OR less than 3% of primary direction
        # This handles noise and small fluctuations gracefully
        noise_threshold = max(50, abs(primary_sum) * 0.03)

        if opposite_sum > noise_threshold:
            # Too much opposite-direction change - not a clean gradual event
            continue

        # Use window_indices (all changes in time window)
        valid_indices = window_indices
        if len(valid_indices) == 0:
            continue

        # Check duration constraint: only count significant changes (not noise)
        # Find first and last timestamps with significant changes (>= 10% of min_significant)
        min_significant_for_duration = threshold * 0.05  # 5% of threshold (~80W for 1600W)
        significant_mask = np.abs(diffs[valid_indices]) >= min_significant_for_duration
        significant_indices = valid_indices[significant_mask]

        if len(significant_indices) == 0:
            # No significant changes found - skip
            continue

        event_start = timestamps[significant_indices[0]]
        event_end = timestamps[significant_indices[-1]]
        duration_minutes = (event_end - event_start) / np.timedelta64(1, 'm')

        if duration_minutes > max_duration_minutes:
            continue

        # Check if it meets threshold but not too much (between 80%-130%)
        # We want to catch gradual events that are slightly below threshold
        abs_sum = abs(window_sum)
        if partial_threshold <= abs_sum <= max_threshold:
            # Valid gradual event!
            events.append({
                'start': event_start,
                'end': event_end,
                'magnitude': window_sum
            })
            used_indices.update(valid_indices)

    if not events:
        return pd.DataFrame(columns=['start', 'end', 'magnitude'])

    # Remove any remaining duplicates
    result_df = pd.DataFrame(events).drop_duplicates().sort_values('start').reset_index(drop=True)

    # Final deduplication: remove overlapping events
    keep = []
    for i in range(len(result_df)):
        overlap = False
        for j in keep:
            if (result_df.loc[i, 'start'] <= result_df.loc[j, 'end'] and
                result_df.loc[i, 'end'] >= result_df.loc[j, 'start']):
                overlap = True
                break
        if not overlap:
            keep.append(i)

    return result_df.loc[keep].reset_index(drop=True) if keep else pd.DataFrame(columns=['start', 'end', 'magnitude'])

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
        logger.info(f"Using experiment config: off_factor={off_threshold_factor}, gradual={use_gradual}")
    else:
        off_threshold_factor = 0.8
        use_gradual = False
        gradual_window = 3

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

        # Add gradual event detection if enabled
        if use_gradual:
            logger.info(f"  Detecting gradual events for {phase}...")
            gradual_on = detect_smart_gradual_events(
                data, diff, threshold, event_type='on',
                window_minutes=gradual_window
            )
            gradual_off = detect_smart_gradual_events(
                data, diff, threshold, event_type='off',
                window_minutes=gradual_window
            )

            # Merge gradual events with sharp events
            if len(gradual_on) > 0:
                logger.info(f"    Found {len(gradual_on)} gradual ON events")
                results_on = pd.concat([results_on, gradual_on], ignore_index=True)
            if len(gradual_off) > 0:
                logger.info(f"    Found {len(gradual_off)} gradual OFF events")
                results_off = pd.concat([results_off, gradual_off], ignore_index=True)

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
