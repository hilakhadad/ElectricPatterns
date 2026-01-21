"""
Evaluation module for segmentation results.
Calculates power and time metrics comparing original vs remaining consumption.
"""
import pandas as pd
import os
from data_util import setup_logging, OUTPUT_BASE_PATH, LOGS_DIRECTORY
from pathlib import Path


def evaluate_segmentation(house_id, run_number, threshold):
    """
    Evaluate segmentation results by calculating power and time metrics.

    Args:
        house_id: House identifier
        run_number: Current iteration number
        threshold: Power threshold in watts

    Returns:
        dict: Evaluation results for each phase
    """
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Evaluation process for house {house_id} in run {run_number}.")

    # Paths
    output_directory = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    current_summarized_path = f"{output_directory}/summarized_{house_id}.csv"

    # For run_0, the baseline is the same file. For run > 0, we need run_0 as baseline
    baseline_directory = f"{OUTPUT_BASE_PATH}/run_0/house_{house_id}"
    baseline_summarized_path = f"{baseline_directory}/summarized_{house_id}.csv"

    # Previous run for calculating iteration-specific metrics
    if run_number > 0:
        prev_directory = f"{OUTPUT_BASE_PATH}/run_{run_number - 1}/house_{house_id}"
        prev_summarized_path = f"{prev_directory}/summarized_{house_id}.csv"
        prev_eval_path = f"{prev_directory}/evaluation_history_{house_id}.csv"

    historical_eval_output_path = f"{output_directory}/evaluation_history_{house_id}.csv"

    # Load current summarized data
    logger.info(f"Loading summarized data from {current_summarized_path}.")
    if not os.path.exists(current_summarized_path):
        logger.error(f"Summarized data file not found: {current_summarized_path}")
        return None

    try:
        current_data = pd.read_csv(current_summarized_path, parse_dates=['timestamp'])
        logger.info(f"Successfully loaded current summarized data ({len(current_data)} rows).")

        # Load baseline (run_0) for calculating cumulative metrics
        if run_number == 0:
            baseline_data = current_data
        else:
            baseline_data = pd.read_csv(baseline_summarized_path, parse_dates=['timestamp'])
            logger.info(f"Loaded baseline data from run_0 ({len(baseline_data)} rows).")

        # Load previous run data for iteration-specific metrics
        prev_data = None
        prev_eval = None
        if run_number > 0:
            if os.path.exists(prev_summarized_path):
                prev_data = pd.read_csv(prev_summarized_path, parse_dates=['timestamp'])
                logger.info(f"Loaded previous run data ({len(prev_data)} rows).")
            if os.path.exists(prev_eval_path):
                prev_eval = pd.read_csv(prev_eval_path)
                logger.info(f"Loaded previous evaluation history.")

    except Exception as e:
        logger.error(f"Failed to read data for house {house_id}: {e}")
        return None

    phases = ['w1', 'w2', 'w3']
    results_list = []

    for phase in phases:
        logger.info(f"Processing phase {phase}.")

        # Get columns
        original_col = f'original_{phase}'
        remaining_col = f'remaining_{phase}'

        # Skip if columns don't exist or are all NaN
        if original_col not in baseline_data.columns:
            logger.warning(f"Column {original_col} not found, skipping phase {phase}.")
            continue

        baseline_original = baseline_data[original_col]
        current_remaining = current_data[remaining_col]

        # Create mask for valid data - rows that exist in BOTH baseline and current
        # NaN values should NOT be counted as explained
        valid_mask = baseline_original.notna() & current_remaining.notna()

        # Mask for minutes above threshold in baseline (run_0) - only where data is valid
        above_th_mask = (baseline_original > threshold) & valid_mask

        # === POWER METRICS ===
        # Total power (all minutes) - for display (use fillna for sum)
        total_power_all = baseline_original.fillna(0).sum()

        # Total power (only minutes above TH) - for calculations
        total_power_above_th = baseline_original[above_th_mask].sum()

        # Explained power cumulative (baseline - current remaining, only for minutes above TH)
        # Only calculated where both values exist (above_th_mask already includes valid_mask)
        explained_power_cumulative = (baseline_original[above_th_mask] - current_remaining[above_th_mask]).clip(lower=0).sum()

        # Explained power percentage
        if total_power_above_th > 0:
            explained_power_cumulative_pct = (explained_power_cumulative / total_power_above_th) * 100
        else:
            explained_power_cumulative_pct = 0.0

        # Iteration-specific power metrics
        if run_number == 0:
            explained_power_iteration = explained_power_cumulative
            explained_power_iteration_pct = explained_power_cumulative_pct
        else:
            prev_remaining = prev_data[remaining_col]
            # Create valid mask for iteration comparison (prev and current both exist)
            valid_iter_mask = above_th_mask & prev_remaining.notna() & current_remaining.notna()
            explained_power_iteration = (prev_remaining[valid_iter_mask] - current_remaining[valid_iter_mask]).clip(lower=0).sum()
            if total_power_above_th > 0:
                explained_power_iteration_pct = (explained_power_iteration / total_power_above_th) * 100
            else:
                explained_power_iteration_pct = 0.0

        # === NEGATIVE VALUES (sanity check) ===
        # Negative remaining values are physically impossible - indicates calculation errors
        negative_mask = current_remaining < 0
        minutes_negative = negative_mask.sum()
        power_negative = current_remaining[negative_mask].sum()  # Will be negative

        if minutes_negative > 0:
            logger.warning(f"Phase {phase} - NEGATIVE VALUES DETECTED: {minutes_negative} minutes, {power_negative:.0f}W total")
            # Save negative values to experiment-level CSV
            _save_negative_values(current_data, current_remaining, negative_mask, house_id, run_number, phase)

        # === MISSING VALUES (rows that disappeared) ===
        # Count rows that exist in baseline but are NaN in current
        minutes_missing = (baseline_original.notna() & current_remaining.isna()).sum()
        if minutes_missing > 0:
            logger.warning(f"Phase {phase} - MISSING VALUES: {minutes_missing} rows exist in baseline but NaN in current run")

        # === TIME METRICS ===
        # Minutes above threshold in baseline (only where data is valid)
        minutes_above_th = above_th_mask.sum()

        # Minutes now below threshold (cumulative - from baseline to current)
        # Only count where both baseline and current have valid data
        minutes_below_th_cumulative = (above_th_mask & (current_remaining <= threshold)).sum()

        # Time percentage cumulative
        if minutes_above_th > 0:
            time_pct_cumulative = (minutes_below_th_cumulative / minutes_above_th) * 100
        else:
            time_pct_cumulative = 0.0

        # Iteration-specific time metrics
        if run_number == 0:
            minutes_below_th_iteration = minutes_below_th_cumulative
            time_pct_iteration = time_pct_cumulative
        else:
            # Minutes that were above TH in prev run but now below TH
            prev_remaining = prev_data[remaining_col]
            # Only count where all three (baseline, prev, current) have valid data
            valid_time_iter_mask = above_th_mask & prev_remaining.notna() & current_remaining.notna()
            minutes_below_th_iteration = (valid_time_iter_mask & (prev_remaining > threshold) & (current_remaining <= threshold)).sum()
            if minutes_above_th > 0:
                time_pct_iteration = (minutes_below_th_iteration / minutes_above_th) * 100
            else:
                time_pct_iteration = 0.0

        # Build result row
        result = {
            'house_id': house_id,
            'run_number': run_number,
            'threshold': threshold,
            'phase': phase,
            # Power metrics
            'total_power_all': total_power_all,
            'total_power_above_th': total_power_above_th,
            'explained_power': explained_power_iteration,
            'explained_power_pct': round(explained_power_iteration_pct, 2),
            'explained_power_cumulative': explained_power_cumulative,
            'explained_power_cumulative_pct': round(explained_power_cumulative_pct, 2),
            # Time metrics
            'minutes_above_th': minutes_above_th,
            'minutes_explained': minutes_below_th_iteration,
            'minutes_explained_pct': round(time_pct_iteration, 2),
            'minutes_explained_cumulative': minutes_below_th_cumulative,
            'minutes_explained_cumulative_pct': round(time_pct_cumulative, 2),
            # Sanity check - negative values (physically impossible)
            'minutes_negative': minutes_negative,
            'power_negative': round(power_negative, 2),
            # Missing values - rows that disappeared (NaN in current but exist in baseline)
            'minutes_missing': minutes_missing,
        }

        results_list.append(result)

        # Log results
        logger.info(f"Phase {phase} - Power: {explained_power_iteration:.0f}W this run, "
                   f"{explained_power_cumulative:.0f}W cumulative ({explained_power_cumulative_pct:.1f}%)")
        logger.info(f"Phase {phase} - Time: {minutes_below_th_iteration} min this run, "
                   f"{minutes_below_th_cumulative}/{minutes_above_th} min cumulative ({time_pct_cumulative:.1f}%)")

    # Create DataFrame from results
    results_df = pd.DataFrame(results_list)

    # Load previous history and append, or start fresh
    if run_number > 0 and prev_eval is not None:
        historical_df = pd.concat([prev_eval, results_df], ignore_index=True)
    else:
        historical_df = results_df

    # Save to CSV
    historical_df.to_csv(historical_eval_output_path, index=False)
    logger.info(f"Evaluation saved to {historical_eval_output_path}")
    logger.info("Evaluation process completed.")

    # Return results dict for potential use in stopping conditions
    return {phase: {k: v for k, v in r.items() if k not in ['house_id', 'run_number', 'threshold', 'phase']}
            for r in results_list for phase in [r['phase']]}


def _save_negative_values(data, remaining_col_data, negative_mask, house_id, run_number, phase):
    """Save negative value details to experiment-level CSV for debugging."""
    negative_values_path = Path(OUTPUT_BASE_PATH) / "negative_values.csv"

    negative_rows = data.loc[negative_mask, ['timestamp']].copy()
    negative_rows['house_id'] = house_id
    negative_rows['run_number'] = run_number
    negative_rows['phase'] = phase
    negative_rows['remaining_power'] = remaining_col_data[negative_mask].values

    # Append to CSV (create with header if doesn't exist)
    file_exists = negative_values_path.exists()
    negative_rows.to_csv(negative_values_path, mode='a', header=not file_exists, index=False)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python eval_segmentation.py <house_id> <run_number> [threshold]")
    else:
        house_id = sys.argv[1]
        run_number = int(sys.argv[2])
        threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 1500

        evaluate_segmentation(house_id, run_number, threshold)
