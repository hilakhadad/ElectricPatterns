"""
Evaluation pipeline step.

Calculates metrics comparing original vs remaining consumption.
"""
import pandas as pd
import os

import core
from core import setup_logging
from disaggregation.rectangle.segmentation.evaluation import calculate_phase_metrics, save_negative_values


def process_evaluation(house_id: str, run_number: int, threshold: int, actual_threshold: int = None) -> dict:
    """
    Evaluate segmentation results - processes monthly files.

    Args:
        house_id: House identifier
        run_number: Current run number
        threshold: Power threshold in watts (evaluation denominator — fixed across iterations)
        actual_threshold: Detection threshold actually used in this iteration (for reporting)

    Returns:
        dict: Evaluation results per phase
    """
    from pathlib import Path

    logger = setup_logging(house_id, run_number, core.LOGS_DIRECTORY)
    logger.info(f"Evaluation for house {house_id}, run {run_number}")

    # Paths - now using monthly folder structure
    output_dir = Path(f"{core.OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}")
    summarized_dir = output_dir / "summarized"
    baseline_dir = Path(f"{core.OUTPUT_BASE_PATH}/run_0/house_{house_id}") / "summarized"
    eval_path = output_dir / f"evaluation_history_{house_id}.csv"

    prev_summarized_dir = None
    prev_eval_path = None
    if run_number > 0:
        prev_dir = Path(f"{core.OUTPUT_BASE_PATH}/run_{run_number - 1}/house_{house_id}")
        prev_summarized_dir = prev_dir / "summarized"
        prev_eval_path = prev_dir / f"evaluation_history_{house_id}.csv"

    # Load data - concatenate all monthly files
    if not summarized_dir.is_dir():
        logger.error(f"Summarized folder not found: {summarized_dir}")
        return None

    summarized_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.pkl"))
    if not summarized_files:
        logger.error(f"No summarized files found in {summarized_dir}")
        return None

    def normalize_timestamps(df):
        """Ensure timestamps are datetime64[ns] for consistent merging."""
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    try:
        # Load and concatenate current data
        current_data = pd.concat([pd.read_pickle(f) for f in summarized_files], ignore_index=True)
        current_data = normalize_timestamps(current_data)

        # Load baseline data
        if run_number == 0:
            baseline_data = current_data
        else:
            baseline_files = sorted(baseline_dir.glob(f"summarized_{house_id}_*.pkl"))
            baseline_data = pd.concat([pd.read_pickle(f) for f in baseline_files], ignore_index=True)
            baseline_data = normalize_timestamps(baseline_data)

        # Load previous run data
        prev_data = None
        prev_eval = None
        if run_number > 0:
            if prev_summarized_dir and prev_summarized_dir.is_dir():
                prev_files = sorted(prev_summarized_dir.glob(f"summarized_{house_id}_*.pkl"))
                if prev_files:
                    prev_data = pd.concat([pd.read_pickle(f) for f in prev_files], ignore_index=True)
                    prev_data = normalize_timestamps(prev_data)
            if prev_eval_path and prev_eval_path.exists():
                prev_eval = pd.read_csv(prev_eval_path)
    except Exception as e:
        logger.error(f"Failed to read data: {e}")
        return None

    # Count duplicate timestamps for data quality reporting
    baseline_duplicates = baseline_data['timestamp'].duplicated().sum()
    current_duplicates = current_data['timestamp'].duplicated().sum() if run_number > 0 else baseline_duplicates
    logger.info(f"Duplicate timestamps - baseline: {baseline_duplicates}, current: {current_duplicates}")

    # Merge data on timestamp for proper alignment
    # This ensures we compare the same timestamps across runs
    if run_number == 0:
        merged_data = baseline_data.copy()
    else:
        # Remove duplicate timestamps to prevent Cartesian product explosion
        baseline_dedup = baseline_data.drop_duplicates(subset=['timestamp'], keep='first')
        current_dedup = current_data.drop_duplicates(subset=['timestamp'], keep='first')

        logger.info(f"Baseline: {len(baseline_data)} rows, {len(baseline_dedup)} unique timestamps")
        logger.info(f"Current: {len(current_data)} rows, {len(current_dedup)} unique timestamps")

        # Merge baseline with current on timestamp
        merged_data = baseline_dedup[['timestamp']].merge(
            current_dedup,
            on='timestamp',
            how='left',
            suffixes=('', '_current')
        )
        # Add baseline columns (columns not already in merged_data)
        baseline_indexed = baseline_dedup.set_index('timestamp')
        for col in baseline_dedup.columns:
            if col != 'timestamp' and col not in merged_data.columns:
                merged_data[col] = baseline_indexed[col].reindex(merged_data['timestamp']).values

        # Overwrite original_{phase} with baseline values (run 0 originals)
        # Current run's "original" is actually previous run's remaining - not the true original
        for phase in ['w1', 'w2', 'w3']:
            original_col = f'original_{phase}'
            if original_col in baseline_indexed.columns:
                merged_data[original_col] = baseline_indexed[original_col].reindex(merged_data['timestamp']).values

        # Merge previous run data if exists
        if prev_data is not None:
            prev_dedup = prev_data.drop_duplicates(subset=['timestamp'], keep='first')
            prev_cols = {f'remaining_{p}': f'prev_remaining_{p}' for p in ['w1', 'w2', 'w3']}
            prev_subset = prev_dedup[['timestamp'] + [f'remaining_{p}' for p in ['w1', 'w2', 'w3']]].rename(columns=prev_cols)
            merged_data = merged_data.merge(prev_subset, on='timestamp', how='left')

    # Calculate metrics
    phases = ['w1', 'w2', 'w3']
    results_list = []

    for phase in phases:
        logger.info(f"Processing phase {phase}")

        original_col = f'original_{phase}'
        remaining_col = f'remaining_{phase}'

        if original_col not in baseline_data.columns:
            logger.warning(f"Column {original_col} not found")
            continue

        # Use merged data - now properly aligned by timestamp
        baseline_original = merged_data[original_col] if original_col in merged_data.columns else baseline_data[original_col]
        current_remaining = merged_data[remaining_col] if remaining_col in merged_data.columns else None
        prev_remaining = merged_data[f'prev_remaining_{phase}'] if f'prev_remaining_{phase}' in merged_data.columns else None

        metrics = calculate_phase_metrics(
            baseline_original, current_remaining, prev_remaining, threshold, run_number
        )

        # Check for negative values in duration columns too
        duration_cols = [f'short_duration_{phase}', f'medium_duration_{phase}', f'long_duration_{phase}']
        for col in duration_cols:
            if col in merged_data.columns:
                neg_mask = merged_data[col] < 0
                neg_count = neg_mask.sum()
                neg_power = merged_data.loc[neg_mask, col].sum()
                if neg_count > 0:
                    metrics['minutes_negative'] += neg_count
                    metrics['power_negative'] += neg_power
                    logger.warning(f"Phase {phase} - NEGATIVE in {col}: {neg_count} min, {neg_power:.0f}W")

        if metrics['minutes_negative'] > 0:
            logger.warning(f"Phase {phase} - TOTAL NEGATIVE: {metrics['minutes_negative']} min, {metrics['power_negative']:.0f}W")
            negative_mask = current_remaining < 0
            save_negative_values(
                merged_data, current_remaining, negative_mask,
                house_id, run_number, phase, core.OUTPUT_BASE_PATH
            )

        result = {
            'house_id': house_id,
            'run_number': run_number,
            'threshold': threshold,
            'actual_threshold': actual_threshold if actual_threshold is not None else threshold,
            'phase': phase,
            'duplicate_timestamps': current_duplicates,
            **metrics
        }
        results_list.append(result)

        logger.info(f"Phase {phase} - Power: {metrics['explained_power']:.0f}W ({metrics['explained_power_cumulative_pct']:.1f}% cumulative)")
        logger.info(f"Phase {phase} - Time: {metrics['minutes_explained_cumulative']}/{metrics['minutes_above_th']} min ({metrics['minutes_explained_cumulative_pct']:.1f}%)")

    # Save results
    results_df = pd.DataFrame(results_list)
    historical_df = pd.concat([prev_eval, results_df], ignore_index=True) if prev_eval is not None else results_df

    historical_df.to_csv(eval_path, index=False)
    logger.info(f"Evaluation saved to {eval_path}")

    return {r['phase']: {k: v for k, v in r.items() if k not in ['house_id', 'run_number', 'threshold', 'phase']}
            for r in results_list}
