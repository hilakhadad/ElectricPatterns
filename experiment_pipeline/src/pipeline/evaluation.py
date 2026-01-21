"""
Evaluation pipeline step.

Calculates metrics comparing original vs remaining consumption.
"""
import pandas as pd
import os

from core import setup_logging, OUTPUT_BASE_PATH, LOGS_DIRECTORY
from segmentation.evaluation import calculate_phase_metrics, save_negative_values


def process_evaluation(house_id: str, run_number: int, threshold: int) -> dict:
    """
    Evaluate segmentation results.

    Args:
        house_id: House identifier
        run_number: Current run number
        threshold: Power threshold in watts

    Returns:
        dict: Evaluation results per phase
    """
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Evaluation for house {house_id}, run {run_number}")

    # Paths
    output_dir = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    current_path = f"{output_dir}/summarized_{house_id}.csv"
    baseline_dir = f"{OUTPUT_BASE_PATH}/run_0/house_{house_id}"
    baseline_path = f"{baseline_dir}/summarized_{house_id}.csv"
    eval_path = f"{output_dir}/evaluation_history_{house_id}.csv"

    prev_path = None
    prev_eval_path = None
    if run_number > 0:
        prev_dir = f"{OUTPUT_BASE_PATH}/run_{run_number - 1}/house_{house_id}"
        prev_path = f"{prev_dir}/summarized_{house_id}.csv"
        prev_eval_path = f"{prev_dir}/evaluation_history_{house_id}.csv"

    # Load data
    if not os.path.exists(current_path):
        logger.error(f"File not found: {current_path}")
        return None

    try:
        current_data = pd.read_csv(current_path, parse_dates=['timestamp'])
        baseline_data = current_data if run_number == 0 else pd.read_csv(baseline_path, parse_dates=['timestamp'])

        prev_data = None
        prev_eval = None
        if run_number > 0:
            if os.path.exists(prev_path):
                prev_data = pd.read_csv(prev_path, parse_dates=['timestamp'])
            if os.path.exists(prev_eval_path):
                prev_eval = pd.read_csv(prev_eval_path)
    except Exception as e:
        logger.error(f"Failed to read data: {e}")
        return None

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

        baseline_original = baseline_data[original_col]
        current_remaining = current_data[remaining_col]
        prev_remaining = prev_data[remaining_col] if prev_data is not None else None

        metrics = calculate_phase_metrics(
            baseline_original, current_remaining, prev_remaining, threshold, run_number
        )

        if metrics['minutes_negative'] > 0:
            logger.warning(f"Phase {phase} - NEGATIVE: {metrics['minutes_negative']} min, {metrics['power_negative']:.0f}W")
            negative_mask = current_remaining < 0
            save_negative_values(
                current_data, current_remaining, negative_mask,
                house_id, run_number, phase, OUTPUT_BASE_PATH
            )

        result = {
            'house_id': house_id,
            'run_number': run_number,
            'threshold': threshold,
            'phase': phase,
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
