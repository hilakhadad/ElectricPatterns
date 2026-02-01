"""
Evaluation pipeline step.

Calculates metrics comparing original vs remaining consumption.
"""
import pandas as pd
import os

import core
from core import setup_logging
from segmentation.evaluation import calculate_phase_metrics, save_negative_values


def process_evaluation(house_id: str, run_number: int, threshold: int) -> dict:
    """
    Evaluate segmentation results - processes monthly files.

    Args:
        house_id: House identifier
        run_number: Current run number
        threshold: Power threshold in watts

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

    summarized_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.csv"))
    if not summarized_files:
        logger.error(f"No summarized files found in {summarized_dir}")
        return None

    try:
        # Load and concatenate current data
        current_data = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in summarized_files], ignore_index=True)

        # Load baseline data
        if run_number == 0:
            baseline_data = current_data
        else:
            baseline_files = sorted(baseline_dir.glob(f"summarized_{house_id}_*.csv"))
            baseline_data = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in baseline_files], ignore_index=True)

        # Load previous run data
        prev_data = None
        prev_eval = None
        if run_number > 0:
            if prev_summarized_dir and prev_summarized_dir.is_dir():
                prev_files = sorted(prev_summarized_dir.glob(f"summarized_{house_id}_*.csv"))
                if prev_files:
                    prev_data = pd.concat([pd.read_csv(f, parse_dates=['timestamp']) for f in prev_files], ignore_index=True)
            if prev_eval_path and prev_eval_path.exists():
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
                house_id, run_number, phase, core.OUTPUT_BASE_PATH
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
