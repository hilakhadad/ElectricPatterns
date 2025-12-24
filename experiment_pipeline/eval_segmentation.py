import pandas as pd
import sys
from data_util import *


def evaluate_segmentation(house_id, run_number, threshold):
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Evaluation process for house {house_id} in run {run_number}.")

    output_directory = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    prev_output_directory = f"{OUTPUT_BASE_PATH}/run_{run_number - 1}/house_{house_id}"
    prev_historical_eval_output_path = f"{prev_output_directory}/evaluation_history_{house_id}.csv"
    historical_eval_output_path = f"{output_directory}/evaluation_history_{house_id}.csv"

    summarized_output_path = f"{output_directory}/summarized_{house_id}.csv"
    prev_summarized_output_path = f"{prev_output_directory}/summarized_{house_id}.csv"

    logger.info(f"Loading summarized data from {summarized_output_path}.")
    if not os.path.exists(summarized_output_path):
        logger.error(f"Summarized data file not found: {summarized_output_path}")
        return

    try:
        data = pd.read_csv(summarized_output_path, parse_dates=['timestamp'])
        logger.info("Successfully loaded current summarized data.")
        prev_data = None
        if os.path.exists(prev_summarized_output_path):
            logger.info(f"Loading previous summarized data from {prev_summarized_output_path}.")
            prev_data = pd.read_csv(prev_summarized_output_path, parse_dates=['timestamp'])
    except Exception as e:
        logger.error(f"Failed to read summarized data for house {house_id}: {e}")
        return

    phases = ['w1', 'w2', 'w3']
    summary_results_list = []

    for phase in phases:
        summary_results = {'house_id': house_id, 'run_number': run_number, 'threshold': threshold, 'phase': phase}

        logger.info(f"Processing phase {phase}.")
        original = data[f'original_{phase}']
        original_bigger_threshold = data[data[f'original_{phase}'] > threshold][f'original_{phase}']
        remaining = data[f'remaining_{phase}']
        explained = original - remaining
        explained_ratio = (explained / original).fillna(0) * 100

        total_minutes = len(data)
        total_bigger_threshold = len(original_bigger_threshold)
        explained_minutes = (explained_ratio > 50).sum()

        summary_results[f'total_minutes'] = total_minutes
        summary_results[f'total_minutes>threshold'] = total_bigger_threshold
        summary_results[f'explained_minutes'] = explained_minutes
        summary_results[f'percentage_explained_minutes'] = (explained_minutes / total_minutes) * 100
        summary_results[f'percentage_explained_minutes>threshold'] = (explained_minutes / total_bigger_threshold) * 100

        logger.info(
            f"Phase {phase}: {explained_minutes}/{total_minutes} minutes explained ({summary_results[f'percentage_explained_minutes']:.2f}%).")
        logger.info(
            f"Phase {phase}: {explained_minutes}/{total_bigger_threshold} minutes bigger than threshold:{threshold} explained ({summary_results[f'percentage_explained_minutes>threshold']:.2f}%).")

        if prev_data is not None:
            prev_remaining = prev_data[f'remaining_{phase}']
            additional_explained = (prev_remaining - remaining).clip(lower=0).sum()
            summary_results[f'additional_explained'] = additional_explained
            logger.info(f"Additional explained energy for {phase}: {additional_explained:.2f}")

        summary_results_list.append(summary_results)

    summary_df = pd.DataFrame(summary_results_list)

    if os.path.exists(prev_historical_eval_output_path):
        logger.info(f"Appending results to historical evaluation file: {prev_historical_eval_output_path}.")
        historical_df = pd.read_csv(prev_historical_eval_output_path)
        historical_df = pd.concat([historical_df, summary_df], ignore_index=True)
    else:
        logger.info(f"Creating new historical evaluation file: {historical_eval_output_path}.")
        historical_df = summary_df

    historical_df.to_csv(historical_eval_output_path, index=False)
    logger.info(f"Historical evaluation successfully updated at {historical_eval_output_path}.")
    logger.info("Evaluation process completed.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python eval.py <house_id> <run_number>")
    else:
        house_id = sys.argv[1]
        run_number = int(sys.argv[2])
        threshold = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_THRESHOLD

        evaluate_segmentation(house_id, run_number, threshold)
