import pandas as pd
import sys
import os
from data_util import *

def evaluate_separation(house_id, run_number, threshold):
    """
    Evaluates the separation effectiveness for a given house and run.
    Stores results in a cumulative evaluation file per house.
    """
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)

    # Define file paths
    original_file = f"{RAW_INPUT_DIRECTORY}/{house_id}.csv"
    processed_file = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}/summarized_{house_id}.csv"

    # Cumulative file should be saved in the current run's directory
    current_cumulative_eval_file = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}/separation_evaluation_cumulative.csv"

    # Look for the cumulative evaluation file from the previous run (if exists)
    prev_cumulative_eval_file = f"{OUTPUT_BASE_PATH}/run_{run_number - 1}/house_{house_id}/separation_evaluation_cumulative.csv"

    # Ensure necessary files exist
    if not os.path.exists(original_file):
        logger.error(f"Original data missing for house {house_id}, run {run_number}. Skipping evaluation.")
        return

    if not os.path.exists(processed_file):
        logger.error(f"Processed data missing for house {house_id}, run {run_number}. Skipping evaluation.")
        return

    logger.info(f"Files found. Loading data for house {house_id}, run {run_number}...")

    df_original = pd.read_csv(original_file)
    df_processed = pd.read_csv(processed_file)

    phases = ['1', '2', '3']

    results = []

    for phase in phases:
        original_col = phase
        separated_cols = [f"short_duration_w{phase}", f"medium_duration_w{phase}", f"long_duration_w{phase}"]

        if original_col not in df_original.columns:
            logger.warning(f"Phase {phase} missing in original data for house {house_id}, run {run_number}. Skipping.")
            continue

        total_consumption = df_original[original_col].sum()
        consumption_above_threshold = df_original[df_original[original_col] > threshold][original_col].sum()
        separated_consumption = df_processed[separated_cols].sum().sum()

        if total_consumption == 0:
            logger.warning(f"Total consumption is 0 for house {house_id}, phase {phase}, run {run_number}. Skipping phase.")
            continue

        percent_total_separated = (separated_consumption / total_consumption) * 100 if total_consumption > 0 else 0
        percent_separated_from_above_threshold = (separated_consumption / consumption_above_threshold) * 100 if consumption_above_threshold > 0 else 0

        results.append({
            "House ID": house_id,
            "Run Number": run_number,
            "Phase": phase,
            "Total Consumption": total_consumption,
            "Threshold Used (TH)": threshold,  # Track threshold used
            "Consumption>threshold": consumption_above_threshold,
            "Separated Consumption": separated_consumption,
            "% Total Separated": percent_total_separated,
            "% Separated from>threshold": percent_separated_from_above_threshold,
            "Newly Separated in Run": 0  # Will be updated later
        })

        logger.info(
            f"Phase {phase} processed for house {house_id}, run {run_number}. Separated {percent_total_separated:.2f}% of total, {percent_separated_from_above_threshold:.2f}% of > {threshold}."
        )

    results_df = pd.DataFrame(results)

    # Load previous cumulative data if it exists in the previous run
    if os.path.exists(prev_cumulative_eval_file):
        cumulative_df = pd.read_csv(prev_cumulative_eval_file)

        # Identify the previous run's data
        prev_run_df = cumulative_df[cumulative_df["Run Number"] == run_number - 1]

        if not prev_run_df.empty:
            for phase in phases:
                prev_separated = prev_run_df[prev_run_df["Phase"] == phase]["Separated Consumption"].sum()
                new_separated = results_df[results_df["Phase"] == phase]["Separated Consumption"].sum()
                improvement = new_separated - prev_separated

                # Update "Newly Separated in Run"
                results_df.loc[results_df["Phase"] == phase, "Newly Separated in Run"] = improvement

                if abs(improvement) > 0.3 * prev_separated:  # Large change warning
                    logger.warning(f"Significant change in separation for house {house_id}, phase {phase}. Improvement: {improvement}W.")

        cumulative_df = pd.concat([cumulative_df, results_df], ignore_index=True)

    else:
        cumulative_df = results_df  # If no previous cumulative file exists, start fresh

    # Compute cumulative separation columns
    cumulative_df["Cumulative Separated Consumption"] = cumulative_df.groupby(["House ID", "Phase"])["Separated Consumption"].cumsum()
    cumulative_df["Cumulative % Total Separated"] = (cumulative_df["Cumulative Separated Consumption"] / cumulative_df["Total Consumption"]) * 100
    cumulative_df["Cumulative % Separated from>threshold"] = (cumulative_df["Cumulative Separated Consumption"] / cumulative_df["Consumption>threshold"]) * 100

    # Ensure the current run's directory exists before saving
    os.makedirs(os.path.dirname(current_cumulative_eval_file), exist_ok=True)

    # Save updated cumulative evaluation file in the current run's directory
    cumulative_df.to_csv(current_cumulative_eval_file, index=False)

    logger.info(f"Evaluation completed for house {house_id}, run {run_number}. Results saved to {current_cumulative_eval_file}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python eval.py <house_id> <run_number> <threshold>")
        sys.exit(1)

    house_id = sys.argv[1]
    run_number = int(sys.argv[2])
    threshold = int(sys.argv[3])

    evaluate_separation(house_id, run_number, threshold)
