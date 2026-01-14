import subprocess
import pandas as pd
import sys
import os
from pathlib import Path

# Add parent directory (experiment_pipeline) to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_util import *


# Define the sequence of scripts and their required inputs and outputs
script_config = [
    {
        "script": "on_off_log.py",
        "input_check": lambda house_id, run_number: os.path.exists(f"{RAW_INPUT_DIRECTORY}/{house_id}.csv"),
        "output_check": lambda house_id, run_number: os.path.exists(f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}/on_off_1600.csv"),
    },
    {
        "script": "new_matcher.py",
        "input_check": lambda house_id, run_number: os.path.exists(f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}/on_off_1600.csv"),
        "output_check": lambda house_id, run_number: os.path.exists(f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}/matches_{house_id}.csv"),
    },
    {
        "script": "segmentation.py",
        "input_check": lambda house_id, run_number: os.path.exists(f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}/matches_{house_id}.csv"),
        "output_check": lambda house_id, run_number: os.path.exists(f"{INPUT_DIRECTORY}/run_{run_number + 1}/HouseholdData/{house_id}.csv"),
    },
    {
        "script": "eval_segmentation.py",
        "input_check": lambda house_id, run_number: os.path.exists(f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}/summarized_{house_id}.csv"),
        "output_check": lambda house_id, run_number: os.path.exists(f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}/separation_evaluation_{house_id}.csv"),
    },
    {
        "script": "visualization_with_mark.py",
        "input_check": lambda house_id, run_number: os.path.exists(f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}/summarized_{house_id}.csv"),
        "output_check": lambda house_id, run_number: os.path.exists(f"{OUTPUT_BASE_PATH}/run_{run_number}/plots/summarized_{house_id}"),
    },
]


def get_last_eval(house_id, run_number, logger):
    """ Retrieves the last evaluation to determine if threshold should change """

    eval_file = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}/separation_evaluation_{house_id}.csv"

    if not os.path.exists(eval_file):
        logger.warning(f"Evaluation file not found for house {house_id}, run {run_number}. Using default threshold.")
        return DEFAULT_THRESHOLD

    df = pd.read_csv(eval_file)

    if len(df) < 2:
        logger.info(f"Not enough data to compare previous runs for house {house_id}. Using default threshold.")
        return DEFAULT_THRESHOLD

    # Get the last two runs for comparison
    last_run = df[df["Run Number"] == run_number]
    prev_run = df[df["Run Number"] == run_number - 1]

    if last_run.empty or prev_run.empty:
        logger.info(f"Missing previous run data for house {house_id}. Using default threshold.")
        return DEFAULT_THRESHOLD

    last_separated = last_run["% Total Separated"].mean()
    prev_separated = prev_run["% Total Separated"].mean()

    improvement = last_separated - prev_separated

    logger.info(
        f"House {house_id}, Run {run_number}: Previous Separated {prev_separated:.2f}%, Current Separated {last_separated:.2f}%, Improvement: {improvement:.2f}%"
    )

    # Stop further threshold adjustments if separation reaches 50% of total consumption
    if last_separated >= STOP_AT_50_PERCENT:
        logger.info(f"Separation reached {STOP_AT_50_PERCENT}%, stopping further threshold changes.")
        return DEFAULT_THRESHOLD

    # Lower the threshold if improvement is less than the required threshold
    if improvement < IMPROVEMENT_THRESHOLD:
        new_threshold = max(DEFAULT_THRESHOLD - THRESHOLD_STEP, MIN_THRESHOLD)
        logger.info(f"Small improvement ({improvement:.2f}%). Lowering threshold to {new_threshold} for next run.")
        return new_threshold

    # If improvement is significant, increase back to the default threshold to verify efficiency
    logger.info(f"Significant improvement detected ({improvement:.2f}%). Raising threshold back to {DEFAULT_THRESHOLD} to check efficiency.")
    return DEFAULT_THRESHOLD


def run_script(script, house_id, run_number, logger, threshold=None):
    """ Runs a given script with optional threshold """

    try:
        python_path = "/sise/home/hilakese/.conda/envs/nilm_stat_env/bin/python"

        command = [python_path, script, house_id, str(run_number)]
        if threshold is not None:
            command.append(str(threshold))

        logger.info(
            f"Running {script} for house {house_id}, run {run_number} with threshold {threshold if threshold else 'default'}...")
        subprocess.run(command, check=True)
        logger.info(f"Completed {script} for house {house_id}, run {run_number}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error while running {script} for house {house_id}, run {run_number}: {e}")
        raise


def process_single_house(house_id, run_number, iterations=10, force=False):
    """ Runs scripts for a single house, adapting the threshold dynamically. """

    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Processing house {house_id} starting from run {run_number} for {iterations} iterations...")

    for i in range(iterations):
        logger.info(f"Iteration {i + 1} for house {house_id}, run {run_number}...")

        threshold = get_last_eval(house_id, run_number, logger)

        for config in script_config:
            script = config["script"]
            input_check = config["input_check"]
            output_check = config["output_check"]

            if not input_check(house_id, run_number):
                logger.warning(f"Skipping {script} for house {house_id}, run {run_number}: Required input not found.")
                break

            if not force and output_check(house_id, run_number):
                logger.info(f"Skipping {script} for house {house_id}, run {run_number}: Output already exists.")
                continue

            run_script(script, house_id, run_number, logger, threshold)

            if script == "new_matcher.py" and not output_check(house_id, run_number):
                for temp_threshold in range(threshold-100, MIN_THRESHOLD-100, -100):
                    if not output_check(house_id, run_number):
                        logger.info(f"🔎 Trying threshold: {temp_threshold}")
                        run_script("on_off_log.py", house_id, run_number, logger, temp_threshold)
                        run_script(script, house_id, run_number, logger, temp_threshold)
                        threshold = temp_threshold
                    else:
                        break

        run_number += 1


def run_all_houses(run_number, iterations=10, force=False):
    """ Runs the process for all houses and evaluates results after the last step. """

    if not os.path.exists(RAW_INPUT_DIRECTORY):
        print("RAW_INPUT_DIRECTORY does not exist.")
        return

    all_houses = [file.split('.')[0] for file in os.listdir(RAW_INPUT_DIRECTORY) if file.endswith('.csv')]

    if not all_houses:
        print("No houses found in RAW_INPUT_DIRECTORY.")
        return

    print(f"Found {len(all_houses)} houses to process.")
    for house_id in all_houses:
        process_single_house(house_id, run_number, iterations=iterations, force=force)

if __name__ == "__main__":

    run_number = int(sys.argv[1])
    force = "--force" in sys.argv

    # Get iterations from command-line arguments (default is 10)
    iterations = 10
    for arg in sys.argv:
        if arg.startswith("--iterations="):
            iterations = int(arg.split("=")[1])

    if len(sys.argv) >= 3 and sys.argv[2] not in ["--force", f"--iterations={iterations}"]:
        house_id = sys.argv[2]
        process_single_house(house_id, run_number, iterations=iterations, force=force)
    else:
        run_all_houses(run_number, iterations=iterations, force=force)
