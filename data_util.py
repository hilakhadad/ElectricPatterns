import logging
import os

# Dynamic thresholds
DEFAULT_THRESHOLD = 1600
THRESHOLD_STEP = 200  # Change step per iteration
MIN_THRESHOLD = 800  # Minimum threshold allowed
IMPROVEMENT_THRESHOLD = 2  # Minimum improvement (%) required to avoid lowering the threshold
STOP_AT_50_PERCENT = 50  # Stop threshold adjustments when separation reaches 50% of total consumption

RAW_INPUT_DIRECTORY = "/sise/shanigu-group/hilakese-dorins/HouseholdData"

BASE = "/sise/shanigu-group/hilakese-dorins/SequenceData/experiment_pipeline"

INPUT_DIRECTORY = BASE

OUTPUT_BASE_PATH = BASE

PLOT_DIR = f"{BASE}/user_request_plots"

LOGS_DIRECTORY = f"{BASE}/logs/"

ERRORS_DIRECTORY = f"{BASE}/errors/"

def setup_logging(house_id, run_number, logs_directory=LOGS_DIRECTORY):
    """
    Set up logging for a specific house and run number.
    Args:
        house_id (str): ID of the house being processed.
        run_number (int): Run number of the experiment_pipeline.
        script_name (str): Name of the script.
        logs_directory (str): Base directory for logs.
    Returns:
        logging.Logger: Configured logger.
    """
    script_name = house_id

    # Create directory for logs if it doesn't exist
    run_log_directory = os.path.join(logs_directory, f"run_{run_number}")
    os.makedirs(run_log_directory, exist_ok=True)

    # Define log file path
    log_file = os.path.join(run_log_directory, f"{script_name}_{house_id}.log")

    # Set up logger
    logger = logging.getLogger(f"{script_name}_{house_id}_run_{run_number}")
    logger.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
