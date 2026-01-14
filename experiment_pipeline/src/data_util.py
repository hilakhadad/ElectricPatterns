import logging
import os

# Dynamic thresholds
DEFAULT_THRESHOLD = 1600
THRESHOLD_STEP = 200  # Change step per iteration
MIN_THRESHOLD = 800  # Minimum threshold allowed
IMPROVEMENT_THRESHOLD = 2  # Minimum improvement (%) required to avoid lowering the threshold
STOP_AT_50_PERCENT = 50  # Stop threshold adjustments when separation reaches 50% of total consumption

# Get the absolute path to the experiment_pipeline directory
import sys
from pathlib import Path
_BASE_DIR = Path(__file__).parent.parent.absolute()  # Go up from src/ to experiment_pipeline/

RAW_INPUT_DIRECTORY = str(_BASE_DIR / "INPUT" / "HouseholdData")

# All outputs go to OUTPUT directory
OUTPUT_ROOT = str(_BASE_DIR / "OUTPUT")

BASE = OUTPUT_ROOT

INPUT_DIRECTORY = OUTPUT_ROOT

OUTPUT_BASE_PATH = OUTPUT_ROOT

PLOT_DIR = f"{OUTPUT_ROOT}/plots"

LOGS_DIRECTORY = f"{OUTPUT_ROOT}/logs/"

ERRORS_DIRECTORY = f"{OUTPUT_ROOT}/errors/"


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
    os.makedirs(logs_directory, exist_ok=True)

    # Define log file path
    log_file = os.path.join(logs_directory, f"{house_id}.log")

    # Set up logger
    logger = logging.getLogger(f"{house_id}_run_{run_number}")
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

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger
