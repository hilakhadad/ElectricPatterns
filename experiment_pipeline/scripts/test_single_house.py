"""
Test script to run the pipeline on a single house locally
Change the LOCAL_INPUT_PATH to your test data directory
"""
import sys
import os
import logging
import pandas as pd
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# CONFIGURATION - CHANGE THESE TO YOUR LOCAL PATHS
# ============================================================================

# Get the project root directory (two levels up from this script)
_SCRIPT_DIR = Path(__file__).parent.parent.absolute()

# Set this to your local test data directory containing CSV files
# By default, uses the experiment_pipeline/INPUT/HouseholdData directory
LOCAL_INPUT_PATH = str(_SCRIPT_DIR / "INPUT" / "HouseholdData")

# Set this to where you want outputs to be saved
# By default, uses the experiment_pipeline/OUTPUT directory
LOCAL_OUTPUT_PATH = str(_SCRIPT_DIR / "OUTPUT")

# Which house to test (e.g., "example", "1", "2", etc.)
HOUSE_ID = "1_debug"  # ← CHANGE THIS IF NEEDED

# Experiment name (e.g., "exp000_baseline", "exp001_gradual_detection")
# If None, uses DEFAULT_THRESHOLD directly without experiment framework
EXPERIMENT_NAME = "exp003_progressive_search"  # ← CHANGE THIS TO RUN DIFFERENT EXPERIMENTS

# Number of iterations to run (each iteration uses output from previous as input)
# After each iteration, detected events are removed from data, allowing detection of
# smaller events that were previously "hidden" by larger ones
MAX_ITERATIONS = 5  # ← CHANGE THIS IF NEEDED

# Test with default threshold or custom (only used if EXPERIMENT_NAME is None)
# Note: Threshold stays CONSTANT across all iterations (for reproducibility)
DEFAULT_THRESHOLD = 1600

# ============================================================================

# Load experiment configuration if specified
from datetime import datetime
from detection_config import get_experiment, save_experiment_metadata

if EXPERIMENT_NAME:
    try:
        exp_config = get_experiment(EXPERIMENT_NAME)

        # Create experiment-specific output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_output_path = str(_SCRIPT_DIR / "OUTPUT" / "experiments" / f"{exp_config.exp_id}_{timestamp}")
        LOCAL_OUTPUT_PATH = exp_output_path

        # Override DEFAULT_THRESHOLD with experiment config
        DEFAULT_THRESHOLD = exp_config.threshold

        print(f"\n{'='*60}")
        print(f"Running Experiment: {exp_config.exp_id}")
        print(f"Description: {exp_config.description}")
        print(f"Output: {exp_output_path}")
        print(f"{'='*60}\n")
    except KeyError as e:
        print(f"Error: {e}")
        print(f"Please check EXPERIMENT_NAME in the script.")
        sys.exit(1)
else:
    exp_config = None
    print(f"\n{'='*60}")
    print(f"Running WITHOUT experiment framework")
    print(f"Using threshold: {DEFAULT_THRESHOLD}W")
    print(f"{'='*60}\n")

# IMPORTANT: Override the global OUTPUT_BASE_PATH before importing modules
import data_util
data_util.OUTPUT_BASE_PATH = LOCAL_OUTPUT_PATH
data_util.OUTPUT_ROOT = LOCAL_OUTPUT_PATH
data_util.INPUT_DIRECTORY = LOCAL_OUTPUT_PATH
data_util.LOGS_DIRECTORY = f"{LOCAL_OUTPUT_PATH}/logs/"

# Create output directories
os.makedirs(LOCAL_OUTPUT_PATH, exist_ok=True)
os.makedirs(f"{LOCAL_OUTPUT_PATH}/logs", exist_ok=True)

# Save experiment metadata if using experiment framework
if EXPERIMENT_NAME and exp_config:
    try:
        # Try to get git hash
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        git_hash = None

    save_experiment_metadata(exp_config, LOCAL_OUTPUT_PATH, git_hash)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{LOCAL_OUTPUT_PATH}/logs/test_{HOUSE_ID}.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def validate_input_file(run_number):
    """Check if the input file exists for given run number"""
    if run_number == 0:
        input_file = f"{LOCAL_INPUT_PATH}/{HOUSE_ID}.csv"
    else:
        # For subsequent iterations, use output from previous iteration
        input_file = f"{LOCAL_OUTPUT_PATH}/run_{run_number}/HouseholdData/{HOUSE_ID}.csv"

    if not os.path.exists(input_file):
        if run_number == 0:
            logger.error(f"Input file not found: {input_file}")
            logger.info(f"Please provide a CSV file at: {input_file}")
            return False
        else:
            # For iterations > 0, missing input means no more events to process
            logger.info(f"No input for iteration {run_number} - previous iteration found no new events")
            return False

    try:
        df = pd.read_csv(input_file)
        logger.info(f"Input file found with {len(df)} rows")
        return True
    except Exception as e:
        logger.error(f"Error reading input file: {e}")
        return False

def run_on_off_detection(run_number):
    """Run on/off event detection"""
    logger.info("\n" + "="*60)
    logger.info(f"STEP 1: ON/OFF EVENT DETECTION (on_off_log.py) - Iteration {run_number}")
    logger.info("="*60)

    try:
        from on_off_log import process_house
        output_dir = f"{LOCAL_OUTPUT_PATH}/run_{run_number}/house_{HOUSE_ID}"
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Running on_off_log for house {HOUSE_ID}...")

        # Pass experiment config if available
        if EXPERIMENT_NAME and exp_config:
            process_house(
                house_id=HOUSE_ID,
                run_number=run_number,
                threshold=DEFAULT_THRESHOLD,
                config=exp_config
            )
        else:
            process_house(
                house_id=HOUSE_ID,
                run_number=run_number,
                threshold=DEFAULT_THRESHOLD
            )
        logger.info("On/off detection completed")
        return True
    except Exception as e:
        logger.error(f"Error in on_off_log: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_matching(run_number):
    """Run event matching"""
    logger.info("\n" + "="*60)
    logger.info(f"STEP 2: EVENT MATCHING (new_matcher.py) - Iteration {run_number}")
    logger.info("="*60)

    try:
        from new_matcher import process_matches

        logger.info(f"Running event matching for house {HOUSE_ID}...")
        process_matches(
            house_id=HOUSE_ID,
            run_number=run_number,
            threshold=DEFAULT_THRESHOLD
        )
        logger.info("Event matching completed")
        return True
    except Exception as e:
        logger.error(f"Error in new_matcher: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_segmentation(run_number):
    """Run data segmentation"""
    logger.info("\n" + "="*60)
    logger.info(f"STEP 3: DATA SEGMENTATION (segmentation.py) - Iteration {run_number}")
    logger.info("="*60)

    try:
        from segmentation import process_segmentation

        logger.info(f"Running segmentation for house {HOUSE_ID}...")
        process_segmentation(
            house_id=HOUSE_ID,
            run_number=run_number
        )
        logger.info("Segmentation completed")
        return True
    except Exception as e:
        logger.error(f"Error in segmentation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_evaluation(run_number):
    """Run segmentation evaluation"""
    logger.info("\n" + "="*60)
    logger.info(f"STEP 4: EVALUATION (eval_segmentation.py) - Iteration {run_number}")
    logger.info("="*60)

    try:
        from eval_segmentation import evaluate_segmentation

        logger.info(f"Running evaluation for house {HOUSE_ID}...")
        evaluate_segmentation(
            house_id=HOUSE_ID,
            run_number=run_number,
            threshold=DEFAULT_THRESHOLD
        )
        logger.info("Evaluation completed")
        return True
    except Exception as e:
        logger.error(f"Error in eval_segmentation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_visualization(run_number):
    """Run visualization"""
    logger.info("\n" + "="*60)
    logger.info(f"STEP 5: VISUALIZATION (visualization_with_mark.py) - Iteration {run_number}")
    logger.info("="*60)

    try:
        from visualization_with_mark import process_visualization

        logger.info(f"Running visualization for house {HOUSE_ID}...")
        process_visualization(
            house_id=HOUSE_ID,
            run_number=run_number,
            threshold=DEFAULT_THRESHOLD
        )
        logger.info("Visualization completed")
        return True
    except Exception as e:
        logger.error(f"Error in visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_single_iteration(run_number):
    """Run all pipeline steps for a single iteration"""
    logger.info("\n" + "#"*60)
    logger.info(f"ITERATION {run_number} of {MAX_ITERATIONS}")
    logger.info("#"*60)

    # Check if input exists for this iteration
    if not validate_input_file(run_number):
        if run_number == 0:
            logger.error("Cannot proceed without valid input file")
            return None  # Fatal error
        else:
            return False  # No more iterations needed

    # Run pipeline steps
    steps = [
        ("On/Off Detection", run_on_off_detection),
        ("Event Matching", run_matching),
        ("Segmentation", run_segmentation),
        ("Evaluation", run_evaluation),
        ("Visualization", run_visualization),
    ]

    results = {}
    for step_name, step_func in steps:
        logger.info(f"\n>>> Running {step_name}...")
        results[step_name] = step_func(run_number)
        if not results[step_name]:
            logger.warning(f"{step_name} failed, but continuing with next steps...")

    # Check if segmentation produced output for next iteration
    next_input = f"{LOCAL_OUTPUT_PATH}/run_{run_number + 1}/HouseholdData/{HOUSE_ID}.csv"
    has_next = os.path.exists(next_input)

    logger.info(f"\nIteration {run_number} completed. Next iteration possible: {has_next}")
    return has_next


def main():
    logger.info("\n" + "="*60)
    logger.info("STARTING LOCAL TEST PIPELINE")
    logger.info("="*60)

    if EXPERIMENT_NAME and exp_config:
        logger.info(f"Experiment: {exp_config.exp_id}")
        logger.info(f"Description: {exp_config.description}")
        logger.info(f"Configuration:")
        for key, value in exp_config.to_dict().items():
            if key not in ['exp_id', 'description']:
                logger.info(f"  - {key}: {value}")

    logger.info(f"House ID: {HOUSE_ID}")
    logger.info(f"Input Path: {LOCAL_INPUT_PATH}")
    logger.info(f"Output Path: {LOCAL_OUTPUT_PATH}")
    logger.info(f"Threshold: {DEFAULT_THRESHOLD}W (constant across all iterations)")
    logger.info(f"Max Iterations: {MAX_ITERATIONS}")

    # Run iterations
    iterations_completed = 0
    for run_number in range(MAX_ITERATIONS):
        result = run_single_iteration(run_number)

        if result is None:
            # Fatal error on first iteration
            logger.error("Pipeline failed on first iteration")
            return

        iterations_completed += 1

        if not result:
            # No more events to process
            logger.info(f"Stopping after {iterations_completed} iterations - no more events found")
            break

    # Final Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Total iterations completed: {iterations_completed}")
    logger.info(f"Output files saved to: {LOCAL_OUTPUT_PATH}")
    logger.info(f"  - run_0/ through run_{iterations_completed - 1}/")
    logger.info(f"Logs saved to: {LOCAL_OUTPUT_PATH}/logs/test_{HOUSE_ID}.log")


if __name__ == "__main__":
    main()
