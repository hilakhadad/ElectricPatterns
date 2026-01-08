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
HOUSE_ID = "1"  # ← CHANGE THIS IF NEEDED

# Experiment name (e.g., "exp000_baseline", "exp001_gradual_detection")
# If None, uses DEFAULT_THRESHOLD directly without experiment framework
EXPERIMENT_NAME = "exp002_lower_TH"  # ← CHANGE THIS TO RUN DIFFERENT EXPERIMENTS

# Test with default threshold or custom (only used if EXPERIMENT_NAME is None)
DEFAULT_THRESHOLD = 1600
RUN_NUMBER = 0

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

def validate_input_file():
    """Check if the input file exists"""
    input_file = f"{LOCAL_INPUT_PATH}/{HOUSE_ID}.csv"
    
    if not os.path.exists(input_file):
        logger.error(f"❌ Input file not found: {input_file}")
        logger.info(f"Please provide a CSV file at: {input_file}")
        logger.info(f"Expected directory: {LOCAL_INPUT_PATH}")
        return False
    
    try:
        df = pd.read_csv(input_file)
        logger.info(f"✓ Input file found with {len(df)} rows and columns: {list(df.columns)}")
        return True
    except Exception as e:
        logger.error(f"❌ Error reading input file: {e}")
        return False

def run_on_off_detection():
    """Run on/off event detection"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: ON/OFF EVENT DETECTION (on_off_log.py)")
    logger.info("="*60)

    try:
        from on_off_log import process_house
        output_dir = f"{LOCAL_OUTPUT_PATH}/run_{RUN_NUMBER}/house_{HOUSE_ID}"
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Running on_off_log for house {HOUSE_ID}...")

        # Pass experiment config if available
        if EXPERIMENT_NAME and exp_config:
            process_house(
                house_id=HOUSE_ID,
                run_number=RUN_NUMBER,
                threshold=DEFAULT_THRESHOLD,
                config=exp_config
            )
        else:
            process_house(
                house_id=HOUSE_ID,
                run_number=RUN_NUMBER,
                threshold=DEFAULT_THRESHOLD
            )
        logger.info("✓ On/off detection completed")
        return True
    except Exception as e:
        logger.error(f"❌ Error in on_off_log: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_matching():
    """Run event matching"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: EVENT MATCHING (new_matcher.py)")
    logger.info("="*60)

    try:
        from new_matcher import process_matches

        logger.info(f"Running event matching for house {HOUSE_ID}...")
        process_matches(
            house_id=HOUSE_ID,
            run_number=RUN_NUMBER,
            threshold=DEFAULT_THRESHOLD
        )
        logger.info("✓ Event matching completed")
        return True
    except Exception as e:
        logger.error(f"❌ Error in new_matcher: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_segmentation():
    """Run data segmentation"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: DATA SEGMENTATION (segmentation.py)")
    logger.info("="*60)

    try:
        from segmentation import process_segmentation

        logger.info(f"Running segmentation for house {HOUSE_ID}...")
        process_segmentation(
            house_id=HOUSE_ID,
            run_number=RUN_NUMBER
        )
        logger.info("✓ Segmentation completed")
        return True
    except Exception as e:
        logger.error(f"❌ Error in segmentation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_evaluation():
    """Run segmentation evaluation"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: EVALUATION (eval_segmentation.py)")
    logger.info("="*60)

    try:
        from eval_segmentation import evaluate_segmentation

        logger.info(f"Running evaluation for house {HOUSE_ID}...")
        evaluate_segmentation(
            house_id=HOUSE_ID,
            run_number=RUN_NUMBER,
            threshold=DEFAULT_THRESHOLD
        )
        logger.info("✓ Evaluation completed")
        return True
    except Exception as e:
        logger.error(f"❌ Error in eval_segmentation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_visualization():
    """Run visualization"""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: VISUALIZATION (visualization_with_mark.py)")
    logger.info("="*60)

    try:
        from visualization_with_mark import process_visualization

        logger.info(f"Running visualization for house {HOUSE_ID}...")
        process_visualization(
            house_id=HOUSE_ID,
            run_number=RUN_NUMBER,
            threshold=DEFAULT_THRESHOLD
        )
        logger.info("✓ Visualization completed")
        return True
    except Exception as e:
        logger.error(f"❌ Error in visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

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
    logger.info(f"Threshold: {DEFAULT_THRESHOLD}W")
    logger.info(f"Run Number: {RUN_NUMBER}")
    
    # Step 0: Validate input
    if not validate_input_file():
        logger.error("\n❌ Cannot proceed without valid input file")
        return
    
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
        results[step_name] = step_func()
        if not results[step_name]:
            logger.warning(f"⚠️  {step_name} failed, but continuing with next steps...")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST PIPELINE SUMMARY")
    logger.info("="*60)
    for step_name, success in results.items():
        status = "✓ PASSED" if success else "❌ FAILED"
        logger.info(f"{step_name}: {status}")
    
    logger.info(f"\nOutput files saved to: {LOCAL_OUTPUT_PATH}")
    logger.info(f"Logs saved to: {LOCAL_OUTPUT_PATH}/logs/test_{HOUSE_ID}.log")

if __name__ == "__main__":
    main()
