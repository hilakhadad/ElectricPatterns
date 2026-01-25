"""
Test script to run the pipeline on a single house locally.
Can be run standalone or imported and called from test_array_of_houses.py
"""
import sys
import os
import logging
import importlib
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_SCRIPT_DIR = Path(__file__).parent.parent.absolute()


def run_pipeline_for_house(
    house_id: str,
    experiment_name: str,
    output_path: str,
    max_iterations: int = 2,
    input_path: str = None,
    quiet: bool = False
) -> dict:
    """
    Run the full pipeline for a single house.

    Args:
        house_id: House ID to process
        experiment_name: Experiment name from core.config
        output_path: Where to save outputs
        max_iterations: Number of iterations to run
        input_path: Path to input CSV files
        quiet: If True, suppress console output

    Returns:
        dict with results: {'success': bool, 'iterations': int, 'error': str or None}
    """
    if input_path is None:
        input_path = str(_SCRIPT_DIR.parent / "INPUT" / "HouseholdData")

    # Reload core.paths to update global paths for this run
    import core.paths
    importlib.reload(core.paths)

    # Override paths for this experiment
    core.paths.OUTPUT_BASE_PATH = output_path
    core.paths.OUTPUT_ROOT = output_path
    core.paths.INPUT_DIRECTORY = output_path
    core.paths.LOGS_DIRECTORY = f"{output_path}/logs/"
    core.paths.RAW_INPUT_DIRECTORY = input_path

    # Reload core modules that contain path variables
    import core.logging_setup
    importlib.reload(core.logging_setup)

    import core
    importlib.reload(core)

    # Reload pipeline modules to pick up updated paths
    # Must reload in correct order: pipeline first, then submodules
    import pipeline
    importlib.reload(pipeline)

    import pipeline.detection
    import pipeline.matching
    import pipeline.segmentation
    import pipeline.evaluation
    import pipeline.visualization
    importlib.reload(pipeline.detection)
    importlib.reload(pipeline.matching)
    importlib.reload(pipeline.segmentation)
    importlib.reload(pipeline.evaluation)
    importlib.reload(pipeline.visualization)

    # Now get the reloaded functions
    process_detection = pipeline.detection.process_detection
    process_matching = pipeline.matching.process_matching
    process_segmentation = pipeline.segmentation.process_segmentation
    process_evaluation = pipeline.evaluation.process_evaluation
    process_visualization = pipeline.visualization.process_visualization

    from core import get_experiment, save_experiment_metadata

    # Load experiment config
    try:
        exp_config = get_experiment(experiment_name)
        threshold = exp_config.threshold
    except KeyError as e:
        return {'success': False, 'iterations': 0, 'error': f"Unknown experiment: {e}"}

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/logs", exist_ok=True)

    # Set up logging
    log_handlers = [logging.FileHandler(f"{output_path}/logs/test_{house_id}.log", encoding='utf-8')]
    if not quiet:
        log_handlers.append(logging.StreamHandler())

    logger = logging.getLogger(f"pipeline_{house_id}_{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    for handler in log_handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    # Save experiment metadata
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except:
        git_hash = None
    save_experiment_metadata(exp_config, output_path, git_hash)

    logger.info(f"Starting pipeline for house {house_id}")
    logger.info(f"Experiment: {exp_config.exp_id} - {exp_config.description}")
    logger.info(f"Threshold: {threshold}W, Max iterations: {max_iterations}")
    logger.info(f"Output path: {output_path}")

    iterations_completed = 0

    for run_number in range(max_iterations):
        logger.info(f"\n{'#'*60}")
        logger.info(f"ITERATION {run_number} of {max_iterations}")
        logger.info(f"{'#'*60}")

        # Check input file
        if run_number == 0:
            input_file = f"{input_path}/{house_id}.csv"
        else:
            input_file = f"{output_path}/run_{run_number}/HouseholdData/{house_id}.csv"

        if not os.path.exists(input_file):
            if run_number == 0:
                return {'success': False, 'iterations': 0, 'error': f"Input file not found: {input_file}"}
            else:
                logger.info(f"No more data for iteration {run_number}, stopping")
                break

        try:
            # Step 1: Detection
            logger.info("Step 1: Detecting ON/OFF events...")
            output_dir = f"{output_path}/run_{run_number}/house_{house_id}"
            os.makedirs(output_dir, exist_ok=True)
            process_detection(house_id=house_id, run_number=run_number, threshold=threshold, config=exp_config)

            # Step 2: Matching
            logger.info("Step 2: Matching events...")
            process_matching(house_id=house_id, run_number=run_number, threshold=threshold)

            # Step 3: Segmentation
            logger.info("Step 3: Segmenting data...")
            process_segmentation(house_id=house_id, run_number=run_number)

            # Step 4: Evaluation
            logger.info("Step 4: Evaluating results...")
            process_evaluation(house_id=house_id, run_number=run_number, threshold=threshold)

            # Step 5: Visualization
            logger.info("Step 5: Creating visualizations...")
            process_visualization(house_id=house_id, run_number=run_number, threshold=threshold)

            iterations_completed += 1

            # Check for next iteration
            next_input = f"{output_path}/run_{run_number + 1}/HouseholdData/{house_id}.csv"
            if not os.path.exists(next_input):
                logger.info("No more events found, stopping iterations")
                break

        except Exception as e:
            logger.error(f"Error in iteration {run_number}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'iterations': iterations_completed, 'error': str(e)}

    logger.info(f"\nPipeline completed: {iterations_completed} iterations")
    return {'success': True, 'iterations': iterations_completed, 'error': None}


# ============================================================================
# STANDALONE CONFIGURATION
# ============================================================================
HOUSE_ID = "140_debug"
EXPERIMENT_NAME = "exp006_partial_matching"
MAX_ITERATIONS = 2


def main():
    """Run pipeline with standalone configuration."""
    from core import get_experiment

    exp_config = get_experiment(EXPERIMENT_NAME)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(_SCRIPT_DIR / "OUTPUT" / "experiments" / f"{exp_config.exp_id}_{timestamp}")

    print(f"\n{'='*60}")
    print(f"Running Experiment: {exp_config.exp_id}")
    print(f"House: {HOUSE_ID}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    result = run_pipeline_for_house(
        house_id=HOUSE_ID,
        experiment_name=EXPERIMENT_NAME,
        output_path=output_path,
        max_iterations=MAX_ITERATIONS,
        quiet=False
    )

    if result['success']:
        print(f"\nSuccess! Completed {result['iterations']} iterations")
        print(f"Output saved to: {output_path}")
    else:
        print(f"\nFailed: {result['error']}")


if __name__ == "__main__":
    main()
