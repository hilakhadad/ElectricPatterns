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
from tqdm import tqdm

# Fix encoding for Windows console (safer approach that won't close stdout)
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
    except Exception:
        pass  # Ignore if it fails

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_SCRIPT_DIR = Path(__file__).parent.parent.absolute()


def run_pipeline_for_house(
    house_id: str,
    experiment_name: str,
    output_path: str,
    max_iterations: int = 2,
    input_path: str = None,
    quiet: bool = False,
    skip_visualization: bool = False
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

    from core import get_experiment, save_experiment_metadata, find_house_data_path

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

        # Check input file/folder (supports both old CSV and new monthly folder structure)
        if run_number == 0:
            input_dir = input_path
        else:
            input_dir = f"{output_path}/run_{run_number}/HouseholdData"

        try:
            input_data_path = find_house_data_path(input_dir, house_id)
            logger.info(f"Found input: {input_data_path}")
        except FileNotFoundError as e:
            if run_number == 0:
                return {'success': False, 'iterations': 0, 'error': str(e)}
            else:
                logger.info(f"No more data for iteration {run_number}, stopping")
                break

        try:
            import time
            step_times = {}

            # Count input files for progress display
            if input_data_path.is_dir():
                num_files = len(list(input_data_path.glob("*.csv")))
            else:
                num_files = 1

            # Define pipeline steps
            steps = [
                ('Detection', lambda: process_detection(house_id=house_id, run_number=run_number, threshold=threshold, config=exp_config)),
                ('Matching', lambda: process_matching(house_id=house_id, run_number=run_number, threshold=threshold)),
                ('Segmentation', lambda: process_segmentation(house_id=house_id, run_number=run_number, skip_large_file=True)),
                ('Evaluation', lambda: process_evaluation(house_id=house_id, run_number=run_number, threshold=threshold)),
            ]

            if not skip_visualization:
                steps.append(('Visualization', lambda: process_visualization(house_id=house_id, run_number=run_number, threshold=threshold)))

            output_dir = f"{output_path}/run_{run_number}/house_{house_id}"
            os.makedirs(output_dir, exist_ok=True)

            # Run pipeline steps with progress bar
            pbar = tqdm(steps, desc=f"House {house_id} iter {run_number} ({num_files} files)", leave=False)
            for step_name, step_func in pbar:
                pbar.set_postfix_str(step_name)
                t0 = time.time()
                step_func()
                step_times[step_name.lower()] = time.time() - t0
                logger.info(f"  {step_name} took {step_times[step_name.lower()]:.1f}s")

            total_time = sum(step_times.values())
            logger.info(f"  Iteration {run_number} total: {total_time:.1f}s")

            iterations_completed += 1

            # Check for next iteration (supports both folder and file structure)
            try:
                next_input_dir = f"{output_path}/run_{run_number + 1}/HouseholdData"
                find_house_data_path(next_input_dir, house_id)
            except FileNotFoundError:
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
