"""
Run the pipeline with dynamic (decreasing) thresholds per iteration.

Each iteration uses a different threshold from the experiment's threshold_schedule,
targeting progressively smaller devices:
  - Iteration 0 (TH=2000): Boilers, water heaters
  - Iteration 1 (TH=1500): Strong ACs
  - Iteration 2 (TH=1100): Medium ACs
  - Iteration 3 (TH=800):  Small ACs

After matching, each iteration's matches are classified as device types.
After all iterations, an activation list summary is generated.

Usage:
    python scripts/test_dynamic_threshold.py --house_id 305
    python scripts/test_dynamic_threshold.py --house_id 305 --skip_visualization
"""
import sys
import os
import logging
import importlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
    except Exception:
        pass

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_SCRIPT_DIR = Path(__file__).parent.parent.absolute()


def run_dynamic_pipeline_for_house(
    house_id: str,
    experiment_name: str,
    output_path: str,
    input_path: str = None,
    quiet: bool = False,
    skip_visualization: bool = False,
) -> dict:
    """
    Run the full pipeline with dynamic thresholds for a single house.

    Each iteration uses a different threshold from the experiment's
    threshold_schedule. After matching, matches are classified as device types.

    Uses standard run_0/, run_1/ naming (compatible with pipeline path system
    and parallel batch execution). Threshold-to-run mapping is stored in
    experiment_metadata.json.

    Args:
        house_id: House ID to process
        experiment_name: Experiment name (must have threshold_schedule)
        output_path: Where to save outputs
        input_path: Path to input CSV files
        quiet: If True, suppress console output
        skip_visualization: If True, skip visualization step

    Returns:
        dict with results: {'success': bool, 'iterations': int, 'error': str or None}
    """
    if input_path is None:
        input_path = str(_SCRIPT_DIR.parent / "INPUT" / "HouseholdData")

    # Reload core.paths to update global paths for this run
    import core.paths
    importlib.reload(core.paths)

    # Override paths for this experiment (same as test_single_house.py)
    core.paths.OUTPUT_BASE_PATH = output_path
    core.paths.OUTPUT_ROOT = output_path
    core.paths.INPUT_DIRECTORY = output_path
    core.paths.LOGS_DIRECTORY = f"{output_path}/logs/"
    core.paths.RAW_INPUT_DIRECTORY = input_path

    # Reload core modules
    import core.logging_setup
    importlib.reload(core.logging_setup)
    import core
    importlib.reload(core)

    # Reload pipeline modules
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

    # Get reloaded functions
    process_detection = pipeline.detection.process_detection
    process_matching = pipeline.matching.process_matching
    process_segmentation = pipeline.segmentation.process_segmentation
    process_evaluation = pipeline.evaluation.process_evaluation
    process_visualization = pipeline.visualization.process_visualization

    from core import get_experiment, save_experiment_metadata, find_house_data_path, find_previous_run_summarized
    from classification.device_classifier import classify_iteration_matches, generate_activation_list

    # Load experiment config
    try:
        exp_config = get_experiment(experiment_name)
    except KeyError as e:
        return {'success': False, 'iterations': 0, 'error': f"Unknown experiment: {e}"}

    threshold_schedule = exp_config.threshold_schedule
    if not threshold_schedule:
        return {'success': False, 'iterations': 0,
                'error': f"Experiment '{experiment_name}' has no threshold_schedule"}

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/logs", exist_ok=True)

    # Set up logging
    log_handlers = [logging.FileHandler(f"{output_path}/logs/test_{house_id}.log", encoding='utf-8')]
    if not quiet:
        log_handlers.append(logging.StreamHandler())

    logger = logging.getLogger(f"dynamic_pipeline_{house_id}_{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    for handler in log_handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    # Save experiment metadata (includes threshold_schedule)
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except Exception:
        git_hash = None
    save_experiment_metadata(exp_config, output_path, git_hash)

    logger.info(f"Starting DYNAMIC THRESHOLD pipeline for house {house_id}")
    logger.info(f"Experiment: {exp_config.exp_id} - {exp_config.description}")
    logger.info(f"Threshold schedule: {threshold_schedule}")
    logger.info(f"Output path: {output_path}")

    iterations_completed = 0

    for run_number, threshold in enumerate(threshold_schedule):
        logger.info(f"\n{'#'*60}")
        logger.info(f"ITERATION {run_number}: THRESHOLD = {threshold}W")
        logger.info(f"{'#'*60}")

        # Standard run directory (pipeline expects this format)
        run_dir = Path(output_path) / f"run_{run_number}"

        # Check input: run 0 reads raw data, run N reads remaining from run N-1
        try:
            if run_number == 0:
                input_data_path = find_house_data_path(input_path, house_id)
            else:
                input_data_path = find_previous_run_summarized(output_path, house_id, run_number)
            logger.info(f"Found input: {input_data_path}")
        except FileNotFoundError as e:
            if run_number == 0:
                return {'success': False, 'iterations': 0, 'error': str(e)}
            else:
                logger.info(f"No summarized data from previous run, stopping at iteration {run_number}")
                break

        try:
            import time
            step_times = {}

            # Count input files for progress display
            if input_data_path.is_dir():
                num_files = len(list(input_data_path.glob("*.csv"))) or len(list(input_data_path.glob("*.pkl")))
            else:
                num_files = 1

            # Define pipeline steps (same as test_single_house.py, with dynamic threshold)
            steps = [
                ('Detection', lambda th=threshold: process_detection(
                    house_id=house_id, run_number=run_number, threshold=th, config=exp_config)),
                ('Matching', lambda th=threshold: process_matching(
                    house_id=house_id, run_number=run_number, threshold=th)),
                ('Segmentation', lambda: process_segmentation(
                    house_id=house_id, run_number=run_number, skip_large_file=True)),
                ('Evaluation', lambda th=threshold: process_evaluation(
                    house_id=house_id, run_number=run_number, threshold=th)),
            ]

            if not skip_visualization:
                steps.append(('Visualization', lambda th=threshold: process_visualization(
                    house_id=house_id, run_number=run_number, threshold=th)))

            output_dir = str(run_dir / f"house_{house_id}")
            os.makedirs(output_dir, exist_ok=True)

            # Run pipeline steps with progress bar
            pbar = tqdm(
                steps,
                desc=f"House {house_id} iter {run_number} TH={threshold}W ({num_files} files)",
                leave=False,
            )
            for step_name, step_func in pbar:
                pbar.set_postfix_str(step_name)
                t0 = time.time()
                step_func()
                step_times[step_name.lower()] = time.time() - t0
                logger.info(f"  {step_name} took {step_times[step_name.lower()]:.1f}s")

            # Classification step (after matching, before next iteration)
            t0 = time.time()
            classify_iteration_matches(
                run_dir=run_dir,
                house_id=house_id,
                run_number=run_number,
                threshold=threshold,
                parent_logger=logger,
            )
            step_times['classification'] = time.time() - t0
            logger.info(f"  Classification took {step_times['classification']:.1f}s")

            total_time = sum(step_times.values())
            logger.info(f"  Iteration {run_number} total: {total_time:.1f}s")

            iterations_completed += 1

            # Check if current run produced summarized output
            current_summarized = run_dir / f"house_{house_id}" / "summarized"
            if not current_summarized.is_dir() or not any(
                current_summarized.glob(f"summarized_{house_id}_*.pkl")
            ):
                logger.info("No summarized output from current run, stopping iterations")
                break

        except Exception as e:
            logger.error(f"Error in iteration {run_number}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'iterations': iterations_completed, 'error': str(e)}

    # Generate activation list after all iterations
    try:
        activation_list = generate_activation_list(
            experiment_dir=Path(output_path),
            house_id=house_id,
            threshold_schedule=threshold_schedule,
        )
        classified_rate = activation_list.get('summary', {}).get('overall_classified_rate', 0)
        logger.info(f"Activation list generated: classified_rate={classified_rate:.1%}")
    except Exception as e:
        logger.error(f"Error generating activation list: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info(f"\nDynamic threshold pipeline completed: {iterations_completed} iterations")
    return {'success': True, 'iterations': iterations_completed, 'error': None}


# ============================================================================
# STANDALONE CONFIGURATION
# ============================================================================
DEFAULT_HOUSE_ID = "305"
DEFAULT_EXPERIMENT_NAME = "exp010_dynamic_threshold"


def main():
    """Run dynamic threshold pipeline with command-line arguments or defaults."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run pipeline with dynamic (decreasing) thresholds per iteration"
    )
    parser.add_argument("--house_id", type=str, default=DEFAULT_HOUSE_ID,
                        help=f"House ID to process (default: {DEFAULT_HOUSE_ID})")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME,
                        help=f"Experiment name (default: {DEFAULT_EXPERIMENT_NAME})")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress console output")
    parser.add_argument("--skip_visualization", action="store_true",
                        help="Skip visualization step (recommended for batch)")

    args = parser.parse_args()

    from core import get_experiment

    exp_config = get_experiment(args.experiment_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_path or str(
        _SCRIPT_DIR / "OUTPUT" / "experiments" / f"{exp_config.exp_id}_{timestamp}"
    )

    print(f"\n{'='*60}")
    print(f"Dynamic Threshold Experiment: {exp_config.exp_id}")
    print(f"House: {args.house_id}")
    print(f"Threshold schedule: {exp_config.threshold_schedule}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    result = run_dynamic_pipeline_for_house(
        house_id=args.house_id,
        experiment_name=args.experiment_name,
        output_path=output_path,
        quiet=args.quiet,
        skip_visualization=args.skip_visualization,
    )

    if result['success']:
        print(f"\nSuccess! Completed {result['iterations']} iterations")
        print(f"Output saved to: {output_path}")
    else:
        print(f"\nFailed: {result['error']}")


if __name__ == "__main__":
    main()
