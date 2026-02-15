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


def _generate_dynamic_evaluation_summary(
    output_path: str,
    house_id: str,
    threshold_schedule: list,
    iterations_completed: int,
    logger,
) -> None:
    """
    Generate threshold-independent evaluation summary for dynamic threshold runs.

    The standard evaluation_history CSV uses `total_power_above_th` as denominator,
    which changes with each threshold — making cumulative percentages misleading.

    This summary uses total original power (fixed) as denominator, so cumulative
    percentages grow monotonically and are directly comparable across iterations
    and with other experiments.

    Saves: dynamic_evaluation_summary_{house_id}.csv at the experiment root.
    """
    import pandas as pd

    output_dir = Path(output_path)

    # Load run_0 summarized data (has original power)
    run0_summarized = output_dir / "run_0" / f"house_{house_id}" / "summarized"
    if not run0_summarized.is_dir():
        logger.warning("Cannot generate dynamic evaluation summary: run_0 summarized not found")
        return

    run0_files = sorted(run0_summarized.glob(f"summarized_{house_id}_*.pkl"))
    if not run0_files:
        logger.warning("Cannot generate dynamic evaluation summary: no run_0 summarized files")
        return

    baseline = pd.concat([pd.read_pickle(f) for f in run0_files], ignore_index=True)
    baseline['timestamp'] = pd.to_datetime(baseline['timestamp'])
    baseline = baseline.drop_duplicates(subset=['timestamp'], keep='first')

    # Calculate total original power per phase (fixed denominator)
    original_power = {}
    for phase in ['w1', 'w2', 'w3']:
        col = f'original_{phase}'
        if col in baseline.columns:
            original_power[phase] = baseline[col].clip(lower=0).sum()
        else:
            original_power[phase] = 0

    rows = []
    prev_remaining = {phase: original_power[phase] for phase in ['w1', 'w2', 'w3']}

    for run_number in range(iterations_completed):
        threshold = threshold_schedule[run_number] if run_number < len(threshold_schedule) else 0

        # Load this run's summarized data
        run_summarized = output_dir / f"run_{run_number}" / f"house_{house_id}" / "summarized"
        if not run_summarized.is_dir():
            break

        run_files = sorted(run_summarized.glob(f"summarized_{house_id}_*.pkl"))
        if not run_files:
            break

        run_data = pd.concat([pd.read_pickle(f) for f in run_files], ignore_index=True)
        run_data['timestamp'] = pd.to_datetime(run_data['timestamp'])
        run_data = run_data.drop_duplicates(subset=['timestamp'], keep='first')

        for phase in ['w1', 'w2', 'w3']:
            remaining_col = f'remaining_{phase}'
            if remaining_col not in run_data.columns:
                continue

            current_remaining = run_data[remaining_col].clip(lower=0).sum()
            orig = original_power[phase]

            # Power explained by THIS iteration
            iteration_explained = max(0, prev_remaining[phase] - current_remaining)
            # Cumulative explained from original
            cumulative_explained = max(0, orig - current_remaining)

            rows.append({
                'run_number': run_number,
                'threshold': threshold,
                'phase': phase,
                'original_power': round(orig, 1),
                'remaining_power': round(current_remaining, 1),
                'iteration_explained': round(iteration_explained, 1),
                'iteration_explained_pct': round(iteration_explained / orig * 100, 2) if orig > 0 else 0,
                'cumulative_explained': round(cumulative_explained, 1),
                'cumulative_explained_pct': round(cumulative_explained / orig * 100, 2) if orig > 0 else 0,
            })

            prev_remaining[phase] = current_remaining

    if not rows:
        logger.warning("No data for dynamic evaluation summary")
        return

    summary_df = pd.DataFrame(rows)
    summary_path = output_dir / f"dynamic_evaluation_summary_{house_id}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Dynamic evaluation summary saved to {summary_path}")

    # Log final cumulative results
    last_run = summary_df[summary_df['run_number'] == summary_df['run_number'].max()]
    for _, row in last_run.iterrows():
        logger.info(
            f"  {row['phase']}: {row['cumulative_explained_pct']:.1f}% of total power explained "
            f"({row['cumulative_explained']:.0f}W / {row['original_power']:.0f}W)"
        )


def _cleanup_intermediate_files(
    experiment_dir: Path,
    house_id: str,
    iterations_completed: int,
    logger,
) -> None:
    """
    Delete intermediate pkl files after unified JSON is built.

    Deletes:
    - on_off/, matches/, unmatched_on/, unmatched_off/, classification/ (all runs)
    - summarized/ for intermediate runs (keep run_0 + last run)

    Keeps:
    - summarized/ for run_0 (baseline for evaluation) and last run (final remaining)
    - evaluation CSV files, logs, device_activations JSON
    """
    import shutil

    logger.info("Starting cleanup of intermediate files...")

    for run_number in range(iterations_completed):
        run_dir = experiment_dir / f"run_{run_number}"
        if not run_dir.exists():
            continue

        house_dir = run_dir / f"house_{house_id}"
        if not house_dir.exists():
            continue

        # Always delete event-level pkl directories
        for subdir in ['on_off', 'matches', 'unmatched_on', 'unmatched_off', 'classification']:
            dir_path = house_dir / subdir
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"  Deleted {dir_path.relative_to(experiment_dir)}")

        # Delete intermediate summarized dirs (keep run_0 and last run)
        if 0 < run_number < iterations_completed - 1:
            summarized_dir = house_dir / "summarized"
            if summarized_dir.exists():
                shutil.rmtree(summarized_dir)
                logger.info(f"  Deleted intermediate {summarized_dir.relative_to(experiment_dir)}")

    logger.info("Cleanup completed")


def run_dynamic_pipeline_for_house(
    house_id: str,
    experiment_name: str,
    output_path: str,
    input_path: str = None,
    quiet: bool = False,
    skip_visualization: bool = False,
    minimal_output: bool = False,
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
        minimal_output: If True, delete intermediate pkl files after building unified JSON

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
    all_device_profiles = {}  # Collect profiles across all iterations

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

            # Define pipeline steps (except Segmentation, which we call directly to capture profiles)
            steps = [
                ('Detection', lambda th=threshold: process_detection(
                    house_id=house_id, run_number=run_number, threshold=th, config=exp_config)),
                ('Matching', lambda th=threshold: process_matching(
                    house_id=house_id, run_number=run_number, threshold=th)),
            ]

            output_dir = str(run_dir / f"house_{house_id}")
            os.makedirs(output_dir, exist_ok=True)

            # Run detection and matching
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

            # Run segmentation separately to capture device profiles
            pbar.set_postfix_str('Segmentation')
            t0 = time.time()
            run_profiles = process_segmentation(
                house_id=house_id,
                run_number=run_number,
                skip_large_file=True,
                capture_device_profiles=True
            )
            step_times['segmentation'] = time.time() - t0
            logger.info(f"  Segmentation took {step_times['segmentation']:.1f}s")
            if run_profiles:
                all_device_profiles[run_number] = run_profiles

            # Run evaluation and optional visualization
            eval_viz_steps = [
                ('Evaluation', lambda th=threshold: process_evaluation(
                    house_id=house_id, run_number=run_number, threshold=th)),
            ]
            if not skip_visualization:
                eval_viz_steps.append(('Visualization', lambda th=threshold: process_visualization(
                    house_id=house_id, run_number=run_number, threshold=th)))

            for step_name, step_func in eval_viz_steps:
                pbar.set_postfix_str(step_name)
                t0 = time.time()
                step_func()
                step_times[step_name.lower()] = time.time() - t0
                logger.info(f"  {step_name} took {step_times[step_name.lower()]:.1f}s")

            pbar.close()

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

    # Generate threshold-independent evaluation summary
    try:
        _generate_dynamic_evaluation_summary(
            output_path=output_path,
            house_id=house_id,
            threshold_schedule=threshold_schedule,
            iterations_completed=iterations_completed,
            logger=logger,
        )
    except Exception as e:
        logger.error(f"Error generating dynamic evaluation summary: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Build unified device activations JSON
    try:
        from output.activation_builder import build_device_activations_json

        json_path = build_device_activations_json(
            experiment_dir=Path(output_path),
            house_id=house_id,
            threshold_schedule=threshold_schedule,
            device_profiles=all_device_profiles,
        )
        logger.info(f"Device activations JSON saved to {json_path}")
    except Exception as e:
        logger.error(f"Error building device activations JSON: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Optional cleanup of intermediate files
    if minimal_output:
        try:
            _cleanup_intermediate_files(Path(output_path), house_id, iterations_completed, logger)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
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
    parser.add_argument("--minimal_output", action="store_true",
                        help="Delete intermediate pkl files after building unified JSON")

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
        minimal_output=args.minimal_output,
    )

    if result['success']:
        print(f"\nSuccess! Completed {result['iterations']} iterations")
        print(f"Output saved to: {output_path}")
    else:
        print(f"\nFailed: {result['error']}")


if __name__ == "__main__":
    main()
