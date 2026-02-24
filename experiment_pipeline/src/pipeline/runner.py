"""
Unified pipeline runner for both static and dynamic threshold experiments.

Handles:
- Static experiments (single threshold, N iterations) -- exp000 through exp008
- Dynamic experiments (threshold_schedule, len(schedule) iterations) -- exp010+

Mode is auto-detected from the experiment's threshold_schedule field:
- threshold_schedule is None -> static mode
- threshold_schedule is a list -> dynamic mode

Implementation is split across:
  - runner.py (this file) -- main run_pipeline() orchestration
  - pipeline_setup.py -- _setup_paths_and_modules, _get_git_hash, _setup_logger
  - post_pipeline.py -- _run_dynamic_post_pipeline
"""
import os
import time
import traceback
from pathlib import Path
from tqdm import tqdm

from .pipeline_setup import _get_git_hash, _setup_paths_and_modules, _setup_logger
from .post_pipeline import _run_dynamic_post_pipeline


_SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()  # experiment_pipeline/


def run_pipeline(
    house_id: str,
    experiment_name: str,
    output_path: str,
    max_iterations: int = 2,
    input_path: str = None,
    quiet: bool = False,
    skip_visualization: bool = False,
    minimal_output: bool = False,
    skip_identification: bool = False,
) -> dict:
    """
    Run the full pipeline for a single house.

    Automatically detects whether the experiment is static or dynamic
    based on exp_config.threshold_schedule.

    Args:
        house_id: House ID to process
        experiment_name: Experiment name from core.config
        output_path: Where to save outputs
        max_iterations: Number of iterations (static experiments only; ignored for dynamic)
        input_path: Path to input CSV files (default: INPUT/HouseholdData)
        quiet: If True, suppress console output
        skip_visualization: If True, skip visualization step
        minimal_output: If True, delete intermediate pkl files after building
                       unified JSON (dynamic experiments only)
        skip_identification: If True, skip session grouping + classification

    Returns:
        dict with results: {'success': bool, 'iterations': int, 'error': str or None}
    """
    if input_path is None:
        input_path = str(_SCRIPT_DIR.parent / "INPUT" / "HouseholdData")

    # Reload paths and pipeline modules
    (
        process_detection, process_matching, process_segmentation,
        process_evaluation, process_visualization,
        get_experiment, save_experiment_metadata,
        find_house_data_path, find_previous_run_summarized,
    ) = _setup_paths_and_modules(output_path, input_path)

    # Load experiment config
    try:
        exp_config = get_experiment(experiment_name)
    except KeyError as e:
        return {'success': False, 'iterations': 0, 'error': f"Unknown experiment: {e}"}

    # Determine mode
    is_dynamic = exp_config.threshold_schedule is not None

    if is_dynamic:
        threshold_schedule = exp_config.threshold_schedule
        iterations = list(enumerate(threshold_schedule))
        evaluation_threshold = min(threshold_schedule)
    else:
        threshold_schedule = None
        iterations = [(i, exp_config.threshold) for i in range(max_iterations)]
        evaluation_threshold = exp_config.threshold

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/logs", exist_ok=True)

    # Set up logging
    logger = _setup_logger(output_path, house_id, quiet)

    # Save experiment metadata
    git_hash = _get_git_hash()
    save_experiment_metadata(exp_config, output_path, git_hash)

    # Log startup info
    if is_dynamic:
        logger.info(f"Starting DYNAMIC THRESHOLD pipeline for house {house_id}")
        logger.info(f"Experiment: {exp_config.exp_id} - {exp_config.description}")
        logger.info(f"Threshold schedule: {threshold_schedule}")
    else:
        logger.info(f"Starting pipeline for house {house_id}")
        logger.info(f"Experiment: {exp_config.exp_id} - {exp_config.description}")
        logger.info(f"Threshold: {exp_config.threshold}W, Max iterations: {max_iterations}")
    logger.info(f"Output path: {output_path}")

    # Preprocessing: normalization (before iteration loop)
    use_normalization = getattr(exp_config, 'use_normalization', False)
    if use_normalization:
        norm_method = getattr(exp_config, 'normalization_method', 'none')
        norm_params = getattr(exp_config, 'normalization_params', None)
        logger.info(f"Normalization enabled: method='{norm_method}'")

        from core.normalization import preprocess_normalize
        t0_norm = time.time()
        preprocessed_base = preprocess_normalize(
            input_path=input_path,
            house_id=house_id,
            output_path=output_path,
            method=norm_method,
            params=norm_params,
            logger=logger,
        )
        logger.info(f"Normalization complete ({time.time() - t0_norm:.1f}s)")

        # Override input paths so all pipeline steps use normalized data
        input_path = preprocessed_base
        import core
        import core.paths
        core.paths.RAW_INPUT_DIRECTORY = preprocessed_base
        core.RAW_INPUT_DIRECTORY = preprocessed_base

    iterations_completed = 0
    all_device_profiles = {}  # Dynamic mode: collect profiles across iterations

    # Main iteration loop
    for run_number, threshold in iterations:
        logger.info(f"\n{'#'*60}")
        if is_dynamic:
            logger.info(f"ITERATION {run_number}: THRESHOLD = {threshold}W")
        else:
            logger.info(f"ITERATION {run_number} of {len(iterations)}")
        logger.info(f"{'#'*60}")

        run_dir = Path(output_path) / f"run_{run_number}"

        # Find input data
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
            step_times = {}

            # Count input files for progress display
            if input_data_path.is_dir():
                num_files = len(list(input_data_path.glob("*.csv"))) or len(list(input_data_path.glob("*.pkl")))
            else:
                num_files = 1

            output_dir = str(run_dir / f"house_{house_id}")
            os.makedirs(output_dir, exist_ok=True)

            if is_dynamic:
                # Dynamic: run detection + matching, then segmentation separately (capture profiles)
                steps = [
                    ('Detection', lambda th=threshold: process_detection(
                        house_id=house_id, run_number=run_number, threshold=th, config=exp_config)),
                    ('Matching', lambda th=threshold: process_matching(
                        house_id=house_id, run_number=run_number, threshold=th)),
                ]

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

                # Segmentation with device profile capture
                pbar.set_postfix_str('Segmentation')
                t0 = time.time()
                run_profiles = process_segmentation(
                    house_id=house_id,
                    run_number=run_number,
                    skip_large_file=True,
                    capture_device_profiles=True,
                    use_nan_imputation=getattr(exp_config, 'use_nan_imputation', False),
                )
                step_times['segmentation'] = time.time() - t0
                logger.info(f"  Segmentation took {step_times['segmentation']:.1f}s")
                if run_profiles:
                    all_device_profiles[run_number] = run_profiles

                # Evaluation and optional visualization
                eval_viz_steps = [
                    ('Evaluation', lambda eth=evaluation_threshold, ath=threshold: process_evaluation(
                        house_id=house_id, run_number=run_number, threshold=eth, actual_threshold=ath)),
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

            else:
                # Static: run all steps in sequence
                steps = [
                    ('Detection', lambda: process_detection(
                        house_id=house_id, run_number=run_number, threshold=threshold, config=exp_config)),
                    ('Matching', lambda: process_matching(
                        house_id=house_id, run_number=run_number, threshold=threshold)),
                    ('Segmentation', lambda: process_segmentation(
                        house_id=house_id, run_number=run_number, skip_large_file=True,
                        use_nan_imputation=getattr(exp_config, 'use_nan_imputation', False))),
                    ('Evaluation', lambda th=threshold: process_evaluation(
                        house_id=house_id, run_number=run_number, threshold=th, actual_threshold=th)),
                ]

                if not skip_visualization:
                    steps.append(('Visualization', lambda: process_visualization(
                        house_id=house_id, run_number=run_number, threshold=threshold)))

                pbar = tqdm(
                    steps,
                    desc=f"House {house_id} iter {run_number} ({num_files} files)",
                    leave=False,
                )
                for step_name, step_func in pbar:
                    pbar.set_postfix_str(step_name)
                    t0 = time.time()
                    step_func()
                    step_times[step_name.lower()] = time.time() - t0
                    logger.info(f"  {step_name} took {step_times[step_name.lower()]:.1f}s")

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
            logger.error(traceback.format_exc())
            return {'success': False, 'iterations': iterations_completed, 'error': str(e)}

    # Dynamic post-pipeline steps
    if is_dynamic and iterations_completed > 0:
        # Guided cycle recovery (before identification, after all iterations)
        use_guided_recovery = getattr(exp_config, 'use_guided_recovery', False)
        if use_guided_recovery:
            try:
                from disaggregation.rectangle.pipeline.recovery_step import process_guided_recovery
                recovery_result = process_guided_recovery(
                    output_path=output_path,
                    house_id=house_id,
                    threshold_schedule=threshold_schedule,
                    config=exp_config,
                    run_logger=logger,
                )
                logger.info(f"Recovery: {recovery_result}")
            except Exception as e:
                logger.error(f"Error in guided recovery: {e}")
                logger.error(traceback.format_exc())

        # Wave recovery: detect wave-shaped patterns missed by rectangle matching
        use_wave_recovery = getattr(exp_config, 'use_wave_recovery', False)
        if use_wave_recovery:
            try:
                from disaggregation.wave_recovery.pipeline.wave_recovery_step import process_wave_recovery
                logger.info(f"{'=' * 60}")
                logger.info("WAVE RECOVERY")
                logger.info(f"{'=' * 60}")
                t0_wave = time.time()
                wave_result = process_wave_recovery(
                    output_path=output_path,
                    house_id=house_id,
                    threshold_schedule=threshold_schedule,
                    config=exp_config,
                    run_logger=logger,
                )
                logger.info(f"Wave recovery complete ({time.time() - t0_wave:.1f}s): {wave_result}")
            except Exception as e:
                logger.error(f"Error in wave recovery: {e}")
                logger.error(traceback.format_exc())

        _run_dynamic_post_pipeline(
            output_path=output_path,
            house_id=house_id,
            threshold_schedule=threshold_schedule,
            iterations_completed=iterations_completed,
            all_device_profiles=all_device_profiles,
            minimal_output=minimal_output,
            skip_identification=skip_identification,
            logger=logger,
        )

    mode_str = "Dynamic threshold pipeline" if is_dynamic else "Pipeline"
    logger.info(f"\n{mode_str} completed: {iterations_completed} iterations")
    return {'success': True, 'iterations': iterations_completed, 'error': None}
