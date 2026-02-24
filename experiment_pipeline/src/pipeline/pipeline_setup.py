"""
Pipeline setup helpers.

Handles path/module reloading, git hash retrieval, and logger configuration.
"""
import logging
import os
import importlib


def _get_git_hash():
    """Get current git commit hash, or None if unavailable."""
    try:
        import subprocess
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except Exception:
        return None


def _setup_paths_and_modules(output_path: str, input_path: str):
    """
    Reload core.paths and pipeline modules to pick up updated paths.

    This is necessary because the pipeline uses module-level globals for paths,
    which must be updated when running multiple houses or experiments in sequence.

    Returns:
        Tuple of (process_detection, process_matching, process_segmentation,
                  process_evaluation, process_visualization, core_imports)
    """
    import core.paths
    importlib.reload(core.paths)

    core.paths.OUTPUT_BASE_PATH = output_path
    core.paths.OUTPUT_ROOT = output_path
    core.paths.INPUT_DIRECTORY = output_path
    core.paths.LOGS_DIRECTORY = f"{output_path}/logs/"
    core.paths.RAW_INPUT_DIRECTORY = input_path

    import core.logging_setup
    importlib.reload(core.logging_setup)

    import core
    importlib.reload(core)

    import disaggregation.rectangle.pipeline.detection_step
    import disaggregation.rectangle.pipeline.matching_step
    import disaggregation.rectangle.pipeline.segmentation_step
    import disaggregation.rectangle.pipeline.evaluation_step
    import disaggregation.rectangle.pipeline.visualization_step
    importlib.reload(disaggregation.rectangle.pipeline.detection_step)
    importlib.reload(disaggregation.rectangle.pipeline.matching_step)
    importlib.reload(disaggregation.rectangle.pipeline.segmentation_step)
    importlib.reload(disaggregation.rectangle.pipeline.evaluation_step)
    importlib.reload(disaggregation.rectangle.pipeline.visualization_step)

    process_detection = disaggregation.rectangle.pipeline.detection_step.process_detection
    process_matching = disaggregation.rectangle.pipeline.matching_step.process_matching
    process_segmentation = disaggregation.rectangle.pipeline.segmentation_step.process_segmentation
    process_evaluation = disaggregation.rectangle.pipeline.evaluation_step.process_evaluation
    process_visualization = disaggregation.rectangle.pipeline.visualization_step.process_visualization

    from core import get_experiment, save_experiment_metadata, find_house_data_path, find_previous_run_summarized

    return (
        process_detection, process_matching, process_segmentation,
        process_evaluation, process_visualization,
        get_experiment, save_experiment_metadata,
        find_house_data_path, find_previous_run_summarized,
    )


def _setup_logger(output_path: str, house_id: str, quiet: bool):
    """Set up per-house logger with file and optional console handlers."""
    log_handlers = [logging.FileHandler(f"{output_path}/logs/test_{house_id}.log", encoding='utf-8')]
    if not quiet:
        log_handlers.append(logging.StreamHandler())

    logger = logging.getLogger(f"pipeline_{house_id}_{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    for handler in log_handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    return logger
