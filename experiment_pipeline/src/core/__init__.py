"""Core infrastructure for experiment pipeline."""

# Config
from .config import (
    ExperimentConfig,
    get_experiment,
    list_experiments,
    save_experiment_metadata,
    EXPERIMENTS,
)

# Paths - all the globals that were in data_util.py
from .paths import (
    PathManager,
    RAW_INPUT_DIRECTORY,
    OUTPUT_ROOT,
    OUTPUT_BASE_PATH,
    INPUT_DIRECTORY,
    BASE,
    PLOT_DIR,
    LOGS_DIRECTORY,
    ERRORS_DIRECTORY,
    DEFAULT_THRESHOLD,
    THRESHOLD_STEP,
    MIN_THRESHOLD,
    IMPROVEMENT_THRESHOLD,
    STOP_AT_50_PERCENT,
    get_experiment_folders,
)

# Logging
from .logging_setup import (
    setup_logging,
    setup_pipeline_logger,
    get_logger,
    PipelineLogger,
)

__all__ = [
    # Config
    'ExperimentConfig',
    'get_experiment',
    'list_experiments',
    'save_experiment_metadata',
    'EXPERIMENTS',
    # Paths
    'PathManager',
    'RAW_INPUT_DIRECTORY',
    'OUTPUT_ROOT',
    'OUTPUT_BASE_PATH',
    'INPUT_DIRECTORY',
    'BASE',
    'PLOT_DIR',
    'LOGS_DIRECTORY',
    'ERRORS_DIRECTORY',
    'DEFAULT_THRESHOLD',
    'THRESHOLD_STEP',
    'MIN_THRESHOLD',
    'IMPROVEMENT_THRESHOLD',
    'STOP_AT_50_PERCENT',
    'get_experiment_folders',
    # Logging
    'setup_logging',
    'setup_pipeline_logger',
    'get_logger',
    'PipelineLogger',
]
