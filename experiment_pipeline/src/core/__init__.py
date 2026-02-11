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

# Data loading
from .data_loader import load_power_data, find_house_data_path, find_previous_run_summarized, build_data_files_dict

# Monthly I/O helpers
from .monthly_io import (
    save_dataframe_by_month,
    load_dataframe_from_folder,
    find_output_path,
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
    # Data loading
    'load_power_data',
    'find_house_data_path',
    'find_previous_run_summarized',
    'build_data_files_dict',
    # Monthly I/O
    'save_dataframe_by_month',
    'load_dataframe_from_folder',
    'find_output_path',
]
