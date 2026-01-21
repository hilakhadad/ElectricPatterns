"""
Centralized path management for experiment pipeline.

Contains all path constants and the PathManager class for organized path handling.
"""
from pathlib import Path
from typing import Optional
import os


# =============================================================================
# BASE DIRECTORIES - Global constants used throughout the pipeline
# =============================================================================

_BASE_DIR = Path(__file__).parent.parent.parent.absolute()  # experiment_pipeline/

# Input paths
RAW_INPUT_DIRECTORY = str(_BASE_DIR / "INPUT" / "HouseholdData")

# Output paths
OUTPUT_ROOT = str(_BASE_DIR / "OUTPUT")
OUTPUT_BASE_PATH = OUTPUT_ROOT
INPUT_DIRECTORY = OUTPUT_ROOT  # After run_0, input comes from previous output
BASE = OUTPUT_ROOT

# Subdirectories
PLOT_DIR = f"{OUTPUT_ROOT}/plots"
LOGS_DIRECTORY = f"{OUTPUT_ROOT}/logs/"
ERRORS_DIRECTORY = f"{OUTPUT_ROOT}/errors/"

# Threshold constants
DEFAULT_THRESHOLD = 1600
THRESHOLD_STEP = 200
MIN_THRESHOLD = 800
IMPROVEMENT_THRESHOLD = 2
STOP_AT_50_PERCENT = 90


# =============================================================================
# PATH MANAGER - Object-oriented path management for experiments
# =============================================================================

class PathManager:
    """
    Manages all paths for a specific experiment/house/run combination.

    Usage:
        paths = PathManager(exp_id='exp005', house_id='2', run_number=0)
        data_file = paths.input_data
        output_dir = paths.output_dir
    """

    def __init__(self, exp_id: str, house_id: str, run_number: int,
                 experiment_timestamp: Optional[str] = None):
        self.exp_id = exp_id
        self.house_id = house_id
        self.run_number = run_number
        self.experiment_timestamp = experiment_timestamp

    @property
    def experiment_dir(self) -> Path:
        """Root directory for this experiment."""
        if self.experiment_timestamp:
            exp_name = f"{self.exp_id}_{self.experiment_timestamp}"
        else:
            exp_name = self.exp_id
        return Path(OUTPUT_ROOT) / "experiments" / exp_name

    @property
    def house_dir(self) -> Path:
        """Directory for this house within the experiment."""
        return self.experiment_dir / f"house_{self.house_id}"

    @property
    def run_dir(self) -> Path:
        """Directory for this specific run."""
        return self.house_dir / f"run_{self.run_number}" / f"house_{self.house_id}"

    @property
    def logs_dir(self) -> Path:
        return self.house_dir / "logs"

    @property
    def plots_dir(self) -> Path:
        return self.house_dir / "plots"

    @property
    def input_data(self) -> Path:
        """Path to input data CSV file."""
        if self.run_number == 0:
            return Path(RAW_INPUT_DIRECTORY) / f"{self.house_id}.csv"
        return self.experiment_dir / f"house_{self.house_id}" / f"run_{self.run_number}" / "HouseholdData" / f"{self.house_id}.csv"

    @property
    def on_off_log(self) -> Path:
        return self.run_dir / f"on_off_{DEFAULT_THRESHOLD}.csv"

    @property
    def matches_file(self) -> Path:
        return self.run_dir / f"matches_{self.house_id}.csv"

    @property
    def segmented_file(self) -> Path:
        return self.run_dir / f"segmented_{self.house_id}.csv"

    @property
    def summarized_file(self) -> Path:
        return self.run_dir / f"summarized_{self.house_id}.csv"

    @property
    def evaluation_file(self) -> Path:
        return self.run_dir / f"evaluation_history_{self.house_id}.csv"

    @property
    def next_run_input(self) -> Path:
        next_run = self.run_number + 1
        return self.experiment_dir / f"house_{self.house_id}" / f"run_{next_run}" / "HouseholdData" / f"{self.house_id}.csv"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / f"{self.house_id}.log"

    def ensure_dirs(self):
        """Create all necessary directories."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.next_run_input.parent.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"PathManager(exp={self.exp_id}, house={self.house_id}, run={self.run_number})"


def get_experiment_folders(exp_name_pattern: str = None) -> list:
    """List all experiment folders, optionally filtered by name pattern."""
    experiments_dir = Path(OUTPUT_ROOT) / "experiments"
    if not experiments_dir.exists():
        return []

    folders = sorted(experiments_dir.iterdir())
    if exp_name_pattern:
        folders = [f for f in folders if exp_name_pattern in f.name]

    return folders
