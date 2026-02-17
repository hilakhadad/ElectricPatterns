"""
Experiment configuration management.

Each experiment has a unique identifier (exp000, exp001, etc.) and a configuration
dictionary that defines detection parameters.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import json
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for a single detection experiment."""
    exp_id: str
    description: str
    threshold: int
    off_threshold_factor: float
    expand_event_factor: float
    use_gradual_detection: bool
    gradual_window_minutes: int
    gradual_direction_consistency: float
    progressive_window_search: bool = False
    use_near_threshold_detection: bool = False
    near_threshold_min_factor: float = 0.85
    near_threshold_max_extend: int = 3
    use_tail_extension: bool = False
    tail_max_extension_minutes: int = 10
    tail_min_residual: int = 100
    tail_noise_tolerance: int = 30
    tail_min_gain: int = 100
    tail_min_residual_fraction: float = 0.05
    threshold_schedule: Optional[List[int]] = None  # Dynamic threshold: list of thresholds per iteration
    use_nan_imputation: bool = False  # Fill short NaN gaps at runtime to prevent false diff() events

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'exp_id': self.exp_id,
            'description': self.description,
            'threshold': self.threshold,
            'off_threshold_factor': self.off_threshold_factor,
            'expand_event_factor': self.expand_event_factor,
            'use_gradual_detection': self.use_gradual_detection,
            'gradual_window_minutes': self.gradual_window_minutes,
            'gradual_direction_consistency': self.gradual_direction_consistency,
            'progressive_window_search': self.progressive_window_search,
            'use_near_threshold_detection': self.use_near_threshold_detection,
            'near_threshold_min_factor': self.near_threshold_min_factor,
            'near_threshold_max_extend': self.near_threshold_max_extend,
            'use_tail_extension': self.use_tail_extension,
            'tail_max_extension_minutes': self.tail_max_extension_minutes,
            'tail_min_residual': self.tail_min_residual,
            'tail_noise_tolerance': self.tail_noise_tolerance,
            'tail_min_gain': self.tail_min_gain,
            'tail_min_residual_fraction': self.tail_min_residual_fraction,
            'threshold_schedule': self.threshold_schedule,
            'use_nan_imputation': self.use_nan_imputation,
        }

    def to_json(self, file_path: str):
        """Save config to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create config from dictionary."""
        return cls(**data)


# Experiment definitions
EXPERIMENTS = {
    'exp000_baseline': ExperimentConfig(
        exp_id='exp000',
        description='Baseline: Original detection with 1600W threshold, 80% for off events',
        threshold=1600,
        off_threshold_factor=0.8,
        expand_event_factor=0.05,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=False,
    ),

    'exp001_gradual_detection': ExperimentConfig(
        exp_id='exp001',
        description='Add smart gradual detection (80-130% range, ±2min window)',
        threshold=1600,
        off_threshold_factor=0.8,
        expand_event_factor=0.05,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=False,
    ),

    'exp002_lower_TH': ExperimentConfig(
        exp_id='exp002',
        description='Lower threshold to 1500W with gradual detection',
        threshold=1500,
        off_threshold_factor=0.8,
        expand_event_factor=0.05,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=False,
    ),

    'exp003_progressive_search': ExperimentConfig(
        exp_id='exp003',
        description='Progressive window search: try 1min, then 2min, then 3min windows',
        threshold=1500,
        off_threshold_factor=0.8,
        expand_event_factor=0.05,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
    ),

    'exp004_noisy_matching': ExperimentConfig(
        exp_id='exp004',
        description='Stage 2 noisy matching: match events with noise between ON/OFF using clipped cumsum segregation',
        threshold=1500,
        off_threshold_factor=0.8,
        expand_event_factor=0.05,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
    ),

    'exp005_asymmetric_windows': ExperimentConfig(
        exp_id='exp005',
        description='Improved gradual ON/OFF detection: asymmetric window search (symmetric, before-only, after-only) to capture events near max_threshold boundary that were previously missed.',
        threshold=1500,
        off_threshold_factor=0.8,
        expand_event_factor=0.2,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
    ),

    'exp006_partial_matching': ExperimentConfig(
        exp_id='exp006',
        description='Stage 3 partial matching: when ON/OFF magnitudes differ >350W, match using min magnitude and create remainder event for next iteration',
        threshold=1500,
        off_threshold_factor=0.8,
        expand_event_factor=0.2,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
    ),

    'exp007_symmetric_threshold': ExperimentConfig(
        exp_id='exp007',
        description='Lower threshold to 1300W with symmetric ON/OFF detection (factor=1.0) to catch more boilers and AC units',
        threshold=1300,
        off_threshold_factor=1.0,
        expand_event_factor=0.2,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
        use_near_threshold_detection=True,
    ),

    'exp008_tail_extension': ExperimentConfig(
        exp_id='exp008',
        description='Tail extension: extend OFF events through residual power decay (monotonic, max 10min) to capture full magnitude',
        threshold=1300,
        off_threshold_factor=1.0,
        expand_event_factor=0.2,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
        use_near_threshold_detection=True,
        use_tail_extension=True,
    ),

    'exp010_dynamic_threshold': ExperimentConfig(
        exp_id='exp010',
        description='Dynamic threshold: 2000->1500->1100->800W per iteration, targeting boilers->strong AC->medium AC->small AC',
        threshold=2000,
        off_threshold_factor=1.0,
        expand_event_factor=0.2,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
        use_near_threshold_detection=False,
        use_tail_extension=True,
        threshold_schedule=[2000, 1500, 1100, 800],
    ),

    'exp012_nan_imputation': ExperimentConfig(
        exp_id='exp012',
        description='NaN imputation test: exp010 + runtime NaN gap filling (ffill<=5min, interp<=60min)',
        threshold=2000,
        off_threshold_factor=1.0,
        expand_event_factor=0.2,
        use_gradual_detection=True,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
        use_near_threshold_detection=False,
        use_tail_extension=True,
        threshold_schedule=[2000, 1500, 1100, 800],
        use_nan_imputation=True,
    ),
}


# Default experiment — Dynamic Threshold (exp010) is the standard pipeline
DEFAULT_EXPERIMENT = 'exp010_dynamic_threshold'


def get_experiment(exp_name: str) -> ExperimentConfig:
    """
    Get experiment configuration by name.

    Args:
        exp_name: Name of experiment (e.g., 'exp000_baseline' or 'exp000')

    Returns:
        ExperimentConfig object

    Raises:
        KeyError: If experiment name not found
    """
    if exp_name in EXPERIMENTS:
        return EXPERIMENTS[exp_name]

    # Try with just exp ID (e.g., 'exp000')
    for key, config in EXPERIMENTS.items():
        if config.exp_id == exp_name:
            return config

    raise KeyError(f"Experiment '{exp_name}' not found. Available: {list(EXPERIMENTS.keys())}")


def list_experiments() -> Dict[str, str]:
    """List all available experiments with descriptions."""
    return {name: config.description for name, config in EXPERIMENTS.items()}


def save_experiment_metadata(exp_config: ExperimentConfig, output_dir: str, git_hash: str = None):
    """
    Save experiment metadata to output directory.

    Args:
        exp_config: Experiment configuration
        output_dir: Directory to save metadata
        git_hash: Optional git commit hash
    """
    import os

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'experiment': exp_config.to_dict(),
        'git_hash': git_hash,
    }

    metadata_path = os.path.join(output_dir, 'experiment_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
