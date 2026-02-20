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


# ============================================================================
# Active experiments
# ============================================================================

EXPERIMENTS = {
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

# ============================================================================
# Legacy experiments (exp000-exp008) — kept for backward compatibility.
# These document the evolution from baseline to dynamic threshold.
# Use get_experiment() with include_legacy=True to access them.
# ============================================================================

LEGACY_EXPERIMENTS = {
    'exp000_baseline': ExperimentConfig(
        exp_id='exp000',
        description='Baseline: Original detection with 1600W threshold, 80% for off events',
        threshold=1600, off_threshold_factor=0.8, expand_event_factor=0.05,
        use_gradual_detection=True, gradual_window_minutes=3, gradual_direction_consistency=0.7,
    ),
    'exp001_gradual_detection': ExperimentConfig(
        exp_id='exp001',
        description='Add smart gradual detection (80-130% range, ±2min window)',
        threshold=1600, off_threshold_factor=0.8, expand_event_factor=0.05,
        use_gradual_detection=True, gradual_window_minutes=3, gradual_direction_consistency=0.7,
    ),
    'exp002_lower_TH': ExperimentConfig(
        exp_id='exp002',
        description='Lower threshold to 1500W with gradual detection',
        threshold=1500, off_threshold_factor=0.8, expand_event_factor=0.05,
        use_gradual_detection=True, gradual_window_minutes=3, gradual_direction_consistency=0.7,
    ),
    'exp003_progressive_search': ExperimentConfig(
        exp_id='exp003',
        description='Progressive window search: try 1min, then 2min, then 3min windows',
        threshold=1500, off_threshold_factor=0.8, expand_event_factor=0.05,
        use_gradual_detection=True, gradual_window_minutes=3, gradual_direction_consistency=0.7,
        progressive_window_search=True,
    ),
    'exp004_noisy_matching': ExperimentConfig(
        exp_id='exp004',
        description='Stage 2 noisy matching: match events with noise between ON/OFF',
        threshold=1500, off_threshold_factor=0.8, expand_event_factor=0.05,
        use_gradual_detection=True, gradual_window_minutes=3, gradual_direction_consistency=0.7,
        progressive_window_search=True,
    ),
    'exp005_asymmetric_windows': ExperimentConfig(
        exp_id='exp005',
        description='Asymmetric window search for gradual ON/OFF detection',
        threshold=1500, off_threshold_factor=0.8, expand_event_factor=0.2,
        use_gradual_detection=True, gradual_window_minutes=3, gradual_direction_consistency=0.7,
        progressive_window_search=True,
    ),
    'exp006_partial_matching': ExperimentConfig(
        exp_id='exp006',
        description='Stage 3 partial matching: ON/OFF magnitude diff >350W creates remainder',
        threshold=1500, off_threshold_factor=0.8, expand_event_factor=0.2,
        use_gradual_detection=True, gradual_window_minutes=3, gradual_direction_consistency=0.7,
        progressive_window_search=True,
    ),
    'exp007_symmetric_threshold': ExperimentConfig(
        exp_id='exp007',
        description='1300W symmetric ON/OFF detection (factor=1.0)',
        threshold=1300, off_threshold_factor=1.0, expand_event_factor=0.2,
        use_gradual_detection=True, gradual_window_minutes=3, gradual_direction_consistency=0.7,
        progressive_window_search=True, use_near_threshold_detection=True,
    ),
    'exp008_tail_extension': ExperimentConfig(
        exp_id='exp008',
        description='Tail extension: extend OFF events through residual power decay',
        threshold=1300, off_threshold_factor=1.0, expand_event_factor=0.2,
        use_gradual_detection=True, gradual_window_minutes=3, gradual_direction_consistency=0.7,
        progressive_window_search=True, use_near_threshold_detection=True, use_tail_extension=True,
    ),
}


# Default experiment — Dynamic Threshold (exp010) is the standard pipeline
DEFAULT_EXPERIMENT = 'exp010_dynamic_threshold'


def get_experiment(exp_name: str, include_legacy: bool = True) -> ExperimentConfig:
    """
    Get experiment configuration by name.

    Args:
        exp_name: Name of experiment (e.g., 'exp010_dynamic_threshold' or 'exp010')
        include_legacy: If True, also search LEGACY_EXPERIMENTS (exp000-exp008)

    Returns:
        ExperimentConfig object

    Raises:
        KeyError: If experiment name not found
    """
    # Search active experiments first
    if exp_name in EXPERIMENTS:
        return EXPERIMENTS[exp_name]
    for key, config in EXPERIMENTS.items():
        if config.exp_id == exp_name:
            return config

    # Search legacy experiments
    if include_legacy:
        if exp_name in LEGACY_EXPERIMENTS:
            return LEGACY_EXPERIMENTS[exp_name]
        for _, config in LEGACY_EXPERIMENTS.items():
            if config.exp_id == exp_name:
                return config

    available = list(EXPERIMENTS.keys())
    if include_legacy:
        available += list(LEGACY_EXPERIMENTS.keys())
    raise KeyError(f"Experiment '{exp_name}' not found. Available: {available}")


def list_experiments(include_legacy: bool = False) -> Dict[str, str]:
    """List available experiments with descriptions.

    Args:
        include_legacy: If True, also include exp000-exp008.
    """
    result = {name: config.description for name, config in EXPERIMENTS.items()}
    if include_legacy:
        result.update({name: f"[LEGACY] {config.description}"
                       for name, config in LEGACY_EXPERIMENTS.items()})
    return result


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
