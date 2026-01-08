"""
Configuration management for on/off event detection experiments.

Each experiment has a unique identifier (exp000, exp001, etc.) and a configuration
dictionary that defines detection parameters.
"""

from dataclasses import dataclass
from typing import Dict, Any
import json
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Configuration for a single detection experiment."""
    exp_id: str
    description: str
    threshold: int
    off_threshold_factor: float  # Multiplier for off threshold (e.g., 0.8 for 80%)
    expand_event_factor: float   # Factor for expand_event threshold (e.g., 0.05 for 5%)
    use_gradual_detection: bool
    gradual_window_minutes: int
    gradual_direction_consistency: float  # e.g., 0.7 for 70%

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
        use_gradual_detection=False,
        gradual_window_minutes=3,
        gradual_direction_consistency=0.7,
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
    ),
}


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
    # Try full name first
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
