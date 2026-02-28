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
    use_settling_extension: bool = False  # Extend event boundaries through transient settling periods
    settling_factor: float = 0.75  # Minimum settled/spike ratio to trigger (0.75 = >25% reversal)
    settling_max_minutes: int = 5  # Maximum minutes to look for settling
    use_split_off_merger: bool = False  # Merge split OFF events from measurement errors
    split_off_max_gap_minutes: int = 2  # Maximum gap between split OFF events
    use_guided_recovery: bool = False  # Search for missed AC cycles at lower threshold (off by default)
    guided_recovery_threshold_factor: float = 0.6  # Recovery threshold = avg_magnitude * factor
    guided_recovery_min_cycles: int = 3  # Minimum matched cycles to establish template
    use_wave_recovery: bool = False  # Post-M1: detect wave-shaped patterns (sharp rise → gradual decay)
    wave_min_rise_watts: int = 500  # Minimum sharp rise to trigger wave detection
    wave_min_duration_minutes: int = 3  # Minimum wave duration
    wave_max_duration_minutes: int = 45  # Maximum wave duration
    wave_monotonic_tolerance: float = 0.15  # Fraction of points allowed to be non-monotonic
    wave_min_decay_fraction: float = 0.3  # Must decay at least 30% from peak to qualify
    use_complementary_off_matching: bool = False  # Stage 4: merge two OFF events that together match an ON
    complementary_off_max_gap_minutes: int = 10  # Max gap between the two OFF events
    use_normalization: bool = False  # Apply normalization preprocessing before pipeline starts
    normalization_method: str = 'none'  # 'ma_detrend', 'phase_balance', 'mad_clean', 'combined'
    normalization_params: Optional[dict] = None  # Method-specific parameters (e.g., window size)

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
            'use_settling_extension': self.use_settling_extension,
            'settling_factor': self.settling_factor,
            'settling_max_minutes': self.settling_max_minutes,
            'use_split_off_merger': self.use_split_off_merger,
            'split_off_max_gap_minutes': self.split_off_max_gap_minutes,
            'use_guided_recovery': self.use_guided_recovery,
            'guided_recovery_threshold_factor': self.guided_recovery_threshold_factor,
            'guided_recovery_min_cycles': self.guided_recovery_min_cycles,
            'use_wave_recovery': self.use_wave_recovery,
            'wave_min_rise_watts': self.wave_min_rise_watts,
            'wave_min_duration_minutes': self.wave_min_duration_minutes,
            'wave_max_duration_minutes': self.wave_max_duration_minutes,
            'wave_monotonic_tolerance': self.wave_monotonic_tolerance,
            'wave_min_decay_fraction': self.wave_min_decay_fraction,
            'use_complementary_off_matching': self.use_complementary_off_matching,
            'complementary_off_max_gap_minutes': self.complementary_off_max_gap_minutes,
            'use_normalization': self.use_normalization,
            'normalization_method': self.normalization_method,
            'normalization_params': self.normalization_params,
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
        gradual_window_minutes=5,
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
        gradual_window_minutes=5,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
        use_near_threshold_detection=False,
        use_tail_extension=True,
        threshold_schedule=[2000, 1500, 1100, 800],
        use_nan_imputation=True,
    ),

    'exp013_settling_splitoff': ExperimentConfig(
        exp_id='exp013',
        description='exp012 + settling extension + split-OFF merger (fixes pits, split shutdowns, improves matching)',
        threshold=2000,
        off_threshold_factor=1.0,
        expand_event_factor=0.2,
        use_gradual_detection=True,
        gradual_window_minutes=5,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
        use_near_threshold_detection=False,
        use_tail_extension=True,
        threshold_schedule=[2000, 1500, 1100, 800],
        use_nan_imputation=True,
        use_settling_extension=True,
        use_split_off_merger=True,
        use_guided_recovery=True,
        use_complementary_off_matching=True,
    ),

    'exp014_wave_recovery': ExperimentConfig(
        exp_id='exp014',
        description='exp013 + post-M1 wave recovery (detects gradual-decay compressor cycles missed by rectangle matching)',
        threshold=2000,
        off_threshold_factor=1.0,
        expand_event_factor=0.2,
        use_gradual_detection=True,
        gradual_window_minutes=5,
        gradual_direction_consistency=0.7,
        progressive_window_search=True,
        use_near_threshold_detection=False,
        use_tail_extension=True,
        threshold_schedule=[2000, 1500, 1100, 800],
        use_nan_imputation=True,
        use_settling_extension=True,
        use_split_off_merger=True,
        use_guided_recovery=True,
        use_wave_recovery=True,
        use_complementary_off_matching=True,
    ),

    'exp015_hole_repair': ExperimentConfig(
        exp_id='exp015',
        description='exp014 + hole repair (fixes rectangle matches that extracted wave-shaped patterns as flat rectangles)',
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
        use_settling_extension=True,
        use_split_off_merger=True,
        use_guided_recovery=True,
        use_wave_recovery=True,
        use_complementary_off_matching=True,
    ),

    'exp016_ma_detrend': ExperimentConfig(
        exp_id='exp016',
        description='exp015 + 2-hour moving average detrending (removes baseline drift, preserves watt scale)',
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
        use_settling_extension=True,
        use_split_off_merger=True,
        use_guided_recovery=True,
        use_wave_recovery=True,
        use_complementary_off_matching=True,
        use_normalization=True,
        normalization_method='ma_detrend',
        normalization_params={'ma_detrend': {'window_minutes': 120}},
    ),

    'exp017_phase_balance': ExperimentConfig(
        exp_id='exp017',
        description='exp015 + phase balancing (equalizes 3-phase baselines to common median)',
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
        use_settling_extension=True,
        use_split_off_merger=True,
        use_guided_recovery=True,
        use_wave_recovery=True,
        use_complementary_off_matching=True,
        use_normalization=True,
        normalization_method='phase_balance',
    ),

    'exp018_mad_clean': ExperimentConfig(
        exp_id='exp018',
        description='exp015 + MAD-based outlier cleaning (removes measurement spikes, preserves real events)',
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
        use_settling_extension=True,
        use_split_off_merger=True,
        use_guided_recovery=True,
        use_wave_recovery=True,
        use_complementary_off_matching=True,
        use_normalization=True,
        normalization_method='mad_clean',
        normalization_params={'mad_clean': {'window_minutes': 240, 'k': 5.0}},
    ),

    'exp019_combined_norm': ExperimentConfig(
        exp_id='exp019',
        description='exp015 + combined normalization (MA detrend → phase balance → MAD clean)',
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
        use_settling_extension=True,
        use_split_off_merger=True,
        use_guided_recovery=True,
        use_wave_recovery=True,
        use_complementary_off_matching=True,
        use_normalization=True,
        normalization_method='combined',
        normalization_params={
            'ma_detrend': {'window_minutes': 120},
            'mad_clean': {'window_minutes': 240, 'k': 5.0},
        },
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


# Default experiment —
DEFAULT_EXPERIMENT = 'exp015_hole_repair'


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
