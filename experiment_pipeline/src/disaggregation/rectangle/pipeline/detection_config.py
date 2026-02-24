"""
Detection configuration extraction.

Extracts detection parameters from an ExperimentConfig object,
providing defaults when config is None or attributes are missing.
"""


def extract_detection_params(config) -> dict:
    """
    Extract detection parameters from an ExperimentConfig.

    Returns a dict with all detection-related parameters, using defaults
    when config is None or when specific attributes are missing.

    Args:
        config: ExperimentConfig instance, or None for all defaults

    Returns:
        Dict with keys: off_threshold_factor, use_gradual, gradual_window,
        progressive_search, use_near_threshold, near_threshold_min_factor,
        near_threshold_max_extend, use_nan_imputation, use_tail_extension,
        tail_max_minutes, tail_min_residual, tail_noise_tolerance,
        tail_min_gain, tail_min_residual_fraction, use_settling_extension,
        settling_factor, settling_max_minutes, use_split_off_merger,
        split_off_max_gap_minutes
    """
    if config:
        return {
            'off_threshold_factor': config.off_threshold_factor,
            'use_gradual': config.use_gradual_detection,
            'gradual_window': config.gradual_window_minutes,
            'progressive_search': getattr(config, 'progressive_window_search', False),
            'use_near_threshold': getattr(config, 'use_near_threshold_detection', False),
            'near_threshold_min_factor': getattr(config, 'near_threshold_min_factor', 0.85),
            'near_threshold_max_extend': getattr(config, 'near_threshold_max_extend', 3),
            'use_nan_imputation': getattr(config, 'use_nan_imputation', False),
            'use_tail_extension': getattr(config, 'use_tail_extension', False),
            'tail_max_minutes': getattr(config, 'tail_max_extension_minutes', 10),
            'tail_min_residual': getattr(config, 'tail_min_residual', 100),
            'tail_noise_tolerance': getattr(config, 'tail_noise_tolerance', 30),
            'tail_min_gain': getattr(config, 'tail_min_gain', 100),
            'tail_min_residual_fraction': getattr(config, 'tail_min_residual_fraction', 0.05),
            'use_settling_extension': getattr(config, 'use_settling_extension', True),
            'settling_factor': getattr(config, 'settling_factor', 0.7),
            'settling_max_minutes': getattr(config, 'settling_max_minutes', 5),
            'use_split_off_merger': getattr(config, 'use_split_off_merger', True),
            'split_off_max_gap_minutes': getattr(config, 'split_off_max_gap_minutes', 2),
        }
    else:
        return {
            'off_threshold_factor': 1.0,
            'use_gradual': True,
            'gradual_window': 3,
            'progressive_search': False,
            'use_near_threshold': False,
            'near_threshold_min_factor': 0.85,
            'near_threshold_max_extend': 3,
            'use_nan_imputation': False,
            'use_tail_extension': False,
            'tail_max_minutes': 10,
            'tail_min_residual': 100,
            'tail_noise_tolerance': 30,
            'tail_min_gain': 100,
            'tail_min_residual_fraction': 0.05,
            'use_settling_extension': False,
            'settling_factor': 0.7,
            'settling_max_minutes': 5,
            'use_split_off_merger': False,
            'split_off_max_gap_minutes': 2,
        }


def format_config_log(params: dict) -> str:
    """Format config parameters as a log string."""
    return (
        f"Config: off_factor={params['off_threshold_factor']}, "
        f"gradual={params['use_gradual']}, "
        f"progressive={params['progressive_search']}, "
        f"near_threshold={params['use_near_threshold']}, "
        f"tail_extension={params['use_tail_extension']}, "
        f"nan_imputation={params['use_nan_imputation']}, "
        f"settling={params['use_settling_extension']}, "
        f"split_off={params['use_split_off_merger']}"
    )
