"""
Metrics calculation module for experiment results.

Contains functions to calculate matching, segmentation, and performance metrics.
"""
from metrics.matching import calculate_matching_metrics
from metrics.segmentation import calculate_segmentation_metrics
from metrics.events import calculate_event_metrics
from metrics.iterations import (
    calculate_iteration_metrics,
    analyze_iteration_progression,
    get_iteration_summary
)
from metrics.patterns import (
    calculate_pattern_metrics,
    get_recurring_patterns_summary,
    find_periodic_patterns,
    detect_ac_patterns,
    detect_boiler_patterns,
    analyze_device_usage_patterns
)
from metrics.monthly import (
    calculate_monthly_metrics,
    get_monthly_summary,
    create_monthly_comparison_table,
    find_common_problematic_months
)

__all__ = [
    'calculate_matching_metrics',
    'calculate_segmentation_metrics',
    'calculate_event_metrics',
    'calculate_iteration_metrics',
    'analyze_iteration_progression',
    'get_iteration_summary',
    'calculate_pattern_metrics',
    'get_recurring_patterns_summary',
    'find_periodic_patterns',
    'detect_ac_patterns',
    'detect_boiler_patterns',
    'analyze_device_usage_patterns',
    'calculate_monthly_metrics',
    'get_monthly_summary',
    'create_monthly_comparison_table',
    'find_common_problematic_months',
]
