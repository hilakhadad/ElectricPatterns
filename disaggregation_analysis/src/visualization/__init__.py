"""
Visualization module for experiment analysis reports.

Generates HTML reports with interactive charts.
"""
from visualization.html_report import generate_html_report, generate_house_html_report
from visualization.charts import (
    create_score_distribution_chart,
    create_matching_rate_distribution_chart,
    create_segmentation_ratio_distribution_chart,
    create_tag_breakdown_chart,
    create_iteration_contribution_chart,
    create_issues_summary_chart,
    create_experiment_summary_table,
    create_duration_distribution_chart,
    create_pattern_detection_chart,
    create_device_detection_chart,
    # Deprecated aliases for backwards compatibility
    create_matching_comparison_chart,
    create_segmentation_chart,
    create_iteration_progress_chart,
    create_issues_heatmap,
)

__all__ = [
    'generate_html_report',
    'generate_house_html_report',
    # New chart functions
    'create_score_distribution_chart',
    'create_matching_rate_distribution_chart',
    'create_segmentation_ratio_distribution_chart',
    'create_tag_breakdown_chart',
    'create_iteration_contribution_chart',
    'create_issues_summary_chart',
    'create_experiment_summary_table',
    'create_duration_distribution_chart',
    'create_pattern_detection_chart',
    'create_device_detection_chart',
    # Deprecated aliases
    'create_matching_comparison_chart',
    'create_segmentation_chart',
    'create_iteration_progress_chart',
    'create_issues_heatmap',
]
