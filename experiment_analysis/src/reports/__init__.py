"""
Report generation module for experiment analysis.

Contains functions to generate per-house and aggregated experiment reports.
"""
from reports.experiment_report import (
    analyze_experiment_house,
    generate_experiment_report,
    load_experiment_data
)
from reports.aggregate_report import (
    aggregate_experiment_results,
    generate_summary_report,
    create_comparison_table,
    get_focus_houses,
    generate_monthly_analysis,
    load_pre_analysis_scores
)

__all__ = [
    'analyze_experiment_house',
    'generate_experiment_report',
    'load_experiment_data',
    'aggregate_experiment_results',
    'generate_summary_report',
    'create_comparison_table',
    'get_focus_houses',
    'generate_monthly_analysis',
    'load_pre_analysis_scores',
]
