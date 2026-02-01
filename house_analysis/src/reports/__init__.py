"""
Report generation module.

Contains functions to generate per-house and aggregated reports.
"""
from reports.house_report import analyze_single_house, generate_house_report, load_house_data
from reports.aggregate_report import aggregate_all_houses, generate_summary_report, create_comparison_table

__all__ = [
    'analyze_single_house',
    'generate_house_report',
    'load_house_data',
    'aggregate_all_houses',
    'generate_summary_report',
    'create_comparison_table',
]
