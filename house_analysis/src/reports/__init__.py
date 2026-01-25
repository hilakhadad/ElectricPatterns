"""
Report generation module.

Contains functions to generate per-house and aggregated reports.
"""
from .house_report import analyze_single_house, generate_house_report
from .aggregate_report import aggregate_all_houses, generate_summary_report

__all__ = [
    'analyze_single_house',
    'generate_house_report',
    'aggregate_all_houses',
    'generate_summary_report',
]
