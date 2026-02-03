"""
Visualization module for house analysis reports.

Generates HTML reports with interactive charts.
"""
from visualization.html_report import generate_html_report, generate_single_house_html_report
from visualization.charts import (
    create_quality_distribution_chart,
    create_coverage_comparison_chart,
    create_phase_balance_chart,
    create_day_night_scatter,
    create_issues_heatmap,
)

__all__ = [
    'generate_html_report',
    'generate_single_house_html_report',
    'create_quality_distribution_chart',
    'create_coverage_comparison_chart',
    'create_phase_balance_chart',
    'create_day_night_scatter',
    'create_issues_heatmap',
]
