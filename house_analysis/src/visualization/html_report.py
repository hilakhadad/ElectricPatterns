"""
HTML report generator for house analysis.

Combines charts and tables into a single HTML report.

This module is a facade -- actual implementations are split across:
  - html_report_aggregate.py  -- multi-house aggregate report builders
  - html_report_single.py     -- per-house section builders and helpers
  - html_report_template.py   -- CSS and HTML boilerplate
"""
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from shared.html_utils import build_quality_dist_bar as _build_quality_dist_bar

from visualization.charts import (
    create_quality_distribution_chart,
    create_phase_balance_chart,
    create_issues_heatmap,
    create_hourly_pattern_chart,
    create_phase_power_chart,
    create_monthly_pattern_chart,
    create_weekly_pattern_chart,
    create_power_heatmap_chart,
    create_power_histogram,
    create_score_breakdown_chart,
    create_quality_flags_chart,
    create_mini_hourly_chart,
    create_year_hourly_chart,
    create_year_heatmap,
    create_wave_monthly_chart,
    create_wave_comparison_chart,
)

from visualization.html_report_aggregate import (
    _generate_summary_section,
    _generate_wave_section,
    _generate_comparison_table,
    _build_filter_bar,
    _generate_charts_section,
    _generate_quality_tiers_section,
)
from visualization.html_report_single import (
    generate_single_house_html_report,
    _get_month_name,
    _coverage_color,
    _coverage_bg,
    _score_color,
    _month_coverage_style,
    _format_small_pct,
    _build_zero_power_warning,
    _build_anomaly_warning,
    _build_findings_tags,
    _build_wave_behavior_section,
)
from visualization.html_report_template import _build_html_document


def generate_html_report(analyses: List[Dict[str, Any]],
                          output_path: str,
                          title: str = "House Analysis Report",
                          per_house_dir: str = '../per_house') -> str:
    """
    Generate complete HTML report from house analyses.

    Args:
        analyses: List of analysis results from analyze_single_house
        output_path: Path to save the HTML file
        title: Report title
        per_house_dir: Relative path to per-house HTML reports

    Returns:
        Path to the generated HTML file
    """
    # Generate all sections
    summary_html = _generate_summary_section(analyses)
    wave_html = _generate_wave_section(analyses)
    table_html, tier_counts, continuity_counts, wave_counts = _generate_comparison_table(analyses, per_house_dir)
    filter_bar_html = _build_filter_bar(tier_counts, continuity_counts, wave_counts)
    charts_html = _generate_charts_section(analyses)
    quality_tiers_html = _generate_quality_tiers_section(analyses)

    # Combine into full HTML
    html_content = _build_html_document(
        title=title,
        summary=summary_html,
        wave=wave_html,
        filter_bar=filter_bar_html,
        table=table_html,
        charts=charts_html,
        quality_tiers=quality_tiers_html,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # Save to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path
