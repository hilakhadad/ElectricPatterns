"""
HTML report generator for experiment analysis.

Combines charts and tables into a single HTML report.

This module is a facade — actual implementations are split across:
  - html_report_aggregate.py  — multi-house aggregate report builders
  - html_report_single.py     — per-house section builders
  - html_templates.py         — CSS and HTML boilerplate
"""
import logging
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path for shared imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from shared.html_utils import assign_tier as _assign_tier

from visualization.html_report_aggregate import (
    _extract_house_data,
    _generate_summary_section,
    _generate_comparison_table,
    _create_classification_overview_chart,
    _generate_charts_section,
)
from visualization.html_report_single import (
    _generate_house_summary,
    _generate_iterations_section,
    _generate_matching_section,
    _generate_segmentation_section,
    _generate_threshold_explanation_html,
    _generate_patterns_section,
    _generate_ac_detection_html,
    _generate_boiler_detection_html,
    _generate_device_usage_html,
    _generate_monthly_breakdown_html,
    _generate_flags_section,
    _generate_house_charts,
)
from visualization.html_templates import (
    _build_html_document,
    _build_house_html_document,
)

logger = logging.getLogger(__name__)


def generate_html_report(analyses: List[Dict[str, Any]],
                         output_path: str,
                         title: str = "Experiment Analysis Report") -> str:
    """
    Generate complete HTML report from experiment analyses.

    Args:
        analyses: List of analysis results from analyze_experiment_house
        output_path: Path to save the HTML file
        title: Report title

    Returns:
        Path to the generated HTML file
    """
    logger.info("Generating aggregate HTML report for %d analyses -> %s", len(analyses), output_path)
    # Extract per-house data for JavaScript filtering
    house_data = _extract_house_data(analyses)

    # Generate all sections
    summary_html = _generate_summary_section(analyses)
    table_html = _generate_comparison_table(analyses)
    charts_html = _generate_charts_section(analyses)

    # Combine into full HTML
    html_content = _build_html_document(
        title=title,
        summary=summary_html,
        table=table_html,
        charts=charts_html,
        house_data_json=json.dumps(house_data, ensure_ascii=False),
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # Save to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info("Aggregate HTML report saved to %s", output_path)
    return output_path


def generate_house_html_report(analysis: Dict[str, Any],
                                output_path: str) -> str:
    """
    Generate HTML report for a single house.

    Args:
        analysis: Analysis result from analyze_experiment_house
        output_path: Path to save the HTML file

    Returns:
        Path to the generated HTML file
    """
    house_id = analysis.get('house_id', 'unknown')
    logger.info("Generating house HTML report for house %s -> %s", house_id, output_path)

    # Generate sections
    summary_html = _generate_house_summary(analysis)
    iterations_html = _generate_iterations_section(analysis)
    matching_html = _generate_matching_section(analysis)
    segmentation_html = _generate_segmentation_section(analysis)
    patterns_html = _generate_patterns_section(analysis)
    monthly_html = _generate_monthly_breakdown_html(analysis)
    flags_html = _generate_flags_section(analysis)
    charts_html = _generate_house_charts(analysis)

    # Add monthly breakdown to patterns section
    if monthly_html:
        patterns_html = patterns_html + monthly_html

    # Build document
    html_content = _build_house_html_document(
        house_id=house_id,
        summary=summary_html,
        iterations=iterations_html,
        matching=matching_html,
        segmentation=segmentation_html,
        patterns=patterns_html,
        flags=flags_html,
        charts=charts_html,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # Save to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info("House HTML report for house %s saved to %s", house_id, output_path)
    return output_path
