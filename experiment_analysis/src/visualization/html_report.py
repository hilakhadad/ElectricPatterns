"""
HTML report generator for experiment analysis.

Combines charts and tables into a single HTML report.
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

from visualization.charts import (
    create_score_distribution_chart,
    create_matching_rate_distribution_chart,
    create_segmentation_ratio_distribution_chart,
    create_tag_breakdown_chart,
    create_experiment_summary_table,
    create_duration_distribution_chart,
    create_device_detection_chart,
)


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
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

    # Save to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


def _generate_summary_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate executive summary HTML."""
    import numpy as np
    valid = [a for a in analyses if a.get('status') != 'no_data']
    n_houses = len(valid)

    if n_houses == 0:
        return '<p>No experiment data found.</p>'

    # Calculate averages
    overall_scores = [a.get('scores', {}).get('overall_score', 0) for a in valid]
    matching_scores = [a.get('scores', {}).get('matching_score', 0) for a in valid]
    seg_ratios = [a.get('first_iteration', {}).get('segmentation', {}).get('segmentation_ratio', 0) for a in valid]
    total_matches = sum(a.get('first_iteration', {}).get('matching', {}).get('total_matches', 0) for a in valid)

    # Threshold explanation rates
    th_explanation_rates = [a.get('threshold_explanation', {}).get('total_explanation_rate', 0) for a in valid]

    avg_overall = sum(overall_scores) / n_houses if n_houses > 0 else 0
    avg_matching = sum(matching_scores) / n_houses if n_houses > 0 else 0
    avg_seg = sum(seg_ratios) / n_houses if n_houses > 0 else 0
    avg_th_expl = np.mean(th_explanation_rates) if th_explanation_rates else 0
    std_th_expl = np.std(th_explanation_rates) if th_explanation_rates else 0

    # Count issues
    n_low_matching = sum(1 for a in valid if a.get('flags', {}).get('low_matching_rate', False))
    n_negative = sum(1 for a in valid if a.get('flags', {}).get('has_negative_values', False))
    n_low_seg = sum(1 for a in valid if a.get('flags', {}).get('low_segmentation', False))

    # Color for threshold explanation rate
    th_color = '#28a745' if avg_th_expl >= 0.8 else '#ffc107' if avg_th_expl >= 0.5 else '#dc3545'

    return f"""
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-number">{n_houses}</div>
            <div class="summary-label">Houses Analyzed</div>
        </div>
        <div class="summary-card" style="border: 2px solid {th_color}; background: linear-gradient(135deg, #fff 0%, {th_color}22 100%);">
            <div class="summary-number" style="color: {th_color};">{avg_th_expl:.1%}</div>
            <div class="summary-label">High-Power Energy Explained</div>
            <div style="font-size: 0.8em; color: #666;">(&gt;1300W) \u00b1{std_th_expl:.1%} std</div>
        </div>
        <div class="summary-card">
            <div class="summary-number">{avg_overall:.0f}</div>
            <div class="summary-label">Avg Overall Score</div>
        </div>
        <div class="summary-card">
            <div class="summary-number">{avg_matching:.0f}%</div>
            <div class="summary-label">Avg Matching Score</div>
        </div>
        <div class="summary-card">
            <div class="summary-number">{avg_seg:.1%}</div>
            <div class="summary-label">Avg Segmentation</div>
        </div>
    </div>
    <div class="summary-alerts">
        <h3>Issues Summary</h3>
        <ul>
            <li class="{'alert' if n_low_matching > 0 else ''}">
                <strong>{n_low_matching}</strong> houses with low matching rate (&lt;50%)
            </li>
            <li class="{'alert' if n_low_seg > 0 else ''}">
                <strong>{n_low_seg}</strong> houses with low segmentation (&lt;30%)
            </li>
            <li class="{'alert' if n_negative > 0 else ''}">
                <strong>{n_negative}</strong> houses with negative power values
            </li>
        </ul>
    </div>
    """


def _generate_comparison_table(analyses: List[Dict[str, Any]]) -> str:
    """Generate sortable comparison table HTML."""
    rows = []

    for a in analyses:
        if a.get('status') == 'no_data':
            continue

        house_id = a.get('house_id', 'unknown')
        iterations = a.get('iterations', {})
        scores = a.get('scores', {})
        first = a.get('first_iteration', {})
        matching = first.get('matching', {})
        seg = first.get('segmentation', {})
        flags = a.get('flags', {})
        patterns = first.get('patterns', {})

        # Device detection
        ac_detection = patterns.get('ac_detection', {})
        boiler_detection = patterns.get('boiler_detection', {})

        has_central_ac = ac_detection.get('has_central_ac', False)
        has_regular_ac = bool(ac_detection.get('regular_ac_by_phase', {}) or ac_detection.get('regular_ac', {}).get('activations'))
        has_boiler = boiler_detection.get('has_boiler', False)

        # Device icons
        central_ac_icon = '<span title="Central AC">❄️</span>' if has_central_ac else ''
        regular_ac_icon = '<span title="Regular AC">🌀</span>' if has_regular_ac else ''
        boiler_icon = '<span title="Boiler">🔥</span>' if has_boiler else ''
        devices = f"{central_ac_icon} {regular_ac_icon} {boiler_icon}".strip() or '-'

        # Flag icons - show which flags are active
        flag_icons = []
        if flags.get('low_matching_rate'):
            flag_icons.append('<span title="Low Matching Rate (<50%)">📉</span>')
        if flags.get('has_negative_values'):
            flag_icons.append('<span title="Negative Power Values">⚡</span>')
        if flags.get('low_segmentation'):
            flag_icons.append('<span title="Low Segmentation (<30%)">📊</span>')
        if flags.get('has_damaged_phases'):
            flag_icons.append('<span title="Damaged Phase(s)">🔌</span>')
        if flags.get('has_recurring_patterns'):
            flag_icons.append('<span title="Has Recurring Patterns">🔄</span>')
        flags_display = ' '.join(flag_icons) if flag_icons else '-'

        # Calculate minutes segmentation ratio (NOT x3 - real time only)
        matched_minutes = matching.get('total_matched_minutes', 0) or 0
        total_days = patterns.get('daily_stats', {}).get('total_days', 0) or 0
        if total_days > 0 and matched_minutes > 0:
            total_available_minutes = total_days * 24 * 60  # Real time, NOT x3
            minutes_seg_ratio = matched_minutes / total_available_minutes
        else:
            minutes_seg_ratio = 0

        # Threshold explanation rate
        th_expl = a.get('threshold_explanation', {})
        th_explanation_rate = th_expl.get('total_explanation_rate', 0) or 0

        # Score badge - check for damaged phases first
        score = scores.get('overall_score', 0) or 0
        has_damaged = flags.get('has_damaged_phases', False)

        if has_damaged:
            damaged_phases = scores.get('damaged_phases', [])
            badge = f'<span class="badge badge-purple">Damaged ({",".join(damaged_phases)})</span>'
        elif score >= 80:
            badge = '<span class="badge badge-green">Excellent</span>'
        elif score >= 60:
            badge = '<span class="badge badge-blue">Good</span>'
        elif score >= 40:
            badge = '<span class="badge badge-orange">Fair</span>'
        else:
            badge = '<span class="badge badge-red">Poor</span>'

        # Color for explanation rate
        th_color = '#28a745' if th_explanation_rate >= 0.8 else '#ffc107' if th_explanation_rate >= 0.5 else '#dc3545'

        # Pre-analysis quality score (from house_analysis)
        pre_quality = a.get('pre_analysis_quality_score', None)
        if pre_quality is not None:
            pre_q_color = '#28a745' if pre_quality >= 75 else '#ffc107' if pre_quality >= 50 else '#dc3545'
            pre_quality_html = f'<span style="color: {pre_q_color}; font-weight: bold;">{pre_quality:.0f}</span>'
        else:
            pre_quality_html = '<span style="color: #999;">-</span>'

        rows.append(f"""
        <tr>
            <td><a href="house_reports/house_{house_id}.html" target="_blank" style="text-decoration: none; color: #1976d2;"><strong>{house_id}</strong></a></td>
            <td>{pre_quality_html}</td>
            <td>{iterations.get('iterations_completed', 0) or 0}</td>
            <td>{(iterations.get('first_iter_matching_rate', 0) or 0):.1%}</td>
            <td>{(seg.get('segmentation_ratio', 0) or 0):.1%}</td>
            <td>{minutes_seg_ratio:.2%}</td>
            <td style="color: {th_color}; font-weight: bold;">{th_explanation_rate:.1%}</td>
            <td style="font-size: 1.2em;">{devices}</td>
            <td>{score:.0f} {badge}</td>
            <td style="font-size: 1.2em;">{flags_display}</td>
        </tr>
        """)

    return f"""
    <p style="font-size: 0.85em; color: #666; margin-bottom: 10px;">
        <strong>Pre-Quality</strong> = quality score from pre-experiment analysis (coverage+days+data quality) |
        <strong>Match Rate</strong> = % of events matched |
        <strong>High-Power Explained</strong> = % of minutes &gt;1300W with segregation
    </p>
    <p style="font-size: 0.85em; color: #666; margin-bottom: 10px;">
        <strong>Devices:</strong> ❄️ Central AC | 🌀 Regular AC | 🔥 Boiler |
        <strong>Issues:</strong> 📉 Low Matching | ⚡ Negatives | 📊 Low Seg | 🔌 Damaged | 🔄 Recurring
    </p>
    <table class="data-table" id="comparison-table">
        <thead>
            <tr>
                <th onclick="sortTable(0)">House ID</th>
                <th onclick="sortTable(1)">Pre-Quality<br><small>(0-100)</small></th>
                <th onclick="sortTable(2)">Iterations</th>
                <th onclick="sortTable(3)">Match Rate<br><small>(events)</small></th>
                <th onclick="sortTable(4)">Segmentation<br><small>(power)</small></th>
                <th onclick="sortTable(5)">Segmentation<br><small>(minutes)</small></th>
                <th onclick="sortTable(6)">High-Power<br><small>Explained</small></th>
                <th onclick="sortTable(7)">Devices</th>
                <th onclick="sortTable(8)">Exp Score</th>
                <th onclick="sortTable(9)">Flags</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def _generate_charts_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate all charts HTML."""

    # Start with summary table (most important)
    summary_table = create_experiment_summary_table(analyses)

    # Organize charts by category (5 charts total)
    charts_by_category = {
        'Matching & Segmentation': [
            create_matching_rate_distribution_chart(analyses),
            create_segmentation_ratio_distribution_chart(analyses),
        ],
        'Match Details': [
            create_tag_breakdown_chart(analyses),
            create_duration_distribution_chart(analyses),
        ],
        'Patterns & Devices': [
            create_device_detection_chart(analyses),
        ],
    }

    html_parts = []

    # Summary table first (full width)
    html_parts.append(f"""
    <div class="summary-section">
        {summary_table}
    </div>
    """)

    # Charts in grid layout by category
    for category, chart_list in charts_by_category.items():
        html_parts.append(f"""
        <h3 style="margin-top: 30px; color: #2c3e50; border-bottom: 2px solid #667eea; padding-bottom: 10px;">
            {category}
        </h3>
        <div class="charts-row" style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: flex-start;">
        """)

        for chart_html in chart_list:
            if chart_html:  # Skip empty charts
                html_parts.append(f"""
            <div class="chart-container" style="flex: 0 1 auto; min-width: 400px; max-width: 600px;">
                <div class="chart-content">
                    {chart_html}
                </div>
            </div>
                """)

        html_parts.append("</div>")

    return '\n'.join(html_parts)


def _build_html_document(title: str, summary: str, table: str,
                         charts: str, generated_at: str) -> str:
    """Build complete HTML document."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}

        header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}

        header .subtitle {{
            opacity: 0.9;
        }}

        section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        section h2 {{
            color: #444;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}

        /* Summary cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }}

        .summary-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }}

        .summary-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2d3748;
        }}

        .summary-label {{
            color: #666;
            margin-top: 5px;
        }}

        .summary-alerts ul {{
            list-style: none;
            padding: 0;
        }}

        .summary-alerts li {{
            padding: 10px 15px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #ddd;
        }}

        .summary-alerts li.alert {{
            border-left-color: #e74c3c;
            background: #fdf2f2;
        }}

        /* Data table */
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}

        .data-table th {{
            background: #2d3748;
            color: white;
            padding: 12px 15px;
            text-align: left;
            cursor: pointer;
            user-select: none;
        }}

        .data-table th:hover {{
            background: #4a5568;
        }}

        .data-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}

        .data-table tr:hover {{
            background: #f8f9fa;
        }}

        /* Badges */
        .badge {{
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 5px;
        }}

        .badge-purple {{ background: #e2d5f0; color: #6f42c1; }}
        .badge-green {{ background: #d4edda; color: #155724; }}
        .badge-blue {{ background: #cce5ff; color: #004085; }}
        .badge-orange {{ background: #fff3cd; color: #856404; }}
        .badge-red {{ background: #f8d7da; color: #721c24; }}

        /* House links in table */
        .house-link {{
            color: #667eea;
            text-decoration: none;
            transition: color 0.2s;
        }}

        .house-link:hover {{
            color: #764ba2;
            text-decoration: underline;
        }}

        /* Chart containers */
        .chart-container {{
            margin-bottom: 30px;
        }}

        .chart-content {{
            min-height: 400px;
        }}

        /* Score tiers */
        .tiers-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}

        .tier-card {{
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid;
        }}

        .tier-purple {{ background: #e2d5f0; border-color: #6f42c1; }}
        .tier-green {{ background: #d4edda; border-color: #28a745; }}
        .tier-blue {{ background: #cce5ff; border-color: #007bff; }}
        .tier-orange {{ background: #fff3cd; border-color: #ffc107; }}
        .tier-red {{ background: #f8d7da; border-color: #dc3545; }}

        .tier-count {{
            font-size: 1.5em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .tier-houses {{
            font-size: 0.9em;
            color: #666;
            word-break: break-word;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            header {{ padding: 20px; }}
            section {{ padding: 15px; }}
            .data-table {{ font-size: 0.8em; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="subtitle">Generated: {generated_at}</div>
        </header>

        <section>
            <h2>Executive Summary</h2>
            {summary}
        </section>

        <section>
            <h2>House Comparison</h2>
            {table}
        </section>

        <section>
            <h2>Analysis Charts</h2>
            {charts}
        </section>

        <footer>
            Experiment Analysis Report | Generated by experiment_analysis module
        </footer>
    </div>

    <script>
        function sortTable(n) {{
            var table = document.getElementById("comparison-table");
            var rows = Array.from(table.rows).slice(1);
            var ascending = table.getAttribute('data-sort-col') !== String(n) ||
                           table.getAttribute('data-sort-dir') !== 'asc';

            rows.sort(function(a, b) {{
                var aVal = a.cells[n].textContent.trim();
                var bVal = b.cells[n].textContent.trim();

                // Try numeric comparison
                var aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                var bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return ascending ? aNum - bNum : bNum - aNum;
                }}

                // Fall back to string comparison
                return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});

            var tbody = table.getElementsByTagName('tbody')[0];
            rows.forEach(function(row) {{ tbody.appendChild(row); }});

            table.setAttribute('data-sort-col', n);
            table.setAttribute('data-sort-dir', ascending ? 'asc' : 'desc');
        }}
    </script>
</body>
</html>
"""


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

    return output_path


def _generate_house_summary(analysis: Dict[str, Any]) -> str:
    """Generate summary section for single house."""
    scores = analysis.get('scores', {})
    iterations = analysis.get('iterations', {})
    first = analysis.get('first_iteration', {})
    matching = first.get('matching', {})
    seg = first.get('segmentation', {})
    flags = analysis.get('flags', {})
    th_expl = analysis.get('threshold_explanation', {})

    overall_score = scores.get('overall_score', 0) or 0
    matching_score = scores.get('matching_score', 0) or 0
    seg_score = scores.get('segmentation_score', 0) or 0
    has_damaged = flags.get('has_damaged_phases', False)
    damaged_phases = scores.get('damaged_phases', []) or []
    pre_quality = analysis.get('pre_analysis_quality_score', None)

    # Threshold explanation metrics
    th_explanation_rate = th_expl.get('total_explanation_rate', 0) or 0
    th_minutes_above = th_expl.get('total_minutes_above_th', 0) or 0
    th_minutes_explained = th_expl.get('total_minutes_explained', 0) or 0
    th_color = '#28a745' if th_explanation_rate >= 0.8 else '#ffc107' if th_explanation_rate >= 0.5 else '#dc3545'

    # Score badge - check damaged first
    if has_damaged:
        badge_class = 'badge-purple'
        badge_text = f'Damaged ({",".join(damaged_phases)})'
    elif overall_score >= 80:
        badge_class = 'badge-green'
        badge_text = 'Excellent'
    elif overall_score >= 60:
        badge_class = 'badge-blue'
        badge_text = 'Good'
    elif overall_score >= 40:
        badge_class = 'badge-orange'
        badge_text = 'Fair'
    else:
        badge_class = 'badge-red'
        badge_text = 'Poor'

    # Calculate minutes segmented ratio
    total_minutes = matching.get('total_matched_minutes', 0) or 0

    return f"""
    <div class="th-explanation-highlight" style="background: linear-gradient(135deg, #fff 0%, {th_color}22 100%); border: 2px solid {th_color}; border-radius: 12px; padding: 20px; margin-bottom: 20px; text-align: center;">
        <h3 style="margin: 0 0 10px 0; color: #333;">High-Power Energy Explained (&gt;1300W)</h3>
        <div style="display: flex; justify-content: center; gap: 50px; flex-wrap: wrap;">
            <div>
                <div style="font-size: 2.5em; font-weight: bold; color: {th_color};">{th_explanation_rate:.1%}</div>
                <div style="color: #666;">Explained Rate</div>
            </div>
            <div>
                <div style="font-size: 1.5em; font-weight: bold; color: #333;">{th_minutes_above:,}</div>
                <div style="color: #666;">Minutes &gt;1300W</div>
            </div>
            <div>
                <div style="font-size: 1.5em; font-weight: bold; color: {th_color};">{th_minutes_explained:,}</div>
                <div style="color: #666;">Minutes Explained</div>
            </div>
        </div>
    </div>
    <div class="summary-grid">
        <div class="summary-card" style="background: #f0f0f0;">
            <div class="summary-number" style="color: {'#28a745' if pre_quality and pre_quality >= 75 else '#ffc107' if pre_quality and pre_quality >= 50 else '#dc3545' if pre_quality else '#999'};">{pre_quality:.0f if pre_quality else '-'}</div>
            <div class="summary-label">Pre-Quality Score</div>
            <div style="font-size: 0.7em; color: #888;">(from house analysis)</div>
        </div>
        <div class="summary-card highlight">
            <div class="summary-number">{overall_score:.0f}</div>
            <div class="summary-label">Experiment Score</div>
            <span class="badge {badge_class}">{badge_text}</span>
        </div>
        <div class="summary-card">
            <div class="summary-number">{matching_score:.0f}</div>
            <div class="summary-label">Matching Score</div>
        </div>
        <div class="summary-card">
            <div class="summary-number">{seg_score:.0f}</div>
            <div class="summary-label">Segmentation Score</div>
        </div>
        <div class="summary-card">
            <div class="summary-number">{iterations.get('iterations_completed', 0) or 0}</div>
            <div class="summary-label">Iterations</div>
        </div>
    </div>
    <div class="segmentation-summary" style="display: flex; gap: 40px; justify-content: center; margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
        <div style="text-align: center;">
            <div style="font-size: 1.5em; font-weight: bold; color: #28a745;">{(seg.get('segmentation_ratio', 0) or 0):.1%}</div>
            <div style="color: #666; font-size: 0.9em;">Power Segmented</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 1.5em; font-weight: bold; color: #17a2b8;">{total_minutes/60:.1f} hrs</div>
            <div style="color: #666; font-size: 0.9em;">Total Matched Duration</div>
            <div style="color: #999; font-size: 0.75em;">(sum across all 3 phases)</div>
        </div>
    </div>
    """


def _generate_iterations_section(analysis: Dict[str, Any]) -> str:
    """Generate iterations progress section."""
    iterations = analysis.get('iterations', {})
    iter_data = iterations.get('iterations_data', [])

    if not iter_data:
        return '<p>No iteration data available.</p>'

    rows = []
    for i, d in enumerate(iter_data):
        total_ev = d.get('total_events', 0) or 0
        on_events = d.get('on_events', total_ev // 2) or 0
        off_events = d.get('off_events', total_ev // 2) or 0
        matches = d.get('total_matches', 0) or 0
        unmatched_on = d.get('unmatched_on', on_events - matches) or 0
        unmatched_off = d.get('unmatched_off', off_events - matches) or 0
        matched_minutes = d.get('matched_minutes', 0) or 0
        matching_rate = d.get('matching_rate', 0) or 0
        matched_power = (d.get('matched_power', 0) or 0) / 1000
        neg_values = d.get('negative_values', 0) or 0

        rows.append(f"""
        <tr>
            <td>{i}</td>
            <td>ON: {on_events}<br>OFF: {off_events}</td>
            <td>{matches}</td>
            <td>ON: {unmatched_on}<br>OFF: {unmatched_off}</td>
            <td>{matching_rate:.1%}</td>
            <td>{matched_power:.1f} kW</td>
            <td>{matched_minutes:.0f} min</td>
            <td>{neg_values}</td>
        </tr>
        """)

    total_minutes = iterations.get('total_matched_minutes', 0) or 0

    # High-Power explanation per iteration table
    th_per_iter = analysis.get('threshold_explanation_per_iteration', []) or []
    th_rows = []
    for th in th_per_iter:
        iteration = th.get('iteration', 0) or 0
        above = th.get('total_minutes_above_th', 0) or 0
        explained = th.get('total_minutes_explained', 0) or 0
        rate = th.get('total_explanation_rate', 0) or 0
        th_color = '#28a745' if rate >= 0.8 else '#ffc107' if rate >= 0.5 else '#dc3545'

        th_rows.append(f"""
        <tr>
            <td>{iteration}</td>
            <td>{above:,}</td>
            <td>{explained:,}</td>
            <td style="color: {th_color}; font-weight: bold;">{rate:.1%}</td>
        </tr>
        """)

    threshold = th_per_iter[0].get('threshold', 1300) if th_per_iter else 1300
    th_table_html = f"""
    <h4 style="margin-top: 20px; color: #2c3e50;">High-Power Energy Explained per Iteration (&gt;{threshold}W)</h4>
    <table class="data-table" style="max-width: 500px;">
        <thead>
            <tr>
                <th>Iter</th>
                <th>Minutes &gt;{threshold}W</th>
                <th>Minutes Explained</th>
                <th>Rate</th>
            </tr>
        </thead>
        <tbody>
            {''.join(th_rows)}
        </tbody>
    </table>
    """ if th_rows else ""

    return f"""
    <table class="data-table">
        <thead>
            <tr>
                <th>Iter</th>
                <th>Events<br><small>(ON/OFF)</small></th>
                <th>Matches</th>
                <th>Unmatched<br><small>(ON/OFF)</small></th>
                <th>Match Rate</th>
                <th>Matched Power</th>
                <th>Matched Minutes</th>
                <th>Negative</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    <div class="metrics-row">
        <div class="metric">
            <span class="metric-label">Events Reduction:</span>
            <span class="metric-value">{(iterations.get('events_reduction_ratio', 0) or 0):.1%}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Matched Power:</span>
            <span class="metric-value">{(iterations.get('total_matched_power', 0) or 0)/1000:.1f} kW</span>
        </div>
        <div class="metric">
            <span class="metric-label">Total Matched Minutes:</span>
            <span class="metric-value">{total_minutes:.0f} min ({total_minutes/60:.1f} hrs)</span>
        </div>
    </div>
    <p style="font-size: 0.8em; color: #888; margin-top: 8px; text-align: center;">
        Note: Minutes and power are summed across all 3 phases. Overlapping events on different phases are counted separately.
    </p>
    {th_table_html}
    """


def _generate_matching_section(analysis: Dict[str, Any]) -> str:
    """Generate matching details section."""
    first = analysis.get('first_iteration', {})
    matching = first.get('matching', {})

    if not matching:
        return '<p>No matching data available.</p>'

    tag_breakdown = matching.get('tag_breakdown', {})
    phase_breakdown = matching.get('phase_breakdown', {})

    # Tag breakdown table
    tag_rows = ''.join(f"<tr><td>{tag}</td><td>{count}</td></tr>"
                       for tag, count in tag_breakdown.items())

    # Phase breakdown table
    phase_rows = ''.join(f"<tr><td>{phase}</td><td>{count}</td></tr>"
                         for phase, count in phase_breakdown.items())

    # Calculate stats
    total_on = matching.get('total_on_events', 0) or 0
    total_off = matching.get('total_off_events', 0) or 0
    matched_on = matching.get('matched_on_count', 0) or 0
    matched_off = matching.get('matched_off_count', matched_on) or 0  # Usually same as ON
    unmatched_on = matching.get('unmatched_on_count', total_on - matched_on) or 0
    unmatched_off = matching.get('unmatched_off_count', total_off - matched_off) or 0
    on_rate = matched_on / total_on if total_on > 0 else 0
    off_rate = matched_off / total_off if total_off > 0 else 0

    return f"""
    <div class="two-columns">
        <div class="column">
            <h4>Match Types</h4>
            <table class="data-table small">
                <thead><tr><th>Tag</th><th>Count</th></tr></thead>
                <tbody>{tag_rows}</tbody>
            </table>
        </div>
        <div class="column">
            <h4>By Phase</h4>
            <table class="data-table small">
                <thead><tr><th>Phase</th><th>Count</th></tr></thead>
                <tbody>{phase_rows}</tbody>
            </table>
        </div>
    </div>

    <h4 style="margin-top: 20px;">Event Statistics</h4>
    <table class="data-table small" style="max-width: 600px;">
        <thead>
            <tr>
                <th></th>
                <th>ON Events</th>
                <th>OFF Events</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Total</strong></td>
                <td>{total_on}</td>
                <td>{total_off}</td>
            </tr>
            <tr>
                <td><strong>Matched</strong></td>
                <td>{matched_on} ({on_rate:.1%})</td>
                <td>{matched_off} ({off_rate:.1%})</td>
            </tr>
            <tr>
                <td><strong>Unmatched</strong></td>
                <td>{unmatched_on}</td>
                <td>{unmatched_off}</td>
            </tr>
        </tbody>
    </table>
    <p style="font-size: 0.85em; color: #666; margin-top: 10px;">
        Note: ON and OFF event counts may differ due to detection algorithm sensitivity and edge effects.
        Matching rate = matched / total events of that type.
    </p>
    """


def _generate_segmentation_section(analysis: Dict[str, Any]) -> str:
    """Generate segmentation details section - shows both power and minutes metrics."""
    first = analysis.get('first_iteration', {})
    seg = first.get('segmentation', {})
    matching = first.get('matching', {})
    patterns = first.get('patterns', {})

    if not seg:
        return '<p>No segmentation data available.</p>'

    neg_count = seg.get('negative_value_count', 0)
    neg_warning = f'<div class="warning" style="margin-bottom: 15px;">⚠ Warning: {neg_count} negative values detected in remaining power!</div>' if neg_count > 0 else ''

    # Power metrics - ensure no None values
    total_power = (seg.get('total_power', 0) or 0) / 1000
    segmented_power = (seg.get('total_segmented_power', 0) or 0) / 1000
    remaining_power = (seg.get('total_remaining_power', 0) or 0) / 1000
    power_seg_ratio = seg.get('segmentation_ratio', 0) or 0

    # Minutes metrics
    # matched_minutes is sum across all 3 phases
    # total_available is real time (NOT multiplied by 3)
    matched_minutes = matching.get('total_matched_minutes', 0) or 0
    total_days = (patterns.get('daily_stats', {}) or {}).get('total_days', 0) or 0

    if total_days > 0:
        total_available_minutes = total_days * 24 * 60  # Real time, NOT x3
        minutes_seg_ratio = matched_minutes / total_available_minutes if total_available_minutes > 0 else 0
    else:
        total_available_minutes = 0
        minutes_seg_ratio = 0

    return f"""
    {neg_warning}
    <h4>Power Segmentation</h4>
    <div class="summary-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 20px;">
        <div class="summary-card">
            <div class="summary-number" style="color: #333; font-size: 1.8em;">{total_power:.1f} kW</div>
            <div class="summary-label">Total Power</div>
        </div>
        <div class="summary-card" style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);">
            <div class="summary-number" style="color: #155724; font-size: 1.8em;">{segmented_power:.1f} kW</div>
            <div class="summary-label">Segmented ({power_seg_ratio:.1%})</div>
        </div>
        <div class="summary-card" style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);">
            <div class="summary-number" style="color: #6c757d; font-size: 1.8em;">{remaining_power:.1f} kW</div>
            <div class="summary-label">Remaining ({1-power_seg_ratio:.1%})</div>
        </div>
    </div>

    <h4>Minutes Segmentation</h4>
    <div class="summary-grid" style="grid-template-columns: repeat(2, 1fr);">
        <div class="summary-card">
            <div class="summary-number" style="color: #333; font-size: 1.8em;">{total_available_minutes/60:.0f} hrs</div>
            <div class="summary-label">Total Time Available</div>
            <div style="font-size: 0.75em; color: #888;">{total_days} days × 24 hrs</div>
        </div>
        <div class="summary-card" style="background: linear-gradient(135deg, #cce5ff 0%, #b8daff 100%);">
            <div class="summary-number" style="color: #004085; font-size: 1.8em;">{matched_minutes/60:.0f} hrs</div>
            <div class="summary-label">Matched ({minutes_seg_ratio:.1%})</div>
            <div style="font-size: 0.75em; color: #666;">{matched_minutes:.0f} min (sum of 3 phases)</div>
        </div>
    </div>
    <p style="font-size: 0.85em; color: #888; margin-top: 10px; text-align: center;">
        Note: Matched minutes are summed across all 3 phases. Percentage can exceed 100% if events overlap.
    </p>
    {_generate_threshold_explanation_html(analysis)}
    """


def _generate_threshold_explanation_html(analysis: Dict[str, Any]) -> str:
    """Generate threshold explanation section."""
    th_expl = analysis.get('threshold_explanation', {})
    if not th_expl or 'total_minutes_above_th' not in th_expl:
        return ''

    threshold = th_expl.get('threshold', 500) or 500
    total_above = th_expl.get('total_minutes_above_th', 0) or 0
    total_explained = th_expl.get('total_minutes_explained', 0) or 0
    total_rate = th_expl.get('total_explanation_rate', 0) or 0

    # Per-phase data
    phase_rows = []
    for phase in ['w1', 'w2', 'w3']:
        above = th_expl.get(f'{phase}_minutes_above_th', 0) or 0
        explained = th_expl.get(f'{phase}_minutes_explained', 0) or 0
        rate = th_expl.get(f'{phase}_explanation_rate', 0) or 0
        color = '#28a745' if rate >= 0.8 else '#ffc107' if rate >= 0.5 else '#dc3545'
        phase_rows.append(f"""
            <tr>
                <td style="padding: 8px 12px;"><strong>{phase.upper()}</strong></td>
                <td style="padding: 8px 12px; text-align: right;">{above:,}</td>
                <td style="padding: 8px 12px; text-align: right;">{explained:,}</td>
                <td style="padding: 8px 12px; text-align: right; color: {color}; font-weight: bold;">{rate:.1%}</td>
            </tr>
        """)

    total_color = '#28a745' if total_rate >= 0.8 else '#ffc107' if total_rate >= 0.5 else '#dc3545'

    return f"""
    <h4 style="margin-top: 25px;">Threshold Explanation (&gt;{threshold}W)</h4>
    <p style="font-size: 0.85em; color: #666; margin-bottom: 15px;">
        Minutes where power exceeded {threshold}W and were explained by segregation (short/medium/long events).
    </p>
    <div class="summary-grid" style="grid-template-columns: repeat(3, 1fr); margin-bottom: 15px;">
        <div class="summary-card">
            <div class="summary-number" style="color: #333; font-size: 1.6em;">{total_above:,}</div>
            <div class="summary-label">Minutes Above TH</div>
        </div>
        <div class="summary-card" style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);">
            <div class="summary-number" style="color: #155724; font-size: 1.6em;">{total_explained:,}</div>
            <div class="summary-label">Minutes Explained</div>
        </div>
        <div class="summary-card" style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);">
            <div class="summary-number" style="color: {total_color}; font-size: 1.6em;">{total_rate:.1%}</div>
            <div class="summary-label">Explanation Rate</div>
        </div>
    </div>
    <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
        <thead>
            <tr style="background: #f8f9fa;">
                <th style="padding: 8px 12px; text-align: left;">Phase</th>
                <th style="padding: 8px 12px; text-align: right;">Above TH</th>
                <th style="padding: 8px 12px; text-align: right;">Explained</th>
                <th style="padding: 8px 12px; text-align: right;">Rate</th>
            </tr>
        </thead>
        <tbody>
            {''.join(phase_rows)}
            <tr style="background: #e9ecef; font-weight: bold;">
                <td style="padding: 8px 12px;">TOTAL</td>
                <td style="padding: 8px 12px; text-align: right;">{total_above:,}</td>
                <td style="padding: 8px 12px; text-align: right;">{total_explained:,}</td>
                <td style="padding: 8px 12px; text-align: right; color: {total_color};">{total_rate:.1%}</td>
            </tr>
        </tbody>
    </table>
    """


def _generate_patterns_section(analysis: Dict[str, Any]) -> str:
    """Generate event patterns section."""
    first = analysis.get('first_iteration', {})
    patterns = first.get('patterns', {})

    if not patterns:
        return '<p>No pattern data available.</p>'

    # Daily statistics
    daily_stats = patterns.get('daily_stats', {})
    recurring = patterns.get('recurring_events', {})
    time_dist = patterns.get('time_distribution', {})

    # Daily stats - simplified to only show total days
    total_days = daily_stats.get('total_days', 0)
    daily_html = f"""
    <div style="margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; display: inline-block;">
        <span style="font-size: 1.5em; font-weight: bold; color: #2d3748;">{total_days}</span>
        <span style="color: #666; margin-left: 10px;">Days of Data Analyzed</span>
    </div>
    """

    # Recurring MATCHES (ON+OFF pairs)
    recurring_matches = patterns.get('recurring_matches', {})
    all_patterns = recurring_matches.get('patterns', [])

    # Filter to show only frequent patterns (every 1-10 days) with duration > 20 min
    # Keep daily, weekly, and patterns with avg_interval_days <= 10
    match_patterns = [
        p for p in all_patterns
        if (p.get('duration_minutes', 0) > 20) and (
            p.get('interval_type') in ('daily', 'weekly')
            or (p.get('avg_interval_days') is not None and p.get('avg_interval_days') <= 10)
        )
    ]

    if match_patterns:
        # Build rows with expandable dates - show ALL patterns, numbered
        pattern_rows = []
        for i, p in enumerate(match_patterns):
            dates_list = p.get('dates', [])
            dates_preview = ', '.join(dates_list[:3]) + ('...' if len(dates_list) > 3 else '')
            all_dates = ', '.join(dates_list)

            pattern_rows.append(f"""
            <tr class="pattern-row" onclick="toggleDates('dates-{i}')">
                <td><strong>{i+1}</strong></td>
                <td>{p.get('avg_start_time', '') or ''}</td>
                <td>{p.get('phase', '') or ''}</td>
                <td>{p.get('magnitude', 0) or 0}W</td>
                <td>{p.get('duration_minutes', 0) or 0} min</td>
                <td>{p.get('interval_type', '') or ''}</td>
                <td>{p.get('occurrences', 0) or 0}</td>
                <td class="dates-cell">{dates_preview}</td>
            </tr>
            <tr id="dates-{i}" class="dates-row" style="display:none;">
                <td colspan="8" class="dates-expanded">
                    <strong>All dates:</strong> {all_dates}
                </td>
            </tr>
            """)

        house_id = analysis.get('house_id', 'unknown')

        recurring_html = f"""
        <h4>Recurring Appliance Usage (Matched ON+OFF Pairs)</h4>
        <div class="metrics-row" style="border-top: none; padding-top: 0; margin-bottom: 15px;">
            <div class="metric">
                <span class="metric-label">Frequent Patterns (≤10 days):</span>
                <span class="metric-value">{len(match_patterns)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">All Patterns:</span>
                <span class="metric-value">{len(all_patterns)}</span>
            </div>
        </div>

        <div style="margin-bottom: 15px; display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
            <label style="display: flex; align-items: center; gap: 5px; cursor: pointer;">
                <input type="checkbox" id="showAllDates" onchange="toggleAllDates(this.checked)">
                <span style="font-size: 0.9em;">Show all dates</span>
            </label>
            <button onclick="showPlotInstructions()" style="padding: 6px 12px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em;">
                Generate Pattern Plots
            </button>
        </div>

        <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
            Click on a row to see all dates when this pattern occurred.
        </p>
        <div style="max-height: 500px; overflow-y: auto;">
        <table class="data-table small">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Start Time</th>
                    <th>Phase</th>
                    <th>Power</th>
                    <th>Duration</th>
                    <th>Frequency</th>
                    <th>Count</th>
                    <th>Dates</th>
                </tr>
            </thead>
            <tbody>{''.join(pattern_rows)}</tbody>
        </table>
        </div>

        <div id="plotInstructions" style="display: none; margin-top: 15px; padding: 15px; background: #e3f2fd; border-radius: 8px; border-left: 4px solid #2196f3;">
            <h5 style="margin: 0 0 10px 0; color: #1976d2;">Generate Pattern Plots</h5>
            <p style="margin: 0 0 10px 0; font-size: 0.9em; color: #333;">
                To generate individual plots for each pattern occurrence, run the following command:
            </p>
            <code style="display: block; background: #263238; color: #80cbc4; padding: 10px; border-radius: 4px; font-size: 0.85em; overflow-x: auto;">
cd experiment_analysis/scripts<br>
python generate_pattern_plots.py --house {house_id}
            </code>
            <p style="margin: 10px 0 0 0; font-size: 0.85em; color: #666;">
                This will create an interactive plot for each date, organized by pattern.
                Add <code>--patterns 1,2,3</code> to generate only specific patterns.
            </p>
        </div>

        <script>
            function toggleDates(id) {{
                var row = document.getElementById(id);
                if (row.style.display === 'none') {{
                    row.style.display = 'table-row';
                }} else {{
                    row.style.display = 'none';
                }}
            }}

            function toggleAllDates(show) {{
                var rows = document.querySelectorAll('.dates-row');
                rows.forEach(function(row) {{
                    row.style.display = show ? 'table-row' : 'none';
                }});
            }}

            function showPlotInstructions() {{
                var div = document.getElementById('plotInstructions');
                div.style.display = div.style.display === 'none' ? 'block' : 'none';
            }}
        </script>
        """
    else:
        recurring_html = """
        <h4>Recurring Appliance Usage</h4>
        <p>No significant recurring match patterns found.</p>
        """

    # AC Detection section
    ac_detection = patterns.get('ac_detection', {})
    ac_html = _generate_ac_detection_html(ac_detection)

    # Boiler Detection section
    boiler_detection = patterns.get('boiler_detection', {})
    boiler_html = _generate_boiler_detection_html(boiler_detection)

    # Device Usage Patterns (seasonal and time of day)
    device_usage = patterns.get('device_usage', {})
    device_usage_html = _generate_device_usage_html(device_usage)


    # Time distribution with ON/OFF breakdown
    by_period = time_dist.get('by_period', {})
    by_period_on = time_dist.get('by_period_on', {})
    by_period_off = time_dist.get('by_period_off', {})

    if by_period:
        periods = [
            ('Night', '00:00-06:00', 'night'),
            ('Morning', '06:00-12:00', 'morning'),
            ('Afternoon', '12:00-18:00', 'afternoon'),
            ('Evening', '18:00-24:00', 'evening'),
        ]

        time_rows = []
        for label, hours, key in periods:
            total = by_period.get(key, 0)
            on_count = by_period_on.get(key, 0)
            off_count = by_period_off.get(key, 0)
            time_rows.append(f"""
            <tr>
                <td><strong>{label}</strong><br><small style="color: #888;">{hours}</small></td>
                <td style="text-align: center;">{total}</td>
                <td style="text-align: center; color: #28a745;">{on_count}</td>
                <td style="text-align: center; color: #dc3545;">{off_count}</td>
            </tr>
            """)

        total_events = sum(by_period.values())
        total_on = sum(by_period_on.values())
        total_off = sum(by_period_off.values())

        time_html = f"""
        <h4>Time of Day Distribution</h4>
        <table class="data-table small" style="max-width: 500px;">
            <thead>
                <tr>
                    <th>Period</th>
                    <th style="text-align: center;">Total Events</th>
                    <th style="text-align: center;">ON</th>
                    <th style="text-align: center;">OFF</th>
                </tr>
            </thead>
            <tbody>
                {''.join(time_rows)}
                <tr style="border-top: 2px solid #ddd; font-weight: bold;">
                    <td>Total</td>
                    <td style="text-align: center;">{total_events}</td>
                    <td style="text-align: center; color: #28a745;">{total_on}</td>
                    <td style="text-align: center; color: #dc3545;">{total_off}</td>
                </tr>
            </tbody>
        </table>
        """
    else:
        time_html = ""

    return daily_html + recurring_html + ac_html + boiler_html + device_usage_html + time_html


def _generate_ac_detection_html(ac_detection: Dict[str, Any]) -> str:
    """Generate HTML section for AC detection results."""
    if not ac_detection:
        return ""

    html_parts = []

    # Central AC section
    central_ac = ac_detection.get('central_ac', {})
    central_activations = central_ac.get('activations', [])  # Now sessions, not individual cycles
    has_central = ac_detection.get('has_central_ac', False)
    total_cycles = central_ac.get('total_cycles', 0)

    if central_activations:
        phases_str = ', '.join(central_ac.get('phases_used', []))
        total_sessions = central_ac.get('total_count', len(central_activations))

        # Build activation rows (now showing sessions) - show ALL
        activation_rows = []
        # Build copyable dates string for toggle
        copyable_dates = []
        for i, act in enumerate(central_activations):
            phase_mags = act.get('phase_magnitudes', {})
            phase_mag_str = ' | '.join(f"{p}: {m}W" for p, m in sorted(phase_mags.items())) if phase_mags else ''
            cycle_count = act.get('cycle_count', 1)

            # Fix negative durations (events crossing midnight)
            duration = act.get('duration_minutes', 0) or 0
            if duration < 0:
                duration = duration + 1440  # Add 24 hours in minutes
            magnitude = act.get('total_magnitude', 0) or act.get('magnitude', 0) or 0

            activation_rows.append(f"""
            <tr>
                <td><strong>{i+1}</strong></td>
                <td>{act.get('date', '')}</td>
                <td>{act.get('on_time', '')}</td>
                <td>{act.get('off_time', '')}</td>
                <td>{duration:.0f} min</td>
                <td>{magnitude}W</td>
                <td>{cycle_count}</td>
            </tr>
            """)
            # Add to copyable format: date on_time-off_time
            copyable_dates.append(f"{act.get('date', '')} {act.get('on_time', '')}-{act.get('off_time', '')}")

        copyable_text = ', '.join(copyable_dates)
        status_badge = '<span class="badge badge-green">Detected</span>' if has_central else '<span class="badge badge-orange">Few activations</span>'

        html_parts.append(f"""
        <h4>Central AC (Multi-Phase) {status_badge}</h4>
        <div class="metrics-row" style="border-top: none; padding-top: 0; margin-bottom: 15px;">
            <div class="metric">
                <span class="metric-label">Total Sessions:</span>
                <span class="metric-value">{central_ac.get('total_count', 0) or 0}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Total Cycles:</span>
                <span class="metric-value">{total_cycles}</span>
            </div>
            <div class="metric">
                <span class="metric-label">Phases:</span>
                <span class="metric-value">{phases_str}</span>
            </div>
        </div>
        <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
            Sessions grouped by 1-hour gaps between compressor cycles. Events synchronized across all active phases (±10 min tolerance).
        </p>
        <p style="font-size: 0.85em; color: #666; margin-bottom: 5px;">
            Showing all {len(central_activations)} sessions
        </p>
        <div style="max-height: 400px; overflow-y: auto;">
        <table class="data-table small">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Date</th>
                    <th>Start Time</th>
                    <th>End Time</th>
                    <th>Duration</th>
                    <th>Power</th>
                    <th>Cycles</th>
                </tr>
            </thead>
            <tbody>{''.join(activation_rows)}</tbody>
        </table>
        </div>
        <div style="margin-top: 10px;">
            <button onclick="toggleCentralACCopyText()" style="padding: 6px 12px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em;">
                Show Copyable Dates
            </button>
        </div>
        <div id="central-ac-copy-text" style="display: none; margin-top: 10px; padding: 10px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;">
            <p style="font-size: 0.8em; color: #666; margin-bottom: 5px;">Copy this text to use for generating plots:</p>
            <textarea readonly style="width: 100%; height: 60px; font-family: monospace; font-size: 0.85em; padding: 5px; border: 1px solid #ced4da; border-radius: 3px;">{copyable_text}</textarea>
        </div>
        <script>
            function toggleCentralACCopyText() {{
                var div = document.getElementById('central-ac-copy-text');
                div.style.display = div.style.display === 'none' ? 'block' : 'none';
            }}
        </script>
        """)
    else:
        html_parts.append("""
        <h4>Central AC (Multi-Phase)</h4>
        <p>No central AC pattern detected (no synchronized multi-phase events found).</p>
        """)

    # Regular AC section - now supports MULTIPLE phases (not just w1)
    regular_ac_by_phase = ac_detection.get('regular_ac_by_phase', {})

    # If old format (single regular_ac), convert to new format
    if not regular_ac_by_phase:
        regular_ac = ac_detection.get('regular_ac', {})
        if regular_ac.get('activations'):
            regular_ac_by_phase = {'w1': regular_ac}

    if regular_ac_by_phase:
        for phase, phase_data in sorted(regular_ac_by_phase.items()):
            activations = phase_data.get('activations', [])  # Now sessions, not individual cycles
            if not activations:
                continue

            total_count = phase_data.get('total_count', len(activations))
            total_cycles = phase_data.get('total_cycles', 0)
            has_regular = total_count >= 3

            # Build activation rows and copyable dates
            activation_rows = []
            copyable_dates = []
            for idx, act in enumerate(activations):
                cycle_count = act.get('cycle_count', 1) or 1
                # Fix negative durations (events crossing midnight)
                duration = act.get('duration_minutes', 0) or 0
                if duration < 0:
                    duration = duration + 1440  # Add 24 hours in minutes
                magnitude = act.get('magnitude', 0) or 0

                activation_rows.append(f"""
                <tr>
                    <td><strong>{idx+1}</strong></td>
                    <td>{act.get('date', '')}</td>
                    <td>{act.get('on_time', '')}</td>
                    <td>{act.get('off_time', '')}</td>
                    <td>{duration:.0f} min</td>
                    <td>{magnitude}W</td>
                    <td>{cycle_count}</td>
                </tr>
                """)
                copyable_dates.append(f"{act.get('date', '')} {act.get('on_time', '')}-{act.get('off_time', '')}")

            copyable_text = ', '.join(copyable_dates)
            status_badge = '<span class="badge badge-green">Detected</span>' if has_regular else '<span class="badge badge-orange">Few activations</span>'
            copy_div_id = f'regular-ac-copy-{phase}'

            html_parts.append(f"""
            <h4>Regular AC (Single Phase - {phase}) {status_badge}</h4>
            <div class="metrics-row" style="border-top: none; padding-top: 0; margin-bottom: 15px;">
                <div class="metric">
                    <span class="metric-label">Total Sessions:</span>
                    <span class="metric-value">{total_count}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Cycles:</span>
                    <span class="metric-value">{total_cycles}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Phase:</span>
                    <span class="metric-value">{phase}</span>
                </div>
            </div>
            <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
                Sessions grouped by 1-hour gaps. High-power events (>800W) on {phase} only, not part of central AC.
            </p>
            <p style="font-size: 0.85em; color: #666; margin-bottom: 5px;">
                Showing all {len(activations)} sessions
            </p>
            <div style="max-height: 400px; overflow-y: auto;">
            <table class="data-table small">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Date</th>
                        <th>Start Time</th>
                        <th>End Time</th>
                        <th>Duration</th>
                        <th>Power</th>
                        <th>Cycles</th>
                    </tr>
                </thead>
                <tbody>{''.join(activation_rows)}</tbody>
            </table>
            </div>
            <div style="margin-top: 10px;">
                <button onclick="document.getElementById('{copy_div_id}').style.display = document.getElementById('{copy_div_id}').style.display === 'none' ? 'block' : 'none'" style="padding: 6px 12px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em;">
                    Show Copyable Dates
                </button>
            </div>
            <div id="{copy_div_id}" style="display: none; margin-top: 10px; padding: 10px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;">
                <p style="font-size: 0.8em; color: #666; margin-bottom: 5px;">Copy this text to use for generating plots:</p>
                <textarea readonly style="width: 100%; height: 60px; font-family: monospace; font-size: 0.85em; padding: 5px; border: 1px solid #ced4da; border-radius: 3px;">{copyable_text}</textarea>
            </div>
            """)
    elif has_central:
        # Only show this if we have central AC but no regular
        html_parts.append("""
        <h4>Regular AC (Single Phase)</h4>
        <p>No single-phase AC pattern detected (all high-power events are part of central AC).</p>
        """)

    return '\n'.join(html_parts)


def _generate_boiler_detection_html(boiler_detection: Dict[str, Any]) -> str:
    """Generate HTML section for boiler detection results."""
    if not boiler_detection:
        return ""

    boiler = boiler_detection.get('boiler', {})
    activations = boiler.get('activations', [])
    has_boiler = boiler_detection.get('has_boiler', False)

    if not activations:
        return """
        <h4>Water Heater (Boiler)</h4>
        <p>No boiler pattern detected (no isolated long high-power events found).</p>
        """

    total_count = boiler.get('total_count', len(activations)) or 0
    avg_duration = boiler.get('avg_duration', 0) or 0
    avg_magnitude = boiler.get('avg_magnitude', 0) or 0

    # Build activation rows and copyable dates
    activation_rows = []
    copyable_dates = []
    for idx, act in enumerate(activations):
        # Fix negative durations (events crossing midnight)
        duration = act.get('duration_minutes', 0) or 0
        if duration < 0:
            duration = duration + 1440  # Add 24 hours in minutes
        magnitude = act.get('magnitude', 0) or 0

        activation_rows.append(f"""
        <tr>
            <td><strong>{idx+1}</strong></td>
            <td>{act.get('date', '')}</td>
            <td>{act.get('on_time', '')}</td>
            <td>{act.get('off_time', '')}</td>
            <td>{duration:.0f} min</td>
            <td>{magnitude}W</td>
            <td>{act.get('phase', '')}</td>
        </tr>
        """)
        copyable_dates.append(f"{act.get('date', '')} {act.get('on_time', '')}-{act.get('off_time', '')}")

    copyable_text = ', '.join(copyable_dates)
    status_badge = '<span class="badge badge-green">Detected</span>' if has_boiler else '<span class="badge badge-orange">Few activations</span>'

    return f"""
    <h4>Water Heater (Boiler) {status_badge}</h4>
    <div class="metrics-row" style="border-top: none; padding-top: 0; margin-bottom: 15px;">
        <div class="metric">
            <span class="metric-label">Total Activations:</span>
            <span class="metric-value">{total_count}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Avg Duration:</span>
            <span class="metric-value">{avg_duration:.0f} min</span>
        </div>
        <div class="metric">
            <span class="metric-label">Avg Power:</span>
            <span class="metric-value">{avg_magnitude:.0f}W</span>
        </div>
    </div>
    <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
        Isolated long-duration (≥25 min) high-power (≥1500W) events with no medium-duration events nearby
    </p>
    <p style="font-size: 0.85em; color: #666; margin-bottom: 5px;">
        Showing all {len(activations)} activations
    </p>
    <div style="max-height: 400px; overflow-y: auto;">
    <table class="data-table small">
        <thead>
            <tr>
                <th>#</th>
                <th>Date</th>
                <th>ON Time</th>
                <th>OFF Time</th>
                <th>Duration</th>
                <th>Power</th>
                <th>Phase</th>
            </tr>
        </thead>
        <tbody>{''.join(activation_rows)}</tbody>
    </table>
    </div>
    <div style="margin-top: 10px;">
        <button onclick="document.getElementById('boiler-copy-text').style.display = document.getElementById('boiler-copy-text').style.display === 'none' ? 'block' : 'none'" style="padding: 6px 12px; background: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85em;">
            Show Copyable Dates
        </button>
    </div>
    <div id="boiler-copy-text" style="display: none; margin-top: 10px; padding: 10px; background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 4px;">
        <p style="font-size: 0.8em; color: #666; margin-bottom: 5px;">Copy this text to use for generating plots:</p>
        <textarea readonly style="width: 100%; height: 60px; font-family: monospace; font-size: 0.85em; padding: 5px; border: 1px solid #ced4da; border-radius: 3px;">{copyable_text}</textarea>
    </div>
    """


def _generate_device_usage_html(device_usage: Dict[str, Any]) -> str:
    """Generate HTML section for device usage patterns by season and time of day."""
    if not device_usage:
        return ""

    html_parts = []

    # Check if we have any data to show
    has_data = False
    for device, data in device_usage.items():
        seasonal = data.get('seasonal', {})
        time_of_day = data.get('time_of_day', {})
        if sum(seasonal.values()) > 0 or sum(time_of_day.values()) > 0:
            has_data = True
            break

    if not has_data:
        return ""

    html_parts.append("""
    <div style="margin-top: 30px; margin-bottom: 20px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h4 style="margin: 0 0 5px 0; color: white;">📊 Device Usage Patterns by Season and Time of Day</h4>
        <p style="margin: 0; font-size: 0.9em; opacity: 0.9;">
            Breakdown of detected device activations by season (Winter: Dec-Feb, Summer: Jun-Aug) and time of day (Day: 06:00-18:00, Night: 18:00-06:00)
        </p>
    </div>
    """)

    device_names = {
        'central_ac': 'Central AC',
        'regular_ac': 'Regular AC',
        'boiler': 'Water Heater (Boiler)'
    }

    for device, data in device_usage.items():
        seasonal = data.get('seasonal', {})
        time_of_day = data.get('time_of_day', {})

        total_seasonal = sum(seasonal.values())
        total_time = sum(time_of_day.values())

        if total_seasonal == 0 and total_time == 0:
            continue

        device_name = device_names.get(device, device)

        # Build seasonal row
        winter = seasonal.get('winter', 0)
        spring = seasonal.get('spring', 0)
        summer = seasonal.get('summer', 0)
        fall = seasonal.get('fall', 0)

        winter_pct = (winter / total_seasonal * 100) if total_seasonal > 0 else 0
        spring_pct = (spring / total_seasonal * 100) if total_seasonal > 0 else 0
        summer_pct = (summer / total_seasonal * 100) if total_seasonal > 0 else 0
        fall_pct = (fall / total_seasonal * 100) if total_seasonal > 0 else 0

        # Build time of day row
        day = time_of_day.get('day', 0)
        night = time_of_day.get('night', 0)

        day_pct = (day / total_time * 100) if total_time > 0 else 0
        night_pct = (night / total_time * 100) if total_time > 0 else 0

        # Choose icon based on device
        device_icon = {'central_ac': '❄️', 'regular_ac': '🌀', 'boiler': '🔥'}.get(device, '⚡')

        html_parts.append(f"""
        <div style="margin-bottom: 20px; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); border-left: 4px solid #667eea;">
            <h5 style="margin: 0 0 15px 0; color: #2c3e50; font-size: 1.1em;">{device_icon} {device_name}</h5>
            <div style="display: flex; flex-wrap: wrap; gap: 40px;">
                <div>
                    <strong style="color: #555; font-size: 0.9em; display: block; margin-bottom: 8px;">By Season:</strong>
                    <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                        <span style="display: inline-block; padding: 6px 12px; background: #3498db; color: white; border-radius: 20px; font-size: 0.85em;">
                            ❄️ Winter: {winter} ({winter_pct:.0f}%)
                        </span>
                        <span style="display: inline-block; padding: 6px 12px; background: #2ecc71; color: white; border-radius: 20px; font-size: 0.85em;">
                            🌸 Spring: {spring} ({spring_pct:.0f}%)
                        </span>
                        <span style="display: inline-block; padding: 6px 12px; background: #e74c3c; color: white; border-radius: 20px; font-size: 0.85em;">
                            ☀️ Summer: {summer} ({summer_pct:.0f}%)
                        </span>
                        <span style="display: inline-block; padding: 6px 12px; background: #f39c12; color: white; border-radius: 20px; font-size: 0.85em;">
                            🍂 Fall: {fall} ({fall_pct:.0f}%)
                        </span>
                    </div>
                </div>
                <div>
                    <strong style="color: #555; font-size: 0.9em; display: block; margin-bottom: 8px;">By Time of Day:</strong>
                    <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                        <span style="display: inline-block; padding: 6px 12px; background: #f1c40f; color: #333; border-radius: 20px; font-size: 0.85em;">
                            ☀️ Day (06-18): {day} ({day_pct:.0f}%)
                        </span>
                        <span style="display: inline-block; padding: 6px 12px; background: #2c3e50; color: white; border-radius: 20px; font-size: 0.85em;">
                            🌙 Night (18-06): {night} ({night_pct:.0f}%)
                        </span>
                    </div>
                </div>
            </div>
        </div>
        """)

    return '\n'.join(html_parts)


def _generate_monthly_breakdown_html(analysis: Dict[str, Any]) -> str:
    """Generate HTML table for monthly breakdown of matching performance."""
    monthly = analysis.get('monthly', {})
    monthly_data = monthly.get('monthly_data', [])

    if not monthly_data:
        return ""

    # Build table rows
    rows = []
    for m in monthly_data:
        month = m.get('month', '?')
        on_events = m.get('on_events', 0) or 0
        off_events = m.get('off_events', on_events) or on_events or 0
        matches = m.get('total_matches', 0) or 0
        matching_rate = m.get('matching_rate', 0) or 0
        matched_power_raw = m.get('matched_power', 0) or 0
        matched_power = matched_power_raw / 1000 if matched_power_raw else 0
        matched_minutes = m.get('matched_minutes', 0) or 0

        # Highlight problematic months
        row_class = ' style="background-color: #fff3cd;"' if matching_rate < 0.4 else ''

        rows.append(f"""
        <tr{row_class}>
            <td>{month}</td>
            <td>{on_events}</td>
            <td>{off_events}</td>
            <td>{matches}</td>
            <td>{matching_rate:.1%}</td>
            <td>{matched_power:.1f} kW</td>
            <td>{matched_minutes:.0f} min</td>
        </tr>
        """)

    # Summary stats
    problematic = monthly.get('problematic_months', []) or []
    best = monthly.get('best_months', []) or []
    avg_rate = monthly.get('avg_monthly_matching_rate', 0) or 0

    summary_html = ""
    if problematic:
        summary_html += f"""
        <div style="margin-top: 10px; padding: 10px; background: #fff3cd; border-radius: 5px; font-size: 0.9em;">
            <strong>Problematic months (&lt;40%):</strong> {', '.join(problematic[:5])}
            {f'... and {len(problematic)-5} more' if len(problematic) > 5 else ''}
        </div>
        """

    return f"""
    <h4>Monthly Breakdown</h4>
    <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
        Matching performance by month (highlighted rows indicate months with &lt;40% matching rate)
    </p>
    <div style="max-height: 400px; overflow-y: auto;">
    <table class="data-table small">
        <thead>
            <tr>
                <th>Month</th>
                <th>ON Events</th>
                <th>OFF Events</th>
                <th>Matches</th>
                <th>Match Rate</th>
                <th>Matched Power</th>
                <th>Matched Minutes</th>
            </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
    </table>
    </div>
    <div class="metrics-row" style="border-top: none; padding-top: 10px;">
        <div class="metric">
            <span class="metric-label">Avg Monthly Rate:</span>
            <span class="metric-value">{avg_rate:.1%}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Months Analyzed:</span>
            <span class="metric-value">{len(monthly_data)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Problematic Months:</span>
            <span class="metric-value">{len(problematic)}</span>
        </div>
    </div>
    {summary_html}
    """


def _generate_flags_section(analysis: Dict[str, Any]) -> str:
    """Generate flags/issues section with colored badges."""
    flags = analysis.get('flags', {})

    if not flags:
        return '<p>No issues detected.</p>'

    active_flags = [(k, v) for k, v in flags.items() if v]

    if not active_flags:
        return '<div class="success" style="padding: 15px; background: #d4edda; border-radius: 8px; color: #155724;">✓ No issues detected.</div>'

    # Define flag colors and icons
    flag_config = {
        'low_matching_rate': {'icon': '📉', 'color': '#dc3545', 'bg': '#f8d7da', 'label': 'Low Matching Rate'},
        'has_negative_values': {'icon': '⚡', 'color': '#dc3545', 'bg': '#f8d7da', 'label': 'Negative Power Values'},
        'low_segmentation': {'icon': '📊', 'color': '#856404', 'bg': '#fff3cd', 'label': 'Low Segmentation'},
        'has_damaged_phases': {'icon': '🔌', 'color': '#dc3545', 'bg': '#f8d7da', 'label': 'Damaged Phase(s)'},
        'has_recurring_patterns': {'icon': '🔄', 'color': '#155724', 'bg': '#d4edda', 'label': 'Recurring Patterns Detected'},
    }

    flag_items = []
    for flag, _ in active_flags:
        config = flag_config.get(flag, {'icon': '⚠', 'color': '#666', 'bg': '#f8f9fa', 'label': flag.replace("_", " ").title()})
        flag_items.append(
            f'<li class="flag-item" style="background: {config["bg"]}; border-left-color: {config["color"]};">'
            f'<span class="flag-icon">{config["icon"]}</span> '
            f'<span style="color: {config["color"]};">{config["label"]}</span></li>'
        )

    return f"""
    <ul class="flags-list">
        {''.join(flag_items)}
    </ul>
    """


def _generate_house_charts(analysis: Dict[str, Any]) -> str:
    """Generate charts for single house."""
    charts_html = []

    first = analysis.get('first_iteration', {})

    # Power Distribution pie chart with percentages
    seg = first.get('segmentation', {})
    if seg:
        chart_id = 'seg-pie-chart'
        segmented = (seg.get('total_segmented_power', 0) or 0) / 1000
        remaining = (seg.get('total_remaining_power', 0) or 0) / 1000
        total = segmented + remaining
        seg_pct = (segmented / total * 100) if total > 0 else 0
        rem_pct = (remaining / total * 100) if total > 0 else 0

        data = {
            'values': [segmented, remaining],
            'labels': [f'Segmented<br>{segmented:.1f} kW ({seg_pct:.1f}%)',
                       f'Remaining<br>{remaining:.1f} kW ({rem_pct:.1f}%)'],
            'type': 'pie',
            'marker': {'colors': ['#28a745', '#6c757d']},
            'textinfo': 'label+percent',
            'textposition': 'inside',
            'hole': 0.4,
            'hovertemplate': '%{label}<extra></extra>'
        }

        layout = {
            'title': 'Power Distribution',
            'annotations': [{
                'text': f'{total:.1f} kW<br>Total',
                'x': 0.5, 'y': 0.5,
                'font': {'size': 14, 'color': '#333'},
                'showarrow': False
            }]
        }

        charts_html.append(f'''
        <div class="chart-container">
            <div id="{chart_id}" style="width:100%;height:350px;"></div>
            <script>
                Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
            </script>
        </div>
        ''')

    # Minutes Distribution pie chart (alongside power chart)
    matching = first.get('matching', {})
    total_matched_minutes = matching.get('total_matched_minutes', 0) or 0
    if total_matched_minutes > 0:
        chart_id = 'minutes-pie-chart'
        # Get minutes by duration category
        duration_minutes = matching.get('duration_minutes_breakdown', {}) or {}
        short_min = duration_minutes.get('short', 0) or 0
        medium_min = duration_minutes.get('medium', 0) or 0
        long_min = duration_minutes.get('long', 0) or 0
        total_min = short_min + medium_min + long_min

        if total_min > 0:
            data = {
                'values': [short_min, medium_min, long_min],
                'labels': [
                    f'Short (1-2 min)<br>{short_min:.0f} min',
                    f'Medium (3-24 min)<br>{medium_min:.0f} min',
                    f'Long (25+ min)<br>{long_min:.0f} min'
                ],
                'type': 'pie',
                'marker': {'colors': ['#17a2b8', '#ffc107', '#dc3545']},
                'textinfo': 'label+percent',
                'textposition': 'inside',
                'hole': 0.4,
            }

            layout = {
                'title': 'Matched Minutes Distribution (sum of all 3 phases)',
                'annotations': [{
                    'text': f'{total_min:.0f} min<br>({total_min/60:.1f} hrs)',
                    'x': 0.5, 'y': 0.5,
                    'font': {'size': 12, 'color': '#333'},
                    'showarrow': False
                }]
            }

            charts_html.append(f'''
            <div class="chart-container">
                <div id="{chart_id}" style="width:100%;height:350px;"></div>
                <script>
                    Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
                </script>
                <p style="text-align: center; font-size: 0.8em; color: #888; margin-top: -10px;">
                    Overlapping events on different phases are counted separately
                </p>
            </div>
            ''')

    # Duration breakdown chart (short/medium/long) - MOVED BEFORE Match Types
    duration_breakdown = first.get('matching', {}).get('duration_breakdown', {})
    if duration_breakdown:
        chart_id = 'duration-chart'
        categories = ['Short (1-2 min)', 'Medium (3-24 min)', 'Long (25+ min)']
        values = [
            duration_breakdown.get('short', 0),
            duration_breakdown.get('medium', 0),
            duration_breakdown.get('long', 0)
        ]
        total = sum(values)

        # Add percentages to labels
        labels_with_pct = []
        for cat, val in zip(categories, values):
            pct = (val / total * 100) if total > 0 else 0
            labels_with_pct.append(f'{cat}<br>{val} ({pct:.0f}%)')

        data = {
            'x': labels_with_pct,
            'y': values,
            'type': 'bar',
            'marker': {'color': ['#17a2b8', '#ffc107', '#dc3545']},
            'text': values,
            'textposition': 'outside'
        }

        layout = {
            'title': 'Match Duration Breakdown',
            'yaxis': {'title': 'Number of Matches'},
            'xaxis': {'title': ''},
        }

        charts_html.append(f'''
        <div class="chart-container">
            <div id="{chart_id}" style="width:100%;height:350px;"></div>
            <script>
                Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
            </script>
        </div>
        ''')

    # Tag breakdown pie chart - MOVED AFTER Duration
    tag_breakdown = first.get('matching', {}).get('tag_breakdown', {})
    if tag_breakdown:
        chart_id = 'tag-pie-chart'
        colors = {'NON-M': '#28a745', 'NOISY': '#ffc107', 'PARTIAL': '#17a2b8'}
        total_tags = sum(tag_breakdown.values())

        # Create labels with counts and percentages
        labels_with_pct = []
        for tag, count in tag_breakdown.items():
            pct = (count / total_tags * 100) if total_tags > 0 else 0
            labels_with_pct.append(f'{tag}<br>{count} ({pct:.1f}%)')

        data = {
            'values': list(tag_breakdown.values()),
            'labels': labels_with_pct,
            'type': 'pie',
            'marker': {'colors': [colors.get(k, '#6c757d') for k in tag_breakdown.keys()]},
            'hole': 0.4,
            'textinfo': 'label',
            'textposition': 'inside'
        }

        layout = {
            'title': 'Match Types',
            'annotations': [{
                'text': f'{total_tags}<br>Matches',
                'x': 0.5, 'y': 0.5,
                'font': {'size': 14, 'color': '#333'},
                'showarrow': False
            }]
        }

        charts_html.append(f'''
        <div class="chart-container">
            <div id="{chart_id}" style="width:100%;height:350px;"></div>
            <script>
                Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
            </script>
        </div>
        ''')

    # Time of Day distribution chart
    patterns = first.get('patterns', {})
    time_dist = patterns.get('time_distribution', {})
    by_period = time_dist.get('by_period', {})

    if by_period:
        chart_id = 'time-dist-chart'
        periods = ['Night (00-06)', 'Morning (06-12)', 'Afternoon (12-18)', 'Evening (18-24)']
        values = [
            by_period.get('night', 0),
            by_period.get('morning', 0),
            by_period.get('afternoon', 0),
            by_period.get('evening', 0)
        ]
        total = sum(values)

        # Add percentages
        labels_with_pct = []
        for period, val in zip(periods, values):
            pct = (val / total * 100) if total > 0 else 0
            labels_with_pct.append(f'{period}<br>{val} ({pct:.0f}%)')

        data = {
            'x': labels_with_pct,
            'y': values,
            'type': 'bar',
            'marker': {'color': ['#2c3e50', '#3498db', '#f39c12', '#9b59b6']}
        }

        layout = {
            'title': 'Events by Time of Day',
            'yaxis': {'title': 'Number of Events'},
        }

        charts_html.append(f'''
        <div class="chart-container">
            <div id="{chart_id}" style="width:100%;height:350px;"></div>
            <script>
                Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
            </script>
        </div>
        ''')

    # Magnitude Minutes bar chart (minutes by power range)
    magnitude_minutes = first.get('matching', {}).get('magnitude_minutes_breakdown', {})
    if magnitude_minutes:
        chart_id = 'magnitude-minutes-chart'
        bins = ['1.3-1.8kW', '1.8-2.5kW', '2.5-3.5kW', '3.5-5kW', '5kW+']
        values = [
            magnitude_minutes.get('1300-1800', 0),
            magnitude_minutes.get('1800-2500', 0),
            magnitude_minutes.get('2500-3500', 0),
            magnitude_minutes.get('3500-5000', 0),
            magnitude_minutes.get('5000+', 0)
        ]
        total = sum(values)

        # Add percentages to labels
        labels_with_pct = []
        for bin_label, val in zip(bins, values):
            pct = (val / total * 100) if total > 0 else 0
            labels_with_pct.append(f'{bin_label}<br>{val:.0f} min ({pct:.0f}%)')

        data = {
            'x': labels_with_pct,
            'y': values,
            'type': 'bar',
            'marker': {'color': ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']}
        }

        layout = {
            'title': 'Matched Minutes by Magnitude (sum of all 3 phases)',
            'yaxis': {'title': 'Total Minutes'},
            'xaxis': {'title': ''},
        }

        charts_html.append(f'''
        <div class="chart-container">
            <div id="{chart_id}" style="width:100%;height:350px;"></div>
            <script>
                Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
            </script>
        </div>
        ''')

    return '\n'.join(charts_html) if charts_html else '<p>No chart data available.</p>'


def _build_house_html_document(house_id: str, summary: str, iterations: str,
                                matching: str, segmentation: str, patterns: str,
                                flags: str, charts: str, generated_at: str) -> str:
    """Build complete HTML document for single house."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House {house_id} - Experiment Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}

        header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}

        header .subtitle {{
            opacity: 0.9;
        }}

        section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        section h2 {{
            color: #444;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }}

        section h4 {{
            color: #555;
            margin-bottom: 10px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}

        .summary-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}

        .summary-card.highlight {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .summary-card.highlight .summary-label {{
            color: rgba(255,255,255,0.9);
        }}

        .summary-number {{
            font-size: 2em;
            font-weight: bold;
            color: #2d3748;
        }}

        .summary-card.highlight .summary-number {{
            color: white;
        }}

        .summary-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}

        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
            margin-top: 8px;
        }}

        .badge-purple {{ background: #e2d5f0; color: #6f42c1; }}
        .badge-green {{ background: #d4edda; color: #155724; }}
        .badge-blue {{ background: #cce5ff; color: #004085; }}
        .badge-orange {{ background: #fff3cd; color: #856404; }}
        .badge-red {{ background: #f8d7da; color: #721c24; }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}

        .data-table.small {{
            font-size: 0.85em;
        }}

        .data-table th {{
            background: #2d3748;
            color: white;
            padding: 10px 12px;
            text-align: left;
        }}

        .data-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #eee;
        }}

        .data-table tr:hover {{
            background: #f8f9fa;
        }}

        .pattern-row {{
            cursor: pointer;
        }}

        .pattern-row:hover {{
            background: #e3f2fd !important;
        }}

        .dates-row {{
            background: #f5f5f5;
        }}

        .dates-expanded {{
            padding: 15px !important;
            font-size: 0.85em;
            color: #555;
            word-break: break-word;
        }}

        .dates-cell {{
            max-width: 150px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .two-columns {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}

        .metrics-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }}

        .metric {{
            display: flex;
            gap: 8px;
        }}

        .metric-label {{
            color: #666;
        }}

        .metric-value {{
            font-weight: bold;
            color: #2d3748;
        }}

        .flags-list {{
            list-style: none;
            padding: 0;
        }}

        .flag-item {{
            padding: 10px 15px;
            margin: 5px 0;
            background: #fff3cd;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }}

        .flag-icon {{
            margin-right: 8px;
        }}

        .warning {{
            padding: 15px;
            background: #f8d7da;
            border-left: 4px solid #dc3545;
            border-radius: 5px;
            margin-bottom: 15px;
            color: #721c24;
        }}

        .success {{
            padding: 15px;
            background: #d4edda;
            border-left: 4px solid #28a745;
            border-radius: 5px;
            color: #155724;
        }}

        .chart-container {{
            margin-bottom: 20px;
            width: 100%;
            max-width: 500px;
        }}

        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            justify-items: start;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #888;
            font-size: 0.9em;
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 10px; }}
            .two-columns {{ grid-template-columns: 1fr; }}
            .charts-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>House {house_id}</h1>
            <div class="subtitle">Experiment Analysis Report | Generated: {generated_at}</div>
        </header>

        <section>
            <h2>Summary</h2>
            {summary}
        </section>

        <section>
            <h2>Issues</h2>
            {flags}
        </section>

        <section>
            <h2>Iterations Progress</h2>
            {iterations}
        </section>

        <section>
            <h2>Matching Details</h2>
            {matching}
        </section>

        <section>
            <h2>Segmentation Details</h2>
            {segmentation}
        </section>

        <section>
            <h2>Event Patterns</h2>
            {patterns}
        </section>

        <section>
            <h2>Charts</h2>
            <div class="charts-grid">
                {charts}
            </div>
        </section>

        <footer>
            House Analysis Report | Generated by experiment_analysis module
        </footer>
    </div>
</body>
</html>
"""
