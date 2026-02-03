"""
HTML report generator for house analysis.

Combines charts and tables into a single HTML report.
"""
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from visualization.charts import (
    create_quality_distribution_chart,
    create_coverage_comparison_chart,
    create_phase_balance_chart,
    create_day_night_scatter,
    create_issues_heatmap,
    create_power_distribution_chart,
)


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
    table_html = _generate_comparison_table(analyses, per_house_dir)
    charts_html = _generate_charts_section(analyses)
    quality_tiers_html = _generate_quality_tiers_section(analyses)

    # Combine into full HTML
    html_content = _build_html_document(
        title=title,
        summary=summary_html,
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


def _generate_summary_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate executive summary HTML."""
    n_houses = len(analyses)

    # Calculate averages
    quality_scores = [a.get('data_quality', {}).get('quality_score', 0) for a in analyses]
    coverage_ratios = [a.get('coverage', {}).get('coverage_ratio', 0) for a in analyses]
    days_spans = [a.get('coverage', {}).get('days_span', 0) for a in analyses]

    avg_quality = sum(quality_scores) / n_houses if n_houses > 0 else 0
    avg_coverage = sum(coverage_ratios) / n_houses if n_houses > 0 else 0
    total_days = sum(days_spans)

    # Count issues
    n_low_quality = sum(1 for a in analyses if a.get('flags', {}).get('low_quality_score', False))
    n_low_coverage = sum(1 for a in analyses if a.get('flags', {}).get('low_coverage', False))
    n_negative = sum(1 for a in analyses if a.get('flags', {}).get('has_negative_values', False))

    return f"""
    <div class="summary-grid">
        <div class="summary-card">
            <div class="summary-number">{n_houses}</div>
            <div class="summary-label">Houses Analyzed</div>
        </div>
        <div class="summary-card">
            <div class="summary-number">{avg_quality:.0f}</div>
            <div class="summary-label">Avg Quality Score</div>
        </div>
        <div class="summary-card">
            <div class="summary-number">{avg_coverage:.1%}</div>
            <div class="summary-label">Avg Coverage</div>
        </div>
        <div class="summary-card">
            <div class="summary-number">{total_days:,}</div>
            <div class="summary-label">Total Days of Data</div>
        </div>
    </div>
    <div class="summary-alerts">
        <h3>Issues Summary</h3>
        <ul>
            <li class="{'alert' if n_low_quality > 0 else ''}">
                <strong>{n_low_quality}</strong> houses with low quality score (&lt;70)
            </li>
            <li class="{'alert' if n_low_coverage > 0 else ''}">
                <strong>{n_low_coverage}</strong> houses with low coverage (&lt;80%)
            </li>
            <li class="{'alert' if n_negative > 0 else ''}">
                <strong>{n_negative}</strong> houses with negative power values
            </li>
        </ul>
    </div>
    """


def _generate_comparison_table(analyses: List[Dict[str, Any]],
                                per_house_dir: str = '../per_house') -> str:
    """Generate sortable comparison table HTML with links to individual reports."""
    rows = []

    for a in analyses:
        house_id = a.get('house_id', 'unknown')
        coverage = a.get('coverage', {})
        quality = a.get('data_quality', {})
        power = a.get('power_statistics', {})
        temporal = a.get('temporal_patterns', {})
        flags = a.get('flags', {})

        # Count active flags
        n_flags = sum(1 for v in flags.values() if v)

        # Quality badge
        score = quality.get('quality_score', 0)
        if score >= 90:
            badge = '<span class="badge badge-green">Excellent</span>'
        elif score >= 75:
            badge = '<span class="badge badge-blue">Good</span>'
        elif score >= 50:
            badge = '<span class="badge badge-orange">Fair</span>'
        else:
            badge = '<span class="badge badge-red">Poor</span>'

        # Link to individual report
        house_link = f'{per_house_dir}/house_{house_id}.html'

        rows.append(f"""
        <tr>
            <td><a href="{house_link}" class="house-link"><strong>{house_id}</strong></a></td>
            <td>{coverage.get('days_span', 0)}</td>
            <td>{coverage.get('coverage_ratio', 0):.1%}</td>
            <td>{score:.0f} {badge}</td>
            <td>{power.get('total_mean', 0):.0f}</td>
            <td>{power.get('phase_balance_ratio', 0):.2f}</td>
            <td>{temporal.get('total_night_day_ratio', 0):.2f}</td>
            <td>{n_flags}</td>
        </tr>
        """)

    return f"""
    <table class="data-table" id="comparison-table">
        <thead>
            <tr>
                <th onclick="sortTable(0)">House ID</th>
                <th onclick="sortTable(1)">Days</th>
                <th onclick="sortTable(2)">Coverage</th>
                <th onclick="sortTable(3)">Quality</th>
                <th onclick="sortTable(4)">Avg Power (W)</th>
                <th onclick="sortTable(5)">Phase Balance</th>
                <th onclick="sortTable(6)">Night/Day</th>
                <th onclick="sortTable(7)">Issues</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def _generate_charts_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate all charts HTML."""
    charts = []

    # Quality distribution
    charts.append(('Quality Score Distribution', create_quality_distribution_chart(analyses)))

    # Coverage comparison
    charts.append(('Data Coverage', create_coverage_comparison_chart(analyses)))

    # Phase balance
    charts.append(('Phase Balance', create_phase_balance_chart(analyses)))

    # Day/Night scatter
    charts.append(('Day vs Night Consumption', create_day_night_scatter(analyses)))

    # Issues heatmap
    charts.append(('Issues Overview', create_issues_heatmap(analyses)))

    # Power distribution (average)
    charts.append(('Power Range Distribution', create_power_distribution_chart(analyses)))

    html_parts = []
    for title, chart_html in charts:
        html_parts.append(f"""
        <div class="chart-container">
            <div class="chart-content">
                {chart_html}
            </div>
        </div>
        """)

    return '\n'.join(html_parts)


def _generate_quality_tiers_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate quality tiers breakdown HTML."""
    tiers = {
        'Excellent (90+)': [],
        'Good (75-89)': [],
        'Fair (50-74)': [],
        'Poor (<50)': []
    }

    for a in analyses:
        house_id = a.get('house_id', 'unknown')
        score = a.get('data_quality', {}).get('quality_score', 0)

        if score >= 90:
            tiers['Excellent (90+)'].append(house_id)
        elif score >= 75:
            tiers['Good (75-89)'].append(house_id)
        elif score >= 50:
            tiers['Fair (50-74)'].append(house_id)
        else:
            tiers['Poor (<50)'].append(house_id)

    html_parts = []
    colors = {'Excellent (90+)': 'green', 'Good (75-89)': 'blue',
              'Fair (50-74)': 'orange', 'Poor (<50)': 'red'}

    for tier, houses in tiers.items():
        color = colors[tier]
        houses_str = ', '.join(str(h) for h in houses) if houses else 'None'
        html_parts.append(f"""
        <div class="tier-card tier-{color}">
            <h4>{tier}</h4>
            <div class="tier-count">{len(houses)} houses</div>
            <div class="tier-houses">{houses_str}</div>
        </div>
        """)

    return f'<div class="tiers-grid">{"".join(html_parts)}</div>'


def generate_single_house_html_report(analysis: Dict[str, Any],
                                       output_path: str) -> str:
    """
    Generate HTML report for a single house.

    Args:
        analysis: Analysis results from analyze_single_house
        output_path: Path to save the HTML file

    Returns:
        Path to the generated HTML file
    """
    house_id = analysis.get('house_id', 'unknown')
    coverage = analysis.get('coverage', {})
    quality = analysis.get('data_quality', {})
    power = analysis.get('power_statistics', {})
    temporal = analysis.get('temporal_patterns', {})
    flags = analysis.get('flags', {})

    # Quality badge
    score = quality.get('quality_score', 0)
    if score >= 90:
        badge_class = 'badge-green'
        badge_text = 'Excellent'
    elif score >= 75:
        badge_class = 'badge-blue'
        badge_text = 'Good'
    elif score >= 50:
        badge_class = 'badge-orange'
        badge_text = 'Fair'
    else:
        badge_class = 'badge-red'
        badge_text = 'Poor'

    # Active flags
    active_flags = [k.replace('_', ' ').title() for k, v in flags.items() if v]
    flags_html = ', '.join(active_flags) if active_flags else 'None'

    # Create power chart for this house
    power_chart = create_power_distribution_chart([analysis], house_id)

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House {house_id} Analysis</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .card h2 {{ margin-top: 0; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .badge {{
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: bold;
            display: inline-block;
        }}
        .badge-green {{ background: #d4edda; color: #155724; }}
        .badge-blue {{ background: #cce5ff; color: #004085; }}
        .badge-orange {{ background: #fff3cd; color: #856404; }}
        .badge-red {{ background: #f8d7da; color: #721c24; }}
        .flags {{ color: #e74c3c; }}
        .back-link {{ margin-bottom: 15px; }}
        .back-link a {{ color: #667eea; text-decoration: none; }}
        .back-link a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="javascript:history.back()">← Back to Summary</a>
        </div>

        <header>
            <h1>House {house_id} Analysis</h1>
            <span class="badge {badge_class}">{badge_text} - Score: {score:.0f}/100</span>
        </header>

        <div class="card">
            <h2>Coverage & Quality</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{coverage.get('days_span', 0)}</div>
                    <div class="metric-label">Days of Data</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{coverage.get('coverage_ratio', 0):.1%}</div>
                    <div class="metric-label">Coverage Ratio</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{quality.get('quality_score', 0):.0f}</div>
                    <div class="metric-label">Quality Score</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{coverage.get('max_gap_minutes', 0):.0f}</div>
                    <div class="metric-label">Max Gap (min)</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Power Statistics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-value">{power.get('total_mean', 0):.0f}W</div>
                    <div class="metric-label">Average Power</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{power.get('total_max', 0):.0f}W</div>
                    <div class="metric-label">Max Power</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{power.get('phase_balance_ratio', 0):.2f}</div>
                    <div class="metric-label">Phase Balance</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{temporal.get('total_night_day_ratio', 0):.2f}</div>
                    <div class="metric-label">Night/Day Ratio</div>
                </div>
            </div>
        </div>

        <div class="card">
            <h2>Power Distribution</h2>
            {power_chart}
        </div>

        <div class="card">
            <h2>Issues & Flags</h2>
            <p class="flags">{flags_html}</p>
        </div>

        <footer style="text-align: center; color: #888; padding: 20px;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </footer>
    </div>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path


def _build_html_document(title: str, summary: str, table: str,
                         charts: str, quality_tiers: str,
                         generated_at: str) -> str:
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
            color: #667eea;
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
            background: #667eea;
            color: white;
            padding: 12px 15px;
            text-align: left;
            cursor: pointer;
            user-select: none;
        }}

        .data-table th:hover {{
            background: #5a6fd6;
        }}

        .data-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}

        .data-table tr:hover {{
            background: #f8f9fa;
        }}

        .house-link {{
            color: #667eea;
            text-decoration: none;
        }}

        .house-link:hover {{
            text-decoration: underline;
        }}

        /* Badges */
        .badge {{
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: bold;
            margin-left: 5px;
        }}

        .badge-green {{ background: #d4edda; color: #155724; }}
        .badge-blue {{ background: #cce5ff; color: #004085; }}
        .badge-orange {{ background: #fff3cd; color: #856404; }}
        .badge-red {{ background: #f8d7da; color: #721c24; }}

        /* Chart containers */
        .chart-container {{
            margin-bottom: 30px;
        }}

        .chart-content {{
            min-height: 400px;
        }}

        /* Quality tiers */
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
            <h2>Quality Tiers</h2>
            {quality_tiers}
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
            House Analysis Report | Generated by house_analysis module
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
