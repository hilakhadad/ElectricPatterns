"""
HTML report generator for house analysis.

Combines charts and tables into a single HTML report.
"""
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from visualization.charts import (
    create_days_distribution_chart,
    create_quality_distribution_chart,
    create_coverage_comparison_chart,
    create_phase_balance_chart,
    create_day_night_scatter,
    create_issues_heatmap,
    create_power_distribution_chart,
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
    n_duplicates = sum(1 for a in analyses if a.get('flags', {}).get('has_duplicate_timestamps', False))

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
            <li class="{'alert' if n_duplicates > 0 else ''}">
                <strong>{n_duplicates}</strong> houses with duplicate timestamps
            </li>
        </ul>
    </div>
    """


def _generate_comparison_table(analyses: List[Dict[str, Any]],
                                per_house_dir: str = '../per_house') -> str:
    """Generate sortable comparison table HTML with links to individual reports."""
    # Friendly flag names for display
    flag_display_names = {
        'low_coverage': 'Coverage < 80%',
        'short_duration': 'Less Than 30 Days',
        'has_large_gaps': 'Gaps Over 1 Hour',
        'many_gaps': '>5% Gaps Over 2min',
        'has_duplicate_timestamps': 'Duplicate Timestamps',
        'has_negative_values': 'Negative Values',
        'many_outliers': '>1% Outliers (3σ)',
        'many_large_jumps': '>500 Jumps Over 2kW',
        'low_quality_score': 'Quality < 70',
        'unbalanced_phases': 'Phase Ratio > 3',
        'single_active_phase': 'Single Active Phase',
        'very_high_power': 'Max Power > 20kW',
        'many_flat_segments': '>70% Flat Readings',
        'unusual_night_ratio': 'Night/Day > 3',
        'has_dead_phase': 'Dead Phase (<2% of sisters)',
        'has_faulty_nan_phase': 'Faulty Phase (NaN≥10%)',
        'many_nan_values': 'NaN > 2%',
        'low_sharp_entry': 'Low Sharp Entry Rate',
        'low_device_signature': 'Low Device Signature',
        'low_power_profile': 'Low Power Profile',
        'low_variability': 'Low Variability',
        'low_data_volume': 'Low Data Volume',
        'low_data_integrity': 'Low Data Integrity',
    }

    rows = []

    for a in analyses:
        house_id = a.get('house_id', 'unknown')
        coverage = a.get('coverage', {})
        quality = a.get('data_quality', {})
        power = a.get('power_statistics', {})
        temporal = a.get('temporal_patterns', {})
        flags = a.get('flags', {})

        # Build issues list (excluding dead_phases_list which is not a boolean)
        active_issues = []
        for flag_key, flag_value in flags.items():
            if flag_key == 'dead_phases_list':
                continue  # Skip the list, use has_dead_phase instead
            if flag_value:
                display_name = flag_display_names.get(flag_key, flag_key.replace('_', ' ').title())
                active_issues.append(display_name)

        # Create issues HTML - show as comma-separated colored badges
        if active_issues:
            issues_html = ', '.join(f'<span class="issue-tag">{issue}</span>' for issue in active_issues[:5])
            if len(active_issues) > 5:
                issues_html += f' <span class="issue-more">+{len(active_issues) - 5}</span>'
        else:
            issues_html = '<span class="no-issues">None</span>'

        # Quality badge with Faulty subcategories
        score = quality.get('quality_score', 0)
        qlabel = flags.get('quality_label')

        if qlabel == 'faulty_both':
            badge = '<span class="badge badge-purple-dark">Faulty (Both)</span>'
        elif qlabel == 'faulty_dead_phase':
            badge = '<span class="badge badge-purple-light">Faulty (Dead Phase)</span>'
        elif qlabel == 'faulty_high_nan':
            badge = '<span class="badge badge-purple">Faulty (High NaN)</span>'
        elif score >= 90:
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
            <td>{coverage.get('avg_nan_pct', 0):.1f}%</td>
            <td>{score:.0f} {badge}</td>
            <td>{power.get('total_mean', 0):.0f}</td>
            <td>{power.get('phase_balance_ratio', 0):.2f}</td>
            <td>{temporal.get('total_night_day_ratio', 0):.2f}</td>
            <td class="issues-cell">{issues_html}</td>
        </tr>
        """)

    return f"""
    <div class="table-legend" style="background: #f8f9fa; border-radius: 8px; padding: 15px; margin-bottom: 15px; font-size: 0.9em;">
        <h4 style="margin: 0 0 10px 0; color: #2c3e50;">Quality Score Calculation (0-100) &mdash; Optimized for Algorithm Performance</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px 20px;">
            <div><span style="color:#e74c3c;">&#9632;</span> <strong>Sharp Entry Rate (20 pts):</strong> Fraction of threshold crossings from single-minute sharp jumps. Houses where power reaches 1300W via stacking score low.</div>
            <div><span style="color:#e67e22;">&#9632;</span> <strong>Device Signature (15 pts):</strong> Boiler-like sustained high power (8 pts) + AC-like compressor cycles (7 pts). Predicts segregation success.</div>
            <div><span style="color:#f39c12;">&#9632;</span> <strong>Power Profile (20 pts):</strong> Penalizes houses stuck in 500-1000W (below detection threshold). Rewards clear low-power baseline where events stand out.</div>
            <div><span style="color:#9b59b6;">&#9632;</span> <strong>Variability (20 pts):</strong> Coefficient of variation (CV) of total power. Higher CV = more device switching activity = better for the algorithm.</div>
            <div><span style="color:#3498db;">&#9632;</span> <strong>Data Volume (15 pts):</strong> Days of data + monthly coverage balance. More data = more patterns to detect.</div>
            <div><span style="color:#2ecc71;">&#9632;</span> <strong>Data Integrity (10 pts):</strong> NaN %, gap frequency, negative values. Basic data quality checks.</div>
        </div>
    </div>
    <div class="column-legend" style="font-size: 0.85em; color: #666; margin-bottom: 10px;">
        <strong>Days</strong> = data duration |
        <strong>Coverage</strong> = % of expected minutes present |
        <strong>NaN %</strong> = avg missing values within existing rows |
        <strong>Phase Balance</strong> = max(phases)/min(phases), ideal=1 |
        <strong>Night/Day</strong> = avg night power / avg day power
    </div>
    <table class="data-table" id="comparison-table">
        <thead>
            <tr>
                <th onclick="sortTable(0)">House ID</th>
                <th onclick="sortTable(1)">Days<br><small>(duration)</small></th>
                <th onclick="sortTable(2)">Coverage<br><small>(completeness)</small></th>
                <th onclick="sortTable(3)">NaN %<br><small>(avg phases)</small></th>
                <th onclick="sortTable(4)">Quality<br><small>(0-100)</small></th>
                <th onclick="sortTable(5)">Avg Power<br><small>(Watts)</small></th>
                <th onclick="sortTable(6)">Phase Balance<br><small>(max/min)</small></th>
                <th onclick="sortTable(7)">Night/Day<br><small>(power ratio)</small></th>
                <th>Issues</th>
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

    # Days distribution (how many houses per days range)
    charts.append(('Houses by Days Range', create_days_distribution_chart(analyses)))

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
    # Use ordered list of tuples to preserve display order
    tiers = [
        ('Faulty — Dead Phase', []),
        ('Faulty — High NaN', []),
        ('Faulty — Both', []),
        ('Excellent (90+)', []),
        ('Good (75-89)', []),
        ('Fair (50-74)', []),
        ('Poor (<50)', []),
    ]
    tier_dict = {name: houses for name, houses in tiers}

    for a in analyses:
        house_id = a.get('house_id', 'unknown')
        score = a.get('data_quality', {}).get('quality_score', 0)
        flags = a.get('flags', {})
        qlabel = flags.get('quality_label')

        # Faulty subcategories take priority over numeric tiers
        if qlabel == 'faulty_both':
            tier_dict['Faulty — Both'].append(house_id)
        elif qlabel == 'faulty_dead_phase':
            tier_dict['Faulty — Dead Phase'].append(house_id)
        elif qlabel == 'faulty_high_nan':
            tier_dict['Faulty — High NaN'].append(house_id)
        elif score >= 90:
            tier_dict['Excellent (90+)'].append(house_id)
        elif score >= 75:
            tier_dict['Good (75-89)'].append(house_id)
        elif score >= 50:
            tier_dict['Fair (50-74)'].append(house_id)
        else:
            tier_dict['Poor (<50)'].append(house_id)

    html_parts = []
    colors = {
        'Faulty — Dead Phase': 'purple-light',
        'Faulty — High NaN': 'purple',
        'Faulty — Both': 'purple-dark',
        'Excellent (90+)': 'green',
        'Good (75-89)': 'blue',
        'Fair (50-74)': 'orange',
        'Poor (<50)': 'red',
    }

    for tier_name, houses in tiers:
        color = colors[tier_name]
        houses_str = ', '.join(str(h) for h in houses) if houses else 'None'
        html_parts.append(f"""
        <div class="tier-card tier-{color}">
            <h4>{tier_name}</h4>
            <div class="tier-count">{len(houses)} houses</div>
            <div class="tier-houses">{houses_str}</div>
        </div>
        """)

    return f'<div class="tiers-grid">{"".join(html_parts)}</div>'


def _get_month_name(month_num: int) -> str:
    """Get month name from number."""
    months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    return months[month_num] if 1 <= month_num <= 12 else str(month_num)


def generate_single_house_html_report(analysis: Dict[str, Any],
                                       output_path: str) -> str:
    """
    Generate HTML report for a single house with year tabs and monthly details.

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
    temporal_by_period = analysis.get('temporal_by_period', {})
    flags = analysis.get('flags', {})

    # Quality badge with Faulty subcategories
    score = quality.get('quality_score', 0)
    qlabel = flags.get('quality_label')
    if qlabel == 'faulty_both':
        badge_class = 'badge-purple-dark'
        badge_text = 'Faulty (Both)'
    elif qlabel == 'faulty_dead_phase':
        badge_class = 'badge-purple-light'
        badge_text = 'Faulty (Dead Phase)'
    elif qlabel == 'faulty_high_nan':
        badge_class = 'badge-purple'
        badge_text = 'Faulty (High NaN)'
    elif score >= 90:
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

    # NaN continuity badge
    nan_continuity = quality.get('nan_continuity_label', '')
    nan_cont_colors = {
        'continuous': ('badge-green', 'Continuous'),
        'minor_gaps': ('badge-blue', 'Minor Gaps'),
        'discontinuous': ('badge-orange', 'Discontinuous'),
        'fragmented': ('badge-red', 'Fragmented'),
    }
    nan_cont_class, nan_cont_text = nan_cont_colors.get(nan_continuity, ('', ''))
    max_nan_pct = quality.get('max_phase_nan_pct', 0)
    nan_badge_html = ''
    if nan_cont_text:
        nan_badge_html = f' <span class="badge {nan_cont_class}" title="NaN continuity: max phase NaN = {max_nan_pct:.1f}%">{nan_cont_text}</span>'

    # Active flags
    active_flags = [k.replace('_', ' ').title() for k, v in flags.items() if v]
    flags_html = ', '.join(active_flags) if active_flags else 'None'

    # Get years from temporal_by_period
    years_data = temporal_by_period.get('by_year', {})
    years = sorted(years_data.keys())

    # Create all charts for "All Data" tab
    hourly_chart = create_hourly_pattern_chart(analysis)
    phase_chart = create_phase_power_chart(analysis)
    monthly_chart = create_monthly_pattern_chart(analysis)
    weekly_chart = create_weekly_pattern_chart(analysis)
    heatmap_chart = create_power_heatmap_chart(analysis)
    histogram_chart = create_power_histogram(analysis)
    score_breakdown_chart = create_score_breakdown_chart(analysis)
    quality_flags_chart = create_quality_flags_chart(analysis)

    # Generate year tabs
    year_tabs_html = '<button class="tab-btn active" onclick="showTab(\'all\')">All Data</button>'
    for year in years:
        year_tabs_html += f'<button class="tab-btn" onclick="showTab(\'{year}\')">{year}</button>'

    # Generate year content sections
    year_sections_html = ""

    for year in years:
        year_data = years_data[year]
        months_data = year_data.get('months', {})

        # Year summary cards
        year_days = year_data.get('days', 0)
        year_coverage = year_data.get('coverage_ratio', 0)
        year_avg_power = year_data.get('avg_power', 0)

        # Year charts
        year_hourly = create_year_hourly_chart(year_data.get('hourly_pattern', {}), year)
        year_heatmap = create_year_heatmap(year_data.get('power_heatmap', {}), year)

        # Monthly cards with mini charts
        months_cards_html = ""
        for month_num in sorted(months_data.keys()):
            month_info = months_data[month_num]
            month_name = _get_month_name(month_num)
            month_days = month_info.get('days', 0)
            month_avg = month_info.get('avg_power', 0)
            month_coverage = month_info.get('coverage_ratio', 0)
            mini_chart = create_mini_hourly_chart(month_info.get('hourly_pattern', {}))

            months_cards_html += f"""
            <div class="month-card">
                <div class="month-header">
                    <strong>{month_name} {year}</strong>
                </div>
                <div class="month-stats">
                    <span>Days: {month_days}</span>
                    <span>Avg: {month_avg:.0f}W</span>
                    <span>Coverage: {month_coverage:.0%}</span>
                </div>
                <div class="mini-chart">
                    {mini_chart}
                </div>
            </div>
            """

        year_sections_html += f"""
        <div id="tab-{year}" class="tab-content" style="display: none;">
            <div class="card">
                <h2>{year} Summary</h2>
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value">{year_days}</div>
                        <div class="metric-label">Days of Data</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{year_coverage:.1%}</div>
                        <div class="metric-label">Coverage</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{year_avg_power:.0f}W</div>
                        <div class="metric-label">Avg Power</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{len(months_data)}</div>
                        <div class="metric-label">Months</div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>{year} Power Patterns</h2>
                <div class="chart-card" style="margin-bottom: 20px;">
                    {year_hourly}
                </div>
                <div class="chart-card">
                    {year_heatmap}
                </div>
            </div>

            <div class="card">
                <h2>Monthly Details</h2>
                <button class="expand-btn" onclick="toggleMonths('{year}')">
                    Show Monthly Details
                </button>
                <div id="months-{year}" class="months-container" style="display: none;">
                    {months_cards_html}
                </div>
            </div>
        </div>
        """

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
        .container {{ max-width: 1100px; margin: 0 auto; }}
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
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{ font-size: 1.6em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #666; font-size: 0.85em; }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 20px;
        }}
        .chart-card {{
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .chart-full-width {{ grid-column: span 2; }}
        @media (max-width: 900px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
            .chart-full-width {{ grid-column: span 1; }}
        }}
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
        .badge-purple {{ background: #e2d5f1; color: #6f42c1; }}
        .badge-purple-light {{ background: #d4c5e2; color: #5a3d7a; }}
        .badge-purple-dark {{ background: #c9a3d4; color: #4a0e6b; }}
        .flags {{ color: #e74c3c; }}
        .back-link {{ margin-bottom: 15px; }}
        .back-link a {{ color: #667eea; text-decoration: none; }}
        .back-link a:hover {{ text-decoration: underline; }}

        /* Year tabs */
        .tabs-container {{
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .tab-btn {{
            padding: 10px 20px;
            border: none;
            background: #e9ecef;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.2s;
        }}
        .tab-btn:hover {{ background: #dee2e6; }}
        .tab-btn.active {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Monthly details */
        .expand-btn {{
            padding: 10px 20px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            margin-bottom: 15px;
        }}
        .expand-btn:hover {{ background: #f0f0ff; }}
        .months-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .month-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #e9ecef;
        }}
        .month-header {{
            font-size: 1.1em;
            margin-bottom: 8px;
            color: #495057;
        }}
        .month-stats {{
            display: flex;
            gap: 15px;
            font-size: 0.85em;
            color: #6c757d;
            margin-bottom: 10px;
        }}
        .mini-chart {{
            background: white;
            border-radius: 4px;
            padding: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="back-link">
            <a href="javascript:history.back()">← Back to Summary</a>
        </div>

        <header>
            <h1>House {house_id} Analysis</h1>
            <span class="badge {badge_class}">{badge_text} - Score: {score:.0f}/100</span>{nan_badge_html}
        </header>

        <div class="tabs-container">
            {year_tabs_html}
        </div>

        <!-- All Data Tab -->
        <div id="tab-all" class="tab-content active">
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
                        <div class="metric-value">{coverage.get('avg_nan_pct', 0):.1f}%</div>
                        <div class="metric-label">Avg NaN %</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{coverage.get('max_gap_minutes', 0):.0f}</div>
                        <div class="metric-label">Max Gap (min)</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{coverage.get('duplicate_timestamps_count', 0):,}</div>
                        <div class="metric-label">Duplicate TS</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{quality.get('sharp_entry_rate', 0):.0%}</div>
                        <div class="metric-label">Sharp Entry Rate</div>
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
                <h2>Power Patterns & Analysis</h2>
                <div class="charts-grid">
                    <div class="chart-card chart-full-width">
                        {hourly_chart}
                    </div>
                    <div class="chart-card">
                        {phase_chart}
                    </div>
                    <div class="chart-card">
                        {histogram_chart}
                    </div>
                    <div class="chart-card">
                        {weekly_chart}
                    </div>
                    <div class="chart-card">
                        {monthly_chart}
                    </div>
                    <div class="chart-card">
                        {score_breakdown_chart}
                        <div style="font-size: 0.82em; color: #555; padding: 8px 12px; background: #f8f9fa; border-radius: 6px; margin-top: 4px; line-height: 1.5;">
                            <strong>Score components explained:</strong><br>
                            <span style="color:#e74c3c;">&#9632;</span> <strong>Sharp Entry Rate</strong> &mdash; % of threshold crossings from single-minute sharp jumps (higher = easier for algorithm)<br>
                            <span style="color:#e67e22;">&#9632;</span> <strong>Device Signature</strong> &mdash; Boiler patterns (sustained high power) + AC patterns (compressor cycles)<br>
                            <span style="color:#f39c12;">&#9632;</span> <strong>Power Profile</strong> &mdash; Penalizes stuck 500-1000W range; rewards clear low-power baseline<br>
                            <span style="color:#9b59b6;">&#9632;</span> <strong>Variability</strong> &mdash; CV of total power (higher = more device activity = better)<br>
                            <span style="color:#3498db;">&#9632;</span> <strong>Data Volume</strong> &mdash; Days of data + monthly coverage balance<br>
                            <span style="color:#2ecc71;">&#9632;</span> <strong>Data Integrity</strong> &mdash; NaN %, gap frequency, negative values
                        </div>
                    </div>
                    <div class="chart-card chart-full-width">
                        {quality_flags_chart}
                    </div>
                    <div class="chart-card chart-full-width">
                        {heatmap_chart}
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Issues & Flags</h2>
                <p class="flags">{flags_html}</p>
            </div>
        </div>

        <!-- Year Tabs Content -->
        {year_sections_html}

        <footer style="text-align: center; color: #888; padding: 20px;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </footer>
    </div>

    <script>
        function showTab(tabId) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {{
                tab.classList.remove('active');
                tab.style.display = 'none';
            }});

            // Deactivate all buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});

            // Show selected tab
            const selectedTab = document.getElementById('tab-' + tabId);
            if (selectedTab) {{
                selectedTab.classList.add('active');
                selectedTab.style.display = 'block';
            }}

            // Activate button
            event.target.classList.add('active');

            // Trigger Plotly resize for charts
            setTimeout(() => {{
                window.dispatchEvent(new Event('resize'));
            }}, 100);
        }}

        function toggleMonths(year) {{
            const container = document.getElementById('months-' + year);
            const btn = event.target;
            if (container.style.display === 'none') {{
                container.style.display = 'grid';
                btn.textContent = 'Hide Monthly Details';
            }} else {{
                container.style.display = 'none';
                btn.textContent = 'Show Monthly Details';
            }}
            // Trigger resize for mini charts
            setTimeout(() => {{
                window.dispatchEvent(new Event('resize'));
            }}, 100);
        }}
    </script>
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
        .badge-purple {{ background: #e2d5f1; color: #6f42c1; }}
        .badge-purple-light {{ background: #d4c5e2; color: #5a3d7a; }}
        .badge-purple-dark {{ background: #c9a3d4; color: #4a0e6b; }}

        /* Issue tags in table */
        .issues-cell {{
            max-width: 300px;
        }}
        .issue-tag {{
            display: inline-block;
            padding: 2px 6px;
            margin: 1px;
            background: #fee2e2;
            color: #991b1b;
            border-radius: 4px;
            font-size: 0.7em;
            white-space: nowrap;
        }}
        .issue-more {{
            color: #666;
            font-size: 0.8em;
            font-style: italic;
        }}
        .no-issues {{
            color: #28a745;
            font-size: 0.85em;
        }}

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
            grid-template-columns: repeat(7, 1fr);
            gap: 12px;
        }}

        @media (max-width: 1200px) {{
            .tiers-grid {{
                grid-template-columns: repeat(4, 1fr);
            }}
        }}

        @media (max-width: 800px) {{
            .tiers-grid {{
                grid-template-columns: repeat(3, 1fr);
            }}
        }}

        @media (max-width: 600px) {{
            .tiers-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}

        .tier-card {{
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid;
        }}

        .tier-card h4 {{
            font-size: 0.85em;
            margin: 0 0 5px 0;
        }}

        .tier-purple {{ background: #e2d5f1; border-color: #6f42c1; }}
        .tier-purple-light {{ background: #d4c5e2; border-color: #5a3d7a; }}
        .tier-purple-dark {{ background: #c9a3d4; border-color: #4a0e6b; }}
        .tier-green {{ background: #d4edda; border-color: #28a745; }}
        .tier-blue {{ background: #cce5ff; border-color: #007bff; }}
        .tier-orange {{ background: #fff3cd; border-color: #ffc107; }}
        .tier-red {{ background: #f8d7da; border-color: #dc3545; }}

        .tier-count {{
            font-size: 1.3em;
            font-weight: bold;
            margin: 8px 0;
        }}

        .tier-houses {{
            font-size: 0.75em;
            color: #666;
            word-break: break-word;
            max-height: 80px;
            overflow-y: auto;
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
