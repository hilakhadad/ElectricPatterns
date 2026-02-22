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
    table_html, tier_counts, continuity_counts = _generate_comparison_table(analyses, per_house_dir)
    filter_bar_html = _build_filter_bar(tier_counts, continuity_counts)
    charts_html = _generate_charts_section(analyses)
    quality_tiers_html = _generate_quality_tiers_section(analyses)

    # Combine into full HTML
    html_content = _build_html_document(
        title=title,
        summary=summary_html,
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
            <div class="summary-number" id="summary-total">{n_houses}</div>
            <div class="summary-label">Houses Analyzed</div>
        </div>
        <div class="summary-card">
            <div class="summary-number" id="summary-avg-score">{avg_quality:.0f}</div>
            <div class="summary-label">Avg Quality Score</div>
        </div>
        <div class="summary-card">
            <div class="summary-number" id="summary-avg-coverage">{avg_coverage:.1%}</div>
            <div class="summary-label">Avg Coverage</div>
        </div>
        <div class="summary-card">
            <div class="summary-number" id="summary-total-days">{total_days:,}</div>
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
                                per_house_dir: str = '../per_house') -> tuple:
    """Generate sortable comparison table HTML with links to individual reports.

    Returns:
        Tuple of (html_string, tier_counts, continuity_counts)
    """
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

    tier_counts = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0,
                   'faulty_dead': 0, 'faulty_nan': 0, 'faulty_both': 0}
    continuity_counts = {'continuous': 0, 'minor_gaps': 0,
                         'discontinuous': 0, 'fragmented': 0, 'unknown': 0}
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

        # Create issues HTML with tooltip for overflow
        if active_issues:
            issues_text = ', '.join(active_issues)
            issues_html = ', '.join(f'<span class="issue-tag">{issue}</span>' for issue in active_issues[:5])
            if len(active_issues) > 5:
                issues_html += f' <span class="issue-more">+{len(active_issues) - 5}</span>'
        else:
            issues_text = 'None'
            issues_html = '<span class="no-issues">None</span>'

        # Quality badge + tier key
        score = quality.get('quality_score', 0)
        qlabel = flags.get('quality_label')

        if qlabel == 'faulty_both':
            badge = '<span class="badge badge-purple-dark">Faulty (Both)</span>'
            tier_key = 'faulty_both'
        elif qlabel == 'faulty_dead_phase':
            badge = '<span class="badge badge-purple-light">Faulty (Dead Phase)</span>'
            tier_key = 'faulty_dead'
        elif qlabel == 'faulty_high_nan':
            badge = '<span class="badge badge-purple">Faulty (High NaN)</span>'
            tier_key = 'faulty_nan'
        elif score >= 90:
            badge = '<span class="badge badge-green">Excellent</span>'
            tier_key = 'excellent'
        elif score >= 75:
            badge = '<span class="badge badge-blue">Good</span>'
            tier_key = 'good'
        elif score >= 50:
            badge = '<span class="badge badge-orange">Fair</span>'
            tier_key = 'fair'
        else:
            badge = '<span class="badge badge-red">Poor</span>'
            tier_key = 'poor'

        tier_counts[tier_key] += 1

        # NaN continuity
        continuity = quality.get('nan_continuity_label', 'unknown')
        if continuity not in continuity_counts:
            continuity = 'unknown'
        continuity_counts[continuity] += 1

        # Link to individual report
        house_link = f'{per_house_dir}/house_{house_id}.html'

        cov_ratio = coverage.get('coverage_ratio', 0)
        no_data_pct = coverage.get('no_data_pct', 0)
        days_span = coverage.get('days_span', 0)

        rows.append(f"""
        <tr data-tier="{tier_key}" data-continuity="{continuity}" data-house-id="{house_id}"
            data-excluded="false" data-score="{score:.1f}" data-coverage="{cov_ratio:.4f}"
            data-nan="{no_data_pct:.2f}" data-days="{days_span}"
            onclick="toggleExcludeRow(event, this)">
            <td><a href="{house_link}" class="house-link"><strong>{house_id}</strong></a></td>
            <td>{days_span}</td>
            <td>{cov_ratio:.1%}</td>
            <td style="color: {'#6f42c1' if no_data_pct >= 10 else 'inherit'};">{no_data_pct:.1f}%</td>
            <td>{score:.0f} {badge}</td>
            <td>{power.get('total_mean', 0):.0f}</td>
            <td>{power.get('phase_balance_ratio', 0):.2f}</td>
            <td>{temporal.get('total_night_day_ratio', 0):.2f}</td>
            <td class="issues-cell" title="{issues_text}">{issues_html}</td>
        </tr>
        """)

    html = f"""
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
        <strong>Days</strong> = calendar days from first to last reading |
        <strong>Coverage</strong> = minutes with data / total minutes in time span |
        <strong>No Data</strong> = % of time span with no reading (gaps + disconnections) |
        <strong>Phase Balance</strong> = max(phases)/min(phases), ideal=1 |
        <strong>Night/Day</strong> = avg night power / avg day power
    </div>
    <div class="table-wrapper">
    <table class="data-table" id="comparison-table">
        <thead>
            <tr>
                <th onclick="sortTable(0)">House ID</th>
                <th onclick="sortTable(1)">Days<br><small>(duration)</small></th>
                <th onclick="sortTable(2)">Coverage<br><small>(completeness)</small></th>
                <th onclick="sortTable(3)">No Data<br><small>(% of span)</small></th>
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
    </div>
    """
    return html, tier_counts, continuity_counts


def _build_filter_bar(tier_counts: Dict[str, int],
                      continuity_counts: Dict[str, int]) -> str:
    """Build the filter bar with tier and NaN continuity checkboxes."""
    tier_labels = {
        'excellent': ('Excellent', '#28a745'),
        'good': ('Good', '#007bff'),
        'fair': ('Fair', '#ffc107'),
        'poor': ('Poor', '#dc3545'),
        'faulty_dead': ('Faulty (Dead)', '#5a3d7a'),
        'faulty_nan': ('Faulty (NaN)', '#6f42c1'),
        'faulty_both': ('Faulty (Both)', '#4a0e6b'),
    }
    continuity_labels = {
        'continuous': ('Continuous', '#28a745'),
        'minor_gaps': ('Minor Gaps', '#007bff'),
        'discontinuous': ('Discontinuous', '#ffc107'),
        'fragmented': ('Fragmented', '#dc3545'),
    }

    # Tier checkboxes
    tier_html = ''
    for key, (label, color) in tier_labels.items():
        count = tier_counts.get(key, 0)
        if count == 0:
            continue
        tier_html += f"""
        <label class="filter-checkbox" style="border-color: {color};">
            <input type="checkbox" checked onchange="updateFilter()" data-filter-tier="{key}">
            <span class="filter-dot" style="background: {color};"></span>
            {label} <span class="filter-count">({count})</span>
        </label>"""

    # Continuity checkboxes
    cont_html = ''
    for key, (label, color) in continuity_labels.items():
        count = continuity_counts.get(key, 0)
        if count == 0:
            continue
        cont_html += f"""
        <label class="filter-checkbox" style="border-color: {color};">
            <input type="checkbox" checked onchange="updateFilter()" data-filter-continuity="{key}">
            <span class="filter-dot" style="background: {color};"></span>
            {label} <span class="filter-count">({count})</span>
        </label>"""

    return f"""
    <div class="filter-bar">
        <div class="filter-group">
            <span class="filter-group-label">Quality Tier:</span>
            {tier_html}
        </div>
        <div class="filter-group">
            <span class="filter-group-label">NaN Continuity:</span>
            {cont_html}
        </div>
    </div>
    """


def _generate_charts_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate all charts HTML (3 charts: quality distribution, phase balance, issues)."""
    charts = [
        create_quality_distribution_chart(analyses),
        create_phase_balance_chart(analyses),
        create_issues_heatmap(analyses),
    ]

    html_parts = []
    for chart_html in charts:
        html_parts.append(f"""
        <div class="chart-container">
            <div class="chart-content">
                {chart_html}
            </div>
        </div>
        """)

    return '\n'.join(html_parts)


def _generate_quality_tiers_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate quality tiers with score component mini-bars."""
    # Score component definitions: (key, label, max_score)
    score_components = [
        ('sharp_entry_score', 'Sharp Entry', 20),
        ('device_signature_score', 'Device Sig.', 15),
        ('power_profile_score', 'Power Profile', 20),
        ('variability_score', 'Variability', 20),
        ('data_volume_score', 'Data Volume', 15),
        ('integrity_score', 'Integrity', 10),
    ]

    # Group houses into tiers
    tiers = [
        ('Excellent (90+)', 'green', 'excellent'),
        ('Good (75-89)', 'blue', 'good'),
        ('Fair (50-74)', 'orange', 'fair'),
        ('Poor (<50)', 'red', 'poor'),
        ('Faulty — Dead Phase', 'purple-light', 'faulty_dead_phase'),
        ('Faulty — High NaN', 'purple', 'faulty_high_nan'),
        ('Faulty — Both', 'purple-dark', 'faulty_both'),
    ]
    tier_houses = {key: [] for _, _, key in tiers}

    for a in analyses:
        score = a.get('data_quality', {}).get('quality_score', 0)
        qlabel = a.get('flags', {}).get('quality_label')

        if qlabel == 'faulty_both':
            tier_houses['faulty_both'].append(a)
        elif qlabel == 'faulty_dead_phase':
            tier_houses['faulty_dead_phase'].append(a)
        elif qlabel == 'faulty_high_nan':
            tier_houses['faulty_high_nan'].append(a)
        elif score >= 90:
            tier_houses['excellent'].append(a)
        elif score >= 75:
            tier_houses['good'].append(a)
        elif score >= 50:
            tier_houses['fair'].append(a)
        else:
            tier_houses['poor'].append(a)

    # Build tier cards
    component_colors = ['#e74c3c', '#e67e22', '#f39c12', '#9b59b6', '#3498db', '#2ecc71']
    html_parts = []

    for tier_name, color, key in tiers:
        houses = tier_houses[key]
        if not houses:
            continue

        # Calculate average scores per component
        mini_bars_html = ''
        for i, (comp_key, comp_label, max_score) in enumerate(score_components):
            avg = sum(h.get('data_quality', {}).get(comp_key, 0) for h in houses) / len(houses)
            pct = min(100, avg / max_score * 100)
            bar_color = component_colors[i]
            mini_bars_html += f"""
            <div class="mini-bar-row">
                <span class="mini-bar-label">{comp_label}</span>
                <div class="mini-bar-container">
                    <div class="mini-bar" style="width: {pct:.0f}%; background: {bar_color};"></div>
                </div>
                <span class="mini-bar-value">{avg:.1f}/{max_score}</span>
            </div>"""

        house_ids = ', '.join(str(h.get('house_id', '?')) for h in houses)
        avg_score = sum(h.get('data_quality', {}).get('quality_score', 0) for h in houses) / len(houses)

        html_parts.append(f"""
        <div class="tier-score-card tier-{color}">
            <div class="tier-header">
                <h4>{tier_name}</h4>
                <div class="tier-count">{len(houses)} houses &middot; avg {avg_score:.0f}</div>
            </div>
            <div class="mini-bars">
                {mini_bars_html}
            </div>
            <div class="tier-houses">{house_ids}</div>
        </div>
        """)

    return f'<div class="tiers-grid">{"".join(html_parts)}</div>'


def _get_month_name(month_num: int) -> str:
    """Get month name from number."""
    months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    return months[month_num] if 1 <= month_num <= 12 else str(month_num)


def _coverage_color(ratio: float) -> str:
    """Return CSS color string for a coverage ratio (0-1)."""
    if ratio >= 0.95:
        return 'color: #155724;'  # dark green
    elif ratio >= 0.80:
        return 'color: #28a745;'  # green
    elif ratio >= 0.65:
        return 'color: #e67e22;'  # orange
    else:
        return 'color: #dc3545;'  # red


def _coverage_bg(ratio: float) -> str:
    """Return CSS background for a coverage ratio (0-1)."""
    if ratio >= 0.95:
        return 'background: #d4edda;'  # light green
    elif ratio >= 0.80:
        return 'background: #e8f5e9;'  # very light green
    elif ratio >= 0.65:
        return 'background: #fff3cd;'  # light yellow
    else:
        return 'background: #f8d7da;'  # light red


def _score_color(score: float) -> str:
    """Return CSS color string for a quality score (0-100)."""
    if score >= 90:
        return 'color: #155724;'
    elif score >= 75:
        return 'color: #004085;'
    elif score >= 50:
        return 'color: #856404;'
    else:
        return 'color: #721c24;'


def _month_coverage_style(ratio: float) -> str:
    """Return inline style for monthly coverage value with gradient coloring."""
    if ratio >= 0.95:
        return 'color: #155724; font-weight: 700;'
    elif ratio >= 0.80:
        return 'color: #28a745; font-weight: 600;'
    elif ratio >= 0.65:
        return 'color: #e67e22; font-weight: 600;'
    elif ratio >= 0.50:
        return 'color: #dc3545; font-weight: 600;'
    else:
        return 'color: #dc3545; font-weight: 700; background: #f8d7da; padding: 1px 4px; border-radius: 3px;'


def _format_small_pct(pct: float, raw_pct: float = None, has_actual_data: bool = False) -> str:
    """Format percentage, showing '<0.1%' when value rounds to 0 but isn't truly 0."""
    if raw_pct is None:
        raw_pct = pct
    if pct == 0 and raw_pct > 0 and has_actual_data:
        return '&lt;0.1%'
    return f'{pct:.1f}%'


def _build_zero_power_warning(quality: Dict[str, Any]) -> str:
    """Build warning HTML for months with zero power readings."""
    zero_months = quality.get('zero_power_months', 0)
    total_months = quality.get('total_months', 0)
    if zero_months == 0:
        return ''

    penalty = quality.get('zero_power_penalty', 0)
    return f'''
            <div class="card" style="border-left: 4px solid #e67e22;">
                <h2 style="color: #e67e22;">Zero-Power Month Anomaly</h2>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 8px;">
                    <strong>{zero_months} out of {total_months} months</strong> have average power near 0W across all phases.
                    This indicates sensor offline, power disconnection, or data acquisition failure during these periods.
                </p>
                <div style="background: #fff3cd; border-radius: 6px; padding: 12px; font-size: 0.88em; color: #856404;">
                    These months contain no usable data for device detection.
                    Coverage percentage may show 100% (rows exist) but with all-zero readings.
                    Quality score penalized by {penalty:.1f} points.
                </div>
            </div>'''


def _build_anomaly_warning(coverage: Dict[str, Any]) -> str:
    """Build anomaly warning HTML if extreme values detected."""
    if not coverage.get('has_anomalies', False):
        return ''

    anomaly_count = coverage.get('anomaly_count', 0)
    anomaly_phases = coverage.get('anomaly_phases', {})
    max_vals = coverage.get('phase_max_values', {})

    phase_details = []
    for ph, count in anomaly_phases.items():
        max_v = max_vals.get(ph, 0)
        phase_details.append(f'{ph}: {count} readings, max {max_v:,.0f}W')

    details_html = '<br>'.join(phase_details)

    return f'''
            <div class="card" style="border-left: 4px solid #dc3545;">
                <h2 style="color: #dc3545;">Anomaly Warning</h2>
                <p style="color: #666; font-size: 0.9em; margin-bottom: 8px;">
                    <strong>{anomaly_count}</strong> readings exceed 20kW per phase &mdash; almost certainly sensor errors.
                    These extreme values distort all statistics (mean, max, std, CV) and charts.
                </p>
                <div style="background: #f8d7da; border-radius: 6px; padding: 12px; font-size: 0.88em; color: #721c24;">
                    {details_html}
                </div>
                <p style="color: #888; font-size: 0.82em; margin-top: 8px;">
                    Consider filtering these outliers before analysis. Quality score penalized by {coverage.get('anomaly_count', 0)} anomalous readings.
                </p>
            </div>'''


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

            is_zero = month_info.get('is_zero_power', False)
            zero_reason = month_info.get('zero_power_reason', '')
            zero_border = 'border: 2px solid #dc3545;' if is_zero else ''
            zero_badge = f'<span style="display:inline-block; background:#dc3545; color:white; font-size:0.7em; padding:1px 6px; border-radius:3px; margin-left:6px;">ANOMALY</span>' if is_zero else ''
            zero_overlay = f'''
                <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
                    background:rgba(220,53,69,0.9); color:white; padding:4px 10px; border-radius:4px;
                    font-size:0.78em; font-weight:600; white-space:nowrap;">{zero_reason}</div>
            ''' if is_zero else ''

            months_cards_html += f"""
            <div class="month-card" style="{zero_border}">
                <div class="month-header">
                    <strong>{month_name} {year}</strong>{zero_badge}
                </div>
                <div class="month-stats">
                    <span>Days: {month_days}</span>
                    <span>Avg: {month_avg:.0f}W</span>
                    <span style="{_month_coverage_style(month_coverage)}">Coverage: {month_coverage:.0%}</span>
                </div>
                <div class="mini-chart" style="position:relative;">
                    {mini_chart}
                    {zero_overlay}
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
                        <div class="metric-value" style="{_coverage_color(year_coverage)}">{year_coverage:.1%}</div>
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
                <p style="color: #666; font-size: 0.9em; margin-bottom: 12px;">
                    Each card shows one month's daily-average hourly pattern.
                    The chart plots the average total power (sum of all 3 phases)
                    for each hour of the day, averaged across all days in that month.
                    This reveals seasonal shifts in consumption patterns (e.g., AC usage in summer).
                </p>
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

        /* Hero card */
        .hero-card {{
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .hero-score {{
            font-size: 3.2em;
            font-weight: bold;
            color: inherit;
        }}
        .hero-max {{
            font-size: 0.4em;
            opacity: 0.7;
        }}
        .hero-badge {{
            margin: 8px 0 12px 0;
        }}
        .hero-subtitle {{
            font-size: 0.85em;
            opacity: 0.8;
        }}
        .hero-badge-green {{ background: #d4edda; color: #155724; }}
        .hero-badge-blue {{ background: #cce5ff; color: #004085; }}
        .hero-badge-orange {{ background: #fff3cd; color: #856404; }}
        .hero-badge-red {{ background: #f8d7da; color: #721c24; }}
        .hero-badge-purple {{ background: #e2d5f1; color: #6f42c1; }}
        .hero-badge-purple-light {{ background: #d4c5e2; color: #5a3d7a; }}
        .hero-badge-purple-dark {{ background: #c9a3d4; color: #4a0e6b; }}

        /* Overview grid */
        .overview-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
        }}
        @media (max-width: 900px) {{
            .overview-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        .overview-item {{
            padding: 15px;
            text-align: center;
        }}
        .overview-value {{
            font-size: 1.6em;
            font-weight: bold;
            color: #333;
        }}
        .overview-label {{
            color: #555;
            font-size: 0.9em;
            font-weight: 600;
            margin-top: 2px;
        }}
        .overview-desc {{
            color: #999;
            font-size: 0.78em;
            margin-top: 4px;
            line-height: 1.3;
        }}
        .two-col-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }}
        @media (max-width: 600px) {{
            .two-col-grid {{ grid-template-columns: 1fr; }}
        }}
        .phase-list {{
            font-size: 0.88em;
            color: #555;
            text-align: left;
            display: inline-block;
        }}
        .phase-list .phase-row {{
            line-height: 1.6;
        }}
        .phase-list .phase-name {{
            font-weight: 700;
            display: inline-block;
            min-width: 28px;
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
            <!-- Quality Score Hero -->
            <div class="hero-card hero-{badge_class}">
                <div class="hero-score">{score:.0f}<span class="hero-max">/100</span></div>
                <div class="hero-badge"><span class="badge {badge_class}">{badge_text}</span>{nan_badge_html}</div>
                <div class="hero-subtitle">
                    Computed from 6 components: Sharp Entry Rate, Device Signature, Power Profile, Variability, Data Volume, Data Integrity
                </div>
            </div>

            <!-- Data Overview -->
            <div class="card">
                <h2>Data Overview</h2>
                <div class="overview-grid">
                    <div class="overview-item">
                        <div class="overview-value">{coverage.get('days_span', 0)}</div>
                        <div class="overview-label">Days of Data</div>
                        <div class="overview-desc">Total calendar days from first to last reading</div>
                    </div>
                    <div class="overview-item" style="{_coverage_bg(coverage.get('coverage_ratio', 0))}">
                        <div class="overview-value" style="{_coverage_color(coverage.get('coverage_ratio', 0))}">{coverage.get('coverage_ratio', 0):.1%}</div>
                        <div class="overview-label">Coverage</div>
                        <div class="overview-desc">Minutes with data / total minutes in time span</div>
                    </div>
                    <div class="overview-item" style="{'background: #e8daf0;' if coverage.get('no_data_pct', 0) >= 5 else ''}">
                        <div class="overview-value" style="color: #6f42c1;">{_format_small_pct(coverage.get('no_data_pct', 0), coverage.get('no_data_pct_raw', 0), coverage.get('no_data_gap_minutes', 0) > 0)}</div>
                        <div class="overview-label" style="color: #4a0e6b;">No Data</div>
                        <div class="overview-desc">Minutes with no reading within the measurement period (gaps + disconnections)</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-value">{quality.get('sharp_entry_rate', 0):.0%}</div>
                        <div class="overview-label">Sharp Entry Rate</div>
                        <div class="overview-desc">% of threshold crossings from single-minute jumps. Higher = better for algorithm</div>
                    </div>
                </div>
            </div>

            <!-- Data Loss Breakdown: NaN gaps vs No-Data gaps -->
            <div class="card">
                <h2>Data Loss Breakdown</h2>
                <p style="color: #666; font-size: 0.85em; margin-bottom: 12px;">
                    Two types of data loss: <strong>No-Data gaps</strong> (entire rows missing &mdash; sensor offline)
                    vs <strong>NaN gaps</strong> (rows exist but values are missing &mdash; sensor reported empty readings).
                </p>
                <div class="two-col-grid">
                    <div class="overview-item" style="{'background: #fce4ec;' if coverage.get('no_data_gap_pct', 0) >= 10 else 'background: #f8f9fa;'}">
                        <div class="overview-label" style="font-size: 1em; font-weight: 600; margin-bottom: 8px;">No-Data Gaps (Missing Rows)</div>
                        <div style="font-size: 1.4em; font-weight: 700; color: {'#dc3545' if coverage.get('no_data_gap_pct', 0) >= 10 else '#6f42c1'};">{_format_small_pct(coverage.get('no_data_gap_pct', 0), coverage.get('no_data_pct_raw', 0), coverage.get('no_data_gap_minutes', 0) > 0)}</div>
                        <div class="phase-list" style="margin-top: 6px;">
                            <div class="phase-row">{coverage.get('no_data_gap_minutes', 0):,} minutes missing</div>
                            <div class="phase-row">{coverage.get('no_data_gap_count', 0)} separate gaps</div>
                            <div class="phase-row">Longest gap: {coverage.get('max_no_data_gap_minutes', coverage.get('max_gap_minutes', 0)):,.0f} min</div>
                        </div>
                        <div class="overview-desc">Timestamps where the sensor didn't report at all</div>
                    </div>
                    <div class="overview-item" style="{'background: #fff3cd;' if coverage.get('nan_gap_pct', 0) >= 5 else 'background: #f8f9fa;'}">
                        <div class="overview-label" style="font-size: 1em; font-weight: 600; margin-bottom: 8px;">NaN Gaps (Empty Values)</div>
                        <div style="font-size: 1.4em; font-weight: 700; color: {'#e67e22' if coverage.get('nan_gap_pct', 0) >= 5 else '#6f42c1'};">{coverage.get('nan_gap_pct', 0):.1f}%</div>
                        <div class="phase-list" style="margin-top: 6px;">
                            <div class="phase-row">{coverage.get('nan_rows_count', 0):,} rows with NaN</div>
                            <div class="phase-row">w1: {quality.get('w1_nan_pct', 0):.1f}%, w2: {quality.get('w2_nan_pct', 0):.1f}%, w3: {quality.get('w3_nan_pct', 0):.1f}%</div>
                        </div>
                        <div class="overview-desc">Rows that exist but have missing values in one or more phases</div>
                    </div>
                </div>
                <div style="margin-top: 10px; padding: 10px; background: #f0f4ff; border-radius: 6px; font-size: 0.85em; color: #555;">
                    <strong>Total data loss:</strong> {quality.get('total_data_loss_pct', 0):.1f}%
                    (No-Data {_format_small_pct(coverage.get('no_data_gap_pct', 0), coverage.get('no_data_pct_raw', 0), coverage.get('no_data_gap_minutes', 0) > 0)} + NaN {coverage.get('nan_gap_pct', 0):.1f}%)
                    &mdash; Continuity: <span class="badge {nan_cont_class}">{nan_cont_text}</span>
                </div>
            </div>

            {_build_anomaly_warning(coverage)}
            {_build_zero_power_warning(quality)}

            <!-- Per-Phase NaN Details -->
            <div class="card">
                <div class="two-col-grid">
                    <div class="overview-item">
                        <div class="overview-label" style="font-size: 1em; font-weight: 600; margin-bottom: 8px;">NaN % per Phase</div>
                        <div class="phase-list">
                            <div class="phase-row"><span class="phase-name">w1:</span> <span style="{_month_coverage_style(1 - quality.get('w1_nan_pct', 0)/100)}">{quality.get('w1_nan_pct', 0):.1f}%</span></div>
                            <div class="phase-row"><span class="phase-name">w2:</span> <span style="{_month_coverage_style(1 - quality.get('w2_nan_pct', 0)/100)}">{quality.get('w2_nan_pct', 0):.1f}%</span></div>
                            <div class="phase-row"><span class="phase-name">w3:</span> <span style="{_month_coverage_style(1 - quality.get('w3_nan_pct', 0)/100)}">{quality.get('w3_nan_pct', 0):.1f}%</span></div>
                        </div>
                        <div class="overview-desc">% of readings missing per phase (within existing rows)</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-label" style="font-size: 1em; font-weight: 600; margin-bottom: 8px;">Max NaN Gap</div>
                        <div class="phase-list">
                            <div class="phase-row"><span class="phase-name">w1:</span> {coverage.get('w1_max_nan_gap_minutes', 0):,} min</div>
                            <div class="phase-row"><span class="phase-name">w2:</span> {coverage.get('w2_max_nan_gap_minutes', 0):,} min</div>
                            <div class="phase-row"><span class="phase-name">w3:</span> {coverage.get('w3_max_nan_gap_minutes', 0):,} min</div>
                        </div>
                        <div class="overview-desc">Longest consecutive NaN streak per phase</div>
                    </div>
                </div>
            </div>

            <!-- Power Statistics -->
            <div class="card">
                <h2>Power Statistics</h2>
                <div class="overview-grid">
                    <div class="overview-item">
                        <div class="overview-value">{power.get('total_mean', 0):,.0f}W</div>
                        <div class="overview-label">Average Power</div>
                        <div class="phase-list" style="margin-top: 4px;">
                            <div class="phase-row"><span class="phase-name">w1:</span> {power.get('phase_w1_mean', 0):,.0f}W</div>
                            <div class="phase-row"><span class="phase-name">w2:</span> {power.get('phase_w2_mean', 0):,.0f}W</div>
                            <div class="phase-row"><span class="phase-name">w3:</span> {power.get('phase_w3_mean', 0):,.0f}W</div>
                        </div>
                        <div class="overview-desc">Average of sum across phases</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-value">{power.get('total_max', 0):,.0f}W</div>
                        <div class="overview-label">Max Power</div>
                        <div class="phase-list" style="margin-top: 4px;">
                            <div class="phase-row"><span class="phase-name">w1:</span> {power.get('phase_w1_max', 0):,.0f}W</div>
                            <div class="phase-row"><span class="phase-name">w2:</span> {power.get('phase_w2_max', 0):,.0f}W</div>
                            <div class="phase-row"><span class="phase-name">w3:</span> {power.get('phase_w3_max', 0):,.0f}W</div>
                        </div>
                        <div class="overview-desc">Peak of sum across phases</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-value">{power.get('phase_balance_ratio', 0):.2f}</div>
                        <div class="overview-label">Phase Balance</div>
                        <div class="overview-desc">Ratio of max to min phase average. Ideal = 1.0</div>
                    </div>
                    <div class="overview-item">
                        <div class="overview-value">{temporal.get('total_night_day_ratio', 0):.2f}</div>
                        <div class="overview-label">Night/Day Ratio</div>
                        <div class="overview-desc">Average night power / average day power</div>
                    </div>
                </div>
            </div>

            <!-- Charts -->
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
                    <div class="chart-card chart-full-width">
                        {score_breakdown_chart}
                        <div style="font-size: 0.82em; color: #555; padding: 10px 14px; background: #f8f9fa; border-radius: 6px; margin-top: 6px; line-height: 1.7;">
                            <strong>Score components explained:</strong><br>
                            <span style="color:#e74c3c;">&#9632;</span> <strong>Sharp Entry Rate (20 pts)</strong> &mdash; Fraction of threshold (1300W) crossings from single-minute jumps. p50=32%. Higher = easier for algorithm. Score: 0-20.<br>
                            <span style="color:#e67e22;">&#9632;</span> <strong>Device Signature (15 pts)</strong> &mdash; Boiler patterns (sustained &ge;2000W for &ge;20min) + AC compressor cycles (1300-3000W ON/OFF). Score: 0-15.<br>
                            <span style="color:#f39c12;">&#9632;</span> <strong>Power Profile (20 pts)</strong> &mdash; Rewards clear low-power baseline (&lt;100W). Penalizes 500-1000W stuck range. Score: 0-20.<br>
                            <span style="color:#9b59b6;">&#9632;</span> <strong>Variability (20 pts)</strong> &mdash; CV of total power (std/mean). Higher = more ON/OFF activity. p50=1.40. Score: 0-20.<br>
                            <span style="color:#3498db;">&#9632;</span> <strong>Data Volume (15 pts)</strong> &mdash; Days of data (0-10pts) + monthly coverage balance (0-5pts). Score: 0-15.<br>
                            <span style="color:#2ecc71;">&#9632;</span> <strong>Data Integrity (10 pts)</strong> &mdash; Penalties for NaN &gt;1%, frequent gaps &gt;5%, negative readings. Score: 0-10.
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


def _build_html_document(title: str, summary: str, filter_bar: str,
                         table: str, charts: str, quality_tiers: str,
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

        /* Filter bar */
        .filter-bar {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
        }}

        .filter-group {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}

        .filter-group:last-child {{
            margin-bottom: 0;
        }}

        .filter-group-label {{
            font-weight: 600;
            font-size: 0.85em;
            color: #555;
            min-width: 110px;
        }}

        .filter-checkbox {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            border-radius: 16px;
            border: 2px solid #ddd;
            font-size: 0.82em;
            cursor: pointer;
            transition: all 0.15s;
            user-select: none;
            background: white;
        }}

        .filter-checkbox:hover {{
            background: #f0f0f0;
        }}

        .filter-checkbox input {{
            display: none;
        }}

        .filter-checkbox input:not(:checked) + .filter-dot {{
            opacity: 0.3;
        }}

        .filter-checkbox input:not(:checked) ~ .filter-count {{
            opacity: 0.4;
        }}

        .filter-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}

        .filter-count {{
            color: #888;
            font-size: 0.9em;
        }}

        /* Table wrapper for horizontal scroll */
        .table-wrapper {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
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
            white-space: nowrap;
        }}

        .data-table th:hover {{
            background: #5a6fd6;
        }}

        .data-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}

        .data-table tr:hover {{
            background: #f0f4ff;
        }}

        .data-table tr[data-excluded="true"] {{
            opacity: 0.35;
        }}

        .data-table tr[data-excluded="true"] .house-link {{
            text-decoration: line-through;
        }}

        .data-table tr.row-hidden {{
            display: none;
        }}

        .data-table tr {{
            cursor: pointer;
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
            max-width: 200px;
            overflow: hidden;
            text-overflow: ellipsis;
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
            max-width: 100%;
            overflow-x: auto;
        }}

        .chart-content {{
            min-height: 400px;
        }}

        /* Quality tiers - score boxes */
        .tiers-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}

        .tier-score-card {{
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid;
        }}

        .tier-score-card .tier-header {{
            margin-bottom: 12px;
        }}

        .tier-score-card h4 {{
            font-size: 0.95em;
            margin: 0;
        }}

        .tier-score-card .tier-count {{
            font-size: 0.82em;
            color: #666;
            margin-top: 2px;
        }}

        .tier-purple {{ background: #f3edf7; border-color: #6f42c1; }}
        .tier-purple-light {{ background: #ece4f0; border-color: #5a3d7a; }}
        .tier-purple-dark {{ background: #e0d0e8; border-color: #4a0e6b; }}
        .tier-green {{ background: #e8f5e9; border-color: #28a745; }}
        .tier-blue {{ background: #e3f2fd; border-color: #007bff; }}
        .tier-orange {{ background: #fff8e1; border-color: #ffc107; }}
        .tier-red {{ background: #fce4ec; border-color: #dc3545; }}

        /* Mini bars for score components */
        .mini-bars {{
            margin-bottom: 10px;
        }}

        .mini-bar-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }}

        .mini-bar-label {{
            font-size: 0.75em;
            color: #555;
            min-width: 80px;
            text-align: right;
        }}

        .mini-bar-container {{
            flex: 1;
            height: 12px;
            background: rgba(0,0,0,0.08);
            border-radius: 6px;
            overflow: hidden;
        }}

        .mini-bar {{
            height: 100%;
            border-radius: 6px;
            transition: width 0.3s;
        }}

        .mini-bar-value {{
            font-size: 0.72em;
            color: #666;
            min-width: 45px;
        }}

        .tier-houses {{
            font-size: 0.75em;
            color: #888;
            word-break: break-word;
            border-top: 1px solid rgba(0,0,0,0.08);
            padding-top: 8px;
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
            <h2>Quality Score Breakdown</h2>
            {quality_tiers}
        </section>

        <section>
            <h2>House Comparison</h2>
            {filter_bar}
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
            var rows = Array.from(table.tBodies[0].rows);
            var ascending = table.getAttribute('data-sort-col') !== String(n) ||
                           table.getAttribute('data-sort-dir') !== 'asc';

            rows.sort(function(a, b) {{
                var aVal = a.cells[n].textContent.trim();
                var bVal = b.cells[n].textContent.trim();

                var aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
                var bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return ascending ? aNum - bNum : bNum - aNum;
                }}

                return ascending ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }});

            var tbody = table.tBodies[0];
            rows.forEach(function(row) {{ tbody.appendChild(row); }});

            table.setAttribute('data-sort-col', n);
            table.setAttribute('data-sort-dir', ascending ? 'asc' : 'desc');
        }}

        function updateFilter() {{
            // Get checked tiers
            var checkedTiers = [];
            document.querySelectorAll('[data-filter-tier]').forEach(function(cb) {{
                if (cb.checked) checkedTiers.push(cb.getAttribute('data-filter-tier'));
            }});

            // Get checked continuity
            var checkedCont = [];
            document.querySelectorAll('[data-filter-continuity]').forEach(function(cb) {{
                if (cb.checked) checkedCont.push(cb.getAttribute('data-filter-continuity'));
            }});

            // Show/hide rows
            var rows = document.querySelectorAll('#comparison-table tbody tr');
            rows.forEach(function(row) {{
                var tier = row.getAttribute('data-tier');
                var cont = row.getAttribute('data-continuity');
                var tierMatch = checkedTiers.length === 0 || checkedTiers.indexOf(tier) !== -1;
                var contMatch = checkedCont.length === 0 || checkedCont.indexOf(cont) !== -1;

                if (tierMatch && contMatch) {{
                    row.classList.remove('row-hidden');
                }} else {{
                    row.classList.add('row-hidden');
                }}
            }});

            updateSummaryCards();
        }}

        function toggleExcludeRow(event, row) {{
            // Don't toggle if clicking a link
            if (event.target.tagName === 'A' || event.target.closest('a')) return;

            var excluded = row.getAttribute('data-excluded') === 'true';
            row.setAttribute('data-excluded', excluded ? 'false' : 'true');
            updateSummaryCards();
        }}

        function updateSummaryCards() {{
            var rows = document.querySelectorAll('#comparison-table tbody tr');
            var totalHouses = 0;
            var totalScore = 0;
            var totalCoverage = 0;
            var totalDays = 0;

            rows.forEach(function(row) {{
                if (row.classList.contains('row-hidden')) return;
                if (row.getAttribute('data-excluded') === 'true') return;

                totalHouses++;
                totalScore += parseFloat(row.getAttribute('data-score') || 0);
                totalCoverage += parseFloat(row.getAttribute('data-coverage') || 0);
                totalDays += parseInt(row.getAttribute('data-days') || 0);
            }});

            var el;
            el = document.getElementById('summary-total');
            if (el) el.textContent = totalHouses;

            el = document.getElementById('summary-avg-score');
            if (el) el.textContent = totalHouses > 0 ? Math.round(totalScore / totalHouses) : 0;

            el = document.getElementById('summary-avg-coverage');
            if (el) el.textContent = totalHouses > 0
                ? (totalCoverage / totalHouses * 100).toFixed(1) + '%'
                : '0.0%';

            el = document.getElementById('summary-total-days');
            if (el) el.textContent = totalDays.toLocaleString();
        }}
    </script>
</body>
</html>
"""
