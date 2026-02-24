"""
Single-house report builder for house analysis.

Extracted from html_report.py — contains per-house report generation.
"""
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from shared.html_utils import build_glossary_section as _build_glossary_section, build_about_section as _build_about_section

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
            <div class="card" id="section-zero-power-warning" style="border-left: 4px solid #e67e22;">
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
            <div class="card" id="section-anomaly-warning" style="border-left: 4px solid #dc3545;">
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


def _build_findings_tags(flags: Dict[str, Any], quality: Dict[str, Any],
                          coverage: Dict[str, Any],
                          wave: Dict[str, Any] = None) -> str:
    """Build clickable findings tags section for the top of per-house report."""

    # Define all possible findings: (flag_key, label, severity, target_section_id)
    # severity: 'critical', 'warning', 'info'
    findings = []

    # --- Critical findings ---
    if flags.get('has_dead_phase'):
        dead = quality.get('dead_phases', [])
        findings.append(('critical', f'Dead Phase ({", ".join(dead)})', 'section-data-loss'))

    if flags.get('has_faulty_nan_phase'):
        faulty = quality.get('faulty_nan_phases', [])
        findings.append(('critical', f'Faulty NaN Phase ({", ".join(faulty)})', 'section-data-loss'))

    cov_ratio = coverage.get('coverage_ratio', 1)
    if cov_ratio < 0.50:
        findings.append(('critical', f'Very Low Coverage ({cov_ratio:.0%})', 'section-data-overview'))

    nan_cont = quality.get('nan_continuity_label', '')
    if nan_cont == 'fragmented':
        loss = quality.get('total_data_loss_pct', 0)
        findings.append(('critical', f'Fragmented Data ({loss:.0f}% loss)', 'section-data-loss'))

    # --- Warning findings ---
    if 0.50 <= cov_ratio < 0.70:
        findings.append(('warning', f'Low Coverage ({cov_ratio:.0%})', 'section-data-overview'))

    if coverage.get('has_anomalies', False):
        count = coverage.get('anomaly_count', 0)
        findings.append(('warning', f'Extreme Outliers ({count} readings >20kW)', 'section-anomaly-warning'))

    if nan_cont == 'discontinuous':
        loss = quality.get('total_data_loss_pct', 0)
        findings.append(('warning', f'Discontinuous Data ({loss:.0f}% loss)', 'section-data-loss'))

    zero_months = quality.get('zero_power_months', 0)
    if zero_months > 0:
        total_m = quality.get('total_months', 0)
        findings.append(('warning', f'Zero-Power Months ({zero_months}/{total_m})', 'section-zero-power-warning'))

    if flags.get('has_negative_values'):
        findings.append(('warning', 'Negative Power Values', 'section-power-stats'))

    if flags.get('unbalanced_phases'):
        ratio = quality.get('phase_balance_ratio', 0)
        findings.append(('warning', f'Unbalanced Phases', 'section-power-stats'))

    # --- Info findings (low scoring components) ---
    if flags.get('low_sharp_entry'):
        findings.append(('info', 'Low Sharp Entry Rate', 'section-charts'))

    if flags.get('low_device_signature'):
        findings.append(('info', 'Low Device Signature', 'section-charts'))

    if flags.get('low_power_profile'):
        findings.append(('info', 'Low Power Profile', 'section-charts'))

    if flags.get('low_variability'):
        findings.append(('info', 'Low Variability', 'section-charts'))

    if flags.get('low_data_volume'):
        findings.append(('info', 'Low Data Volume', 'section-data-overview'))

    if flags.get('low_data_integrity'):
        findings.append(('info', 'Low Data Integrity', 'section-data-loss'))

    # Wave behavior findings
    if wave is None:
        wave = {}
    wave_cls = wave.get('wave_classification', 'no_waves')
    if wave_cls == 'wave_dominant':
        dom = ', '.join(wave.get('dominant_phases', []))
        findings.append(('info', f'Wave Dominant ({dom})', 'section-wave-behavior'))
    elif wave_cls == 'has_waves':
        findings.append(('info', 'Wave Behavior Detected', 'section-wave-behavior'))

    if not findings:
        return '''
            <div class="card findings-section">
                <h2>Findings</h2>
                <div class="findings-ok">No significant issues found.</div>
            </div>'''

    tags_html = ''
    for severity, label, target in findings:
        tags_html += f'<a href="#{target}" class="finding-tag finding-tag-{severity}">{label}</a>\n'

    # Penalty summary if anomaly penalties were applied
    penalty_html = ''
    penalty_total = quality.get('anomaly_penalties', 0)
    if penalty_total > 0:
        base = quality.get('base_quality_score', 0)
        final = quality.get('quality_score', 0)
        details = quality.get('anomaly_penalty_details', [])
        reasons = ', '.join(f'-{d["deduction"]}pts ({d["reason"]})' for d in details)
        penalty_html = f'''
                <div class="penalty-summary">
                    <strong>Score adjusted:</strong> Base {base:.0f} &minus; {penalty_total:.0f} penalty = <strong>{final:.0f}</strong>
                    <br><span style="font-size: 0.9em;">{reasons}</span>
                </div>'''

    return f'''
            <div class="card findings-section">
                <h2>Findings</h2>
                <p style="color: #666; font-size: 0.85em; margin-bottom: 8px;">
                    Click a tag to jump to the relevant section.
                </p>
                <div class="findings-tags">
                    {tags_html}
                </div>
                {penalty_html}
            </div>'''


def _build_wave_behavior_section(wave: Dict[str, Any], wave_chart: str) -> str:
    """Build the wave behavior card content for per-house report."""
    wave_cls = wave.get('wave_classification', 'no_waves')
    max_score = wave.get('max_wave_score', 0)
    dominant_phases = wave.get('dominant_phases', [])
    peak_season = wave.get('peak_season')

    # Classification badge
    cls_badge_map = {
        'wave_dominant': '<span class="badge badge-orange">Wave Dominant</span>',
        'has_waves': '<span class="badge badge-blue">Has Waves</span>',
        'no_waves': '<span style="padding:5px 12px; border-radius:15px; background:#e9ecef; color:#6c757d; font-weight:bold;">No Waves</span>',
    }
    cls_badge = cls_badge_map.get(wave_cls, '')

    # Season display
    season_map = {
        'summer': 'Summer (Jun-Sep)',
        'winter': 'Winter (Dec-Mar)',
        'seasonal': 'Seasonal',
        'year_round': 'Year Round',
    }
    season_text = season_map.get(peak_season, '—')

    # Per-phase table rows
    phases = wave.get('phases', {})
    phase_rows = ''
    for col in ['w1', 'w2', 'w3', '1', '2', '3']:
        if col not in phases:
            continue
        p = phases[col]
        ws = p.get('wave_score', 0)
        period = p.get('dominant_period_minutes')
        period_str = f'{period} min' if period else '—'
        prevalence = p.get('wave_prevalence_pct', 0)
        wave_mins = p.get('total_wave_minutes', 0)
        amplitude = p.get('avg_amplitude', 0)

        # Color the score
        if ws >= 0.50:
            score_style = 'color: #e67e22; font-weight: bold;'
        elif ws >= 0.25:
            score_style = 'color: #3498db; font-weight: bold;'
        else:
            score_style = 'color: #95a5a6;'

        display_name = f'Phase {col.replace("w", "")}'
        phase_rows += f"""
        <tr>
            <td><strong>{display_name}</strong></td>
            <td style="{score_style}">{ws:.3f}</td>
            <td>{period_str}</td>
            <td>{prevalence:.1f}%</td>
            <td>{wave_mins:,}</td>
            <td>{amplitude:.0f}W</td>
        </tr>"""

    return f"""
    <div class="overview-grid">
        <div class="overview-item">
            <div class="overview-value">{cls_badge}</div>
            <div class="overview-label">Classification</div>
        </div>
        <div class="overview-item">
            <div class="overview-value" style="color: {'#e67e22' if max_score >= 0.5 else '#3498db' if max_score >= 0.25 else '#95a5a6'};">{max_score:.3f}</div>
            <div class="overview-label">Max Wave Score</div>
            <div class="overview-desc">0 = no cycling, 1 = perfect regular cycling</div>
        </div>
        <div class="overview-item">
            <div class="overview-value">{', '.join(dominant_phases) if dominant_phases else '—'}</div>
            <div class="overview-label">Dominant Phases</div>
            <div class="overview-desc">Phases with wave score &ge; 0.25</div>
        </div>
        <div class="overview-item">
            <div class="overview-value">{season_text}</div>
            <div class="overview-label">Peak Season</div>
            <div class="overview-desc">When wave activity concentrates</div>
        </div>
    </div>
    <table style="width:100%; border-collapse:collapse; margin-top:15px; font-size:0.9em;">
        <thead>
            <tr style="background:#f8f9fa; text-align:left;">
                <th style="padding:8px 12px;">Phase</th>
                <th style="padding:8px 12px;">Wave Score</th>
                <th style="padding:8px 12px;">Cycle Period</th>
                <th style="padding:8px 12px;">Prevalence</th>
                <th style="padding:8px 12px;">Wave Minutes</th>
                <th style="padding:8px 12px;">Avg Amplitude</th>
            </tr>
        </thead>
        <tbody>
            {phase_rows}
        </tbody>
    </table>
    <div class="chart-card" style="margin-top:15px;">
        {wave_chart}
    </div>
    """


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

    # Wave behavior
    wave_behavior = analysis.get('wave_behavior', {})
    wave_chart = create_wave_monthly_chart(analysis)

    # Build findings tags
    findings_html = _build_findings_tags(flags, quality, coverage, wave_behavior)

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

    # Build shared about / glossary sections
    about_html = _build_about_section('house_pre_analysis')
    glossary_html = _build_glossary_section()

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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            background-color: #FAFBFF;
            color: #3D3D50;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        header {{
            background: linear-gradient(135deg, #7B9BC4 0%, #B488B4 100%);
            color: white;
            padding: 30px;
            border-radius: 16px;
            margin-bottom: 20px;
        }}
        .card {{
            background: #FFFFFF;
            border-radius: 14px;
            padding: 22px;
            margin-bottom: 20px;
            box-shadow: 0 2px 12px rgba(120,100,160,0.07);
            border: 1px solid #E8E4F0;
        }}
        .card h2 {{ margin-top: 0; border-bottom: 2px solid #E8E4F0; padding-bottom: 10px; color: #3D3D50; }}
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
        .metric-value {{ font-size: 1.6em; font-weight: bold; color: #7B9BC4; }}
        .metric-label {{ color: #7D7D92; font-size: 0.85em; }}
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
        .badge-green {{ background: #D8F0E0; color: #3A6A4A; }}
        .badge-blue {{ background: #D0E4F4; color: #2A5A7A; }}
        .badge-orange {{ background: #F5ECD5; color: #6A5A2A; }}
        .badge-red {{ background: #F5D8D8; color: #6A3030; }}
        .badge-purple {{ background: #E5D8F0; color: #5A3A7A; }}
        .badge-purple-light {{ background: #d4c5e2; color: #5a3d7a; }}
        .badge-purple-dark {{ background: #c9a3d4; color: #4a0e6b; }}
        .flags {{ color: #e74c3c; }}
        .back-link {{ margin-bottom: 15px; }}
        .back-link a {{ color: #7B9BC4; text-decoration: none; }}
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
            background: linear-gradient(135deg, #7B9BC4 0%, #B488B4 100%);
            color: white;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Monthly details */
        .expand-btn {{
            padding: 10px 20px;
            border: 2px solid #7B9BC4;
            background: white;
            color: #7B9BC4;
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

        /* Findings tags */
        .findings-section {{
            margin-bottom: 20px;
        }}
        .findings-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}
        .finding-tag {{
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            cursor: pointer;
            text-decoration: none;
            transition: transform 0.15s, box-shadow 0.15s;
            display: inline-block;
        }}
        .finding-tag:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .finding-tag-critical {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .finding-tag-warning {{
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
        }}
        .finding-tag-info {{
            background: #cce5ff;
            color: #004085;
            border: 1px solid #b8daff;
        }}
        .findings-ok {{
            padding: 10px 16px;
            background: #d4edda;
            color: #155724;
            border-radius: 8px;
            font-size: 0.9em;
            font-weight: 600;
        }}
        .penalty-summary {{
            margin-top: 10px;
            padding: 10px 14px;
            background: #fff3cd;
            border-radius: 8px;
            font-size: 0.85em;
            color: #856404;
            border: 1px solid #ffeeba;
        }}
        html {{
            scroll-behavior: smooth;
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

        {about_html}

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
                    {'Base score: ' + str(round(quality.get('base_quality_score', score))) + ' &minus; ' + str(round(quality.get('anomaly_penalties', 0))) + ' anomaly penalty' if quality.get('anomaly_penalties', 0) > 0 else 'Computed from 6 components: Sharp Entry Rate, Device Signature, Power Profile, Variability, Data Volume, Data Integrity'}
                </div>
            </div>

            {findings_html}

            <!-- Data Overview -->
            <div class="card" id="section-data-overview">
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
            <div class="card" id="section-data-loss">
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
            <div class="card" id="section-power-stats">
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

            <!-- Wave Behavior Pre-Analysis -->
            <div class="card" id="section-wave-behavior">
                <h2>Wave Behavior Pre-Analysis</h2>
                <p style="color: #666; font-size: 0.88em; margin-bottom: 15px;">
                    Detects periodic cycling patterns (AC compressor) in raw power data.
                    Uses autocorrelation of minute-to-minute diffs at lags 3&ndash;30 min.
                    High wave score = regular ON/OFF cycling within 800&ndash;4000W windows.
                </p>
                {_build_wave_behavior_section(wave_behavior, wave_chart)}
            </div>

            <!-- Charts -->
            <div class="card" id="section-charts">
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

            <div class="card" id="section-flags">
                <h2>Issues & Flags</h2>
                <p class="flags">{flags_html}</p>
            </div>
        </div>

        <!-- Year Tabs Content -->
        {year_sections_html}

        {glossary_html}

        <footer style="text-align: center; color: #888; padding: 20px;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | ElectricPatterns
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


