"""
Aggregate report section builders for house analysis.

Extracted from html_report.py — multi-house comparison functions.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from shared.html_utils import build_quality_dist_bar as _build_quality_dist_bar

from typing import List, Dict, Any

from visualization.charts import (
    create_quality_distribution_chart,
    create_phase_balance_chart,
    create_issues_heatmap,
    create_wave_comparison_chart,
    create_imbalance_comparison_chart,
)


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

    # Compute tier counts for quality distribution bar
    tier_counts = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0,
                   'faulty_dead_phase': 0, 'faulty_high_nan': 0, 'faulty_both': 0, 'unknown': 0}
    for a in analyses:
        score = a.get('data_quality', {}).get('quality_score', 0)
        qlabel = a.get('flags', {}).get('quality_label')
        if qlabel == 'faulty_both':
            tier_counts['faulty_both'] += 1
        elif qlabel == 'faulty_dead_phase':
            tier_counts['faulty_dead_phase'] += 1
        elif qlabel == 'faulty_high_nan':
            tier_counts['faulty_high_nan'] += 1
        elif score >= 90:
            tier_counts['excellent'] += 1
        elif score >= 75:
            tier_counts['good'] += 1
        elif score >= 50:
            tier_counts['fair'] += 1
        else:
            tier_counts['poor'] += 1

    quality_bar = _build_quality_dist_bar(tier_counts, n_houses)

    return f"""
    <!-- Row 1: Houses + Days -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:18px;">
        <div class="summary-card">
            <div class="summary-number" id="summary-total">{n_houses}</div>
            <div class="summary-label">Houses Analyzed</div>
        </div>
        <div class="summary-card">
            <div class="summary-number" id="summary-total-days">{total_days:,}</div>
            <div class="summary-label">Total Days of Data</div>
        </div>
    </div>
    <!-- Row 2: Quality distribution -->
    {quality_bar}
    <!-- Row 3: Report-specific metrics -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:18px;">
        <div class="summary-card">
            <div class="summary-number" id="summary-avg-score">{avg_quality:.0f}</div>
            <div class="summary-label">Avg Quality Score</div>
        </div>
        <div class="summary-card">
            <div class="summary-number" id="summary-avg-coverage">{avg_coverage:.1%}</div>
            <div class="summary-label">Avg Coverage</div>
        </div>
    </div>
    """


def _generate_wave_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate wave behavior summary as a separate section."""
    n_wave_dominant = sum(1 for a in analyses if a.get('wave_behavior', {}).get('wave_classification') == 'wave_dominant')
    n_has_waves = sum(1 for a in analyses if a.get('wave_behavior', {}).get('wave_classification') == 'has_waves')
    n_no_waves = sum(1 for a in analyses if a.get('wave_behavior', {}).get('wave_classification') == 'no_waves')

    return f"""
    <div class="summary-grid">
        <div class="summary-card" style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);">
            <div class="summary-number" style="color: #e67e22;">{n_wave_dominant}</div>
            <div class="summary-label">Wave Dominant</div>
        </div>
        <div class="summary-card" style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);">
            <div class="summary-number" style="color: #3498db;">{n_has_waves}</div>
            <div class="summary-label">Has Waves</div>
        </div>
        <div class="summary-card" style="background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);">
            <div class="summary-number" style="color: #95a5a6;">{n_no_waves}</div>
            <div class="summary-label">No Waves</div>
        </div>
    </div>
    """


def _generate_comparison_table(analyses: List[Dict[str, Any]],
                                per_house_dir: str = '../per_house') -> tuple:
    """Generate sortable comparison table HTML with links to individual reports.

    Returns:
        Tuple of (html_string, tier_counts, continuity_counts, wave_counts)
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
        'high_phase_imbalance': 'High Phase Imbalance',
    }

    tier_counts = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0,
                   'faulty_dead_phase': 0, 'faulty_high_nan': 0, 'faulty_both': 0}
    continuity_counts = {'continuous': 0, 'minor_gaps': 0,
                         'discontinuous': 0, 'fragmented': 0, 'unknown': 0}
    wave_counts = {'wave_dominant': 0, 'has_waves': 0, 'no_waves': 0}
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
            tier_key = 'faulty_dead_phase'
        elif qlabel == 'faulty_high_nan':
            badge = '<span class="badge badge-purple">Faulty (High NaN)</span>'
            tier_key = 'faulty_high_nan'
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

        # Phase imbalance
        imbalance = a.get('phase_imbalance', {})
        imb_mean = imbalance.get('imbalance_mean', 0)
        imb_label = imbalance.get('imbalance_label', 'balanced')
        if imb_mean > 0.5:
            imb_color = '#dc3545'
        elif imb_mean > 0.2:
            imb_color = '#e67e22'
        else:
            imb_color = '#28a745'

        # Wave behavior
        wave = a.get('wave_behavior', {})
        wave_cls = wave.get('wave_classification', 'no_waves')

        # Wave classification count (must be after wave_cls assignment)
        wave_counts[wave_cls] = wave_counts.get(wave_cls, 0) + 1
        wave_score = wave.get('max_wave_score', 0)
        wave_badge_map = {
            'wave_dominant': '<span class="badge badge-orange">Wave Dominant</span>',
            'has_waves': '<span class="badge badge-blue">Has Waves</span>',
            'no_waves': '<span style="color: #95a5a6;">No Waves</span>',
        }
        wave_badge = wave_badge_map.get(wave_cls, '')

        rows.append(f"""
        <tr data-tier="{tier_key}" data-continuity="{continuity}" data-wave="{wave_cls}"
            data-house-id="{house_id}"
            data-excluded="false" data-score="{score:.1f}" data-coverage="{cov_ratio:.4f}"
            data-nan="{no_data_pct:.2f}" data-days="{days_span}"
            onclick="toggleExcludeRow(event, this)">
            <td><a href="{house_link}" class="house-link"><strong>{house_id}</strong></a></td>
            <td>{days_span}</td>
            <td>{cov_ratio:.1%}</td>
            <td style="color: {'#6f42c1' if no_data_pct >= 10 else 'inherit'};">{no_data_pct:.1f}%</td>
            <td>{score:.0f} {badge}</td>
            <td>{wave_score:.2f} {wave_badge}</td>
            <td>{power.get('total_mean', 0):.0f}</td>
            <td>{power.get('phase_balance_ratio', 0):.2f}</td>
            <td style="color: {imb_color}; font-weight: {'bold' if imb_mean > 0.5 else 'normal'};">{imb_mean:.3f}</td>
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
        <strong>Imbalance</strong> = per-minute std/mean across phases, 0=balanced |
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
                <th onclick="sortTable(5)">Wave<br><small>(score)</small></th>
                <th onclick="sortTable(6)">Avg Power<br><small>(Watts)</small></th>
                <th onclick="sortTable(7)">Phase Balance<br><small>(max/min)</small></th>
                <th onclick="sortTable(8)">Imbalance<br><small>(std/mean)</small></th>
                <th onclick="sortTable(9)">Night/Day<br><small>(power ratio)</small></th>
                <th>Issues</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    </div>
    """
    return html, tier_counts, continuity_counts, wave_counts


def _build_filter_bar(tier_counts: Dict[str, int],
                      continuity_counts: Dict[str, int],
                      wave_counts: Dict[str, int] = None) -> str:
    """Build the filter bar with tier, NaN continuity, and wave checkboxes."""
    tier_labels = {
        'excellent': ('Excellent', '#28a745'),
        'good': ('Good', '#007bff'),
        'fair': ('Fair', '#ffc107'),
        'poor': ('Poor', '#dc3545'),
        'faulty_dead_phase': ('Faulty (Dead)', '#5a3d7a'),
        'faulty_high_nan': ('Faulty (NaN)', '#6f42c1'),
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

    # Wave classification checkboxes
    wave_labels = {
        'wave_dominant': ('Wave Dominant', '#e67e22'),
        'has_waves': ('Has Waves', '#3498db'),
        'no_waves': ('No Waves', '#95a5a6'),
    }
    wave_html = ''
    if wave_counts:
        for key, (label, color) in wave_labels.items():
            count = wave_counts.get(key, 0)
            if count == 0:
                continue
            wave_html += f"""
            <label class="filter-checkbox" style="border-color: {color};">
                <input type="checkbox" checked onchange="updateFilter()" data-filter-wave="{key}">
                <span class="filter-dot" style="background: {color};"></span>
                {label} <span class="filter-count">({count})</span>
            </label>"""

    wave_group = f"""
        <div class="filter-group">
            <span class="filter-group-label">Wave Behavior:</span>
            {wave_html}
        </div>""" if wave_html else ''

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
        {wave_group}
    </div>
    """


def _generate_charts_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate all charts HTML."""
    charts = [
        create_quality_distribution_chart(analyses),
        create_wave_comparison_chart(analyses),
        create_phase_balance_chart(analyses),
        create_imbalance_comparison_chart(analyses),
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


