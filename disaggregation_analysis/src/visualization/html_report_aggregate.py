"""
Aggregate report section builders for disaggregation analysis.

Extracted from html_report.py — contains multi-house comparison functions.
"""
import logging
import numpy as np
from typing import List, Dict, Any

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from shared.html_utils import assign_tier as _assign_tier

from visualization.charts import (
    create_score_distribution_chart,
    create_matching_rate_distribution_chart,
    create_segmentation_ratio_distribution_chart,
    create_tag_breakdown_chart,
    create_experiment_summary_table,
    create_duration_distribution_chart,
    create_device_detection_chart,
)

logger = logging.getLogger(__name__)


def _extract_house_data(analyses: List[Dict[str, Any]]) -> list:
    """Extract per-house metrics for JavaScript filtering."""
    house_data = []
    for a in analyses:
        if a.get('status') == 'no_data':
            continue

        house_id = a.get('house_id', 'unknown')
        iterations = a.get('iterations', {}) or {}
        scores = a.get('scores', {}) or {}
        first = a.get('first_iteration', {}) or {}
        matching = first.get('matching', {}) or {}
        seg = first.get('segmentation', {}) or {}
        flags = a.get('flags', {}) or {}
        patterns = first.get('patterns', {}) or {}
        th_expl = a.get('threshold_explanation', {}) or {}

        pre_quality = a.get('pre_analysis_quality_score', None)
        tier = _assign_tier(pre_quality)

        # Minutes ratio calculation
        matched_minutes = matching.get('total_matched_minutes', 0) or 0
        total_days = (patterns.get('daily_stats', {}) or {}).get('total_days', 0) or 0
        if total_days > 0 and matched_minutes > 0:
            minutes_ratio = matched_minutes / (total_days * 24 * 60)
        else:
            minutes_ratio = 0

        # Device detection
        ac_detection = patterns.get('ac_detection', {}) or {}
        boiler_detection = patterns.get('boiler_detection', {}) or {}

        house_data.append({
            'id': str(house_id),
            'tier': tier,
            'matching_rate': (iterations.get('first_iter_matching_rate', 0) or 0),
            'seg_ratio': (seg.get('segmentation_ratio', 0) or 0),
            'minutes_ratio': minutes_ratio,
            'th_explanation_rate': (th_expl.get('total_explanation_rate', 0) or 0),
            'overall_score': (scores.get('overall_score', 0) or 0),
            'matching_score': (scores.get('matching_score', 0) or 0),
            'tags': matching.get('tag_breakdown', {}) or {},
            'durations': matching.get('duration_breakdown', {}) or {},
            'has_central_ac': ac_detection.get('has_central_ac', False),
            'central_ac_count': (ac_detection.get('central_ac', {}) or {}).get('total_count', 0) or 0,
            'has_regular_ac': bool(ac_detection.get('regular_ac_by_phase', {}) or (ac_detection.get('regular_ac', {}) or {}).get('activations')),
            'regular_ac_count': (ac_detection.get('regular_ac', {}) or {}).get('total_count', 0) or 0,
            'has_boiler': boiler_detection.get('has_boiler', False),
            'boiler_count': (boiler_detection.get('boiler', {}) or {}).get('total_count', 0) or 0,
            'has_recurring_patterns': (patterns.get('recurring_matches', {}) or {}).get('pattern_count', 0) > 0,
            'total_days': total_days,
            'matched_minutes': matched_minutes,
            'total_matches': matching.get('total_matches', 0) or 0,
            'flags': {k: bool(v) for k, v in flags.items()},
        })

        # Classification data (dynamic threshold experiments only)
        classification = a.get('classification', {})
        if classification:
            house_data[-1]['classified_rate'] = classification.get('classified_rate', 0)
            house_data[-1]['device_power_pct'] = classification.get('device_power_pct', 0)
            house_data[-1]['total_segregated_power_pct'] = classification.get('total_segregated_power_pct', 0)
            house_data[-1]['device_breakdown'] = classification.get('device_breakdown', {})
            house_data[-1]['has_classification'] = True
        else:
            house_data[-1]['has_classification'] = False

    return house_data


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


def _generate_summary_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate executive summary HTML."""
    import numpy as np
    valid = [a for a in analyses if a.get('status') != 'no_data']
    n_houses = len(valid)

    if n_houses == 0:
        return '<p>No experiment data found.</p>'

    # Calculate averages
    overall_scores = [(a.get('scores', {}) or {}).get('overall_score', 0) or 0 for a in valid]
    matching_scores = [(a.get('scores', {}) or {}).get('matching_score', 0) or 0 for a in valid]
    seg_ratios = [((a.get('first_iteration', {}) or {}).get('segmentation', {}) or {}).get('segmentation_ratio', 0) or 0 for a in valid]
    total_matches = sum(((a.get('first_iteration', {}) or {}).get('matching', {}) or {}).get('total_matches', 0) or 0 for a in valid)

    # Threshold explanation rates
    th_explanation_rates = [(a.get('threshold_explanation', {}) or {}).get('total_explanation_rate', 0) or 0 for a in valid]

    avg_overall = sum(overall_scores) / n_houses if n_houses > 0 else 0
    avg_matching = sum(matching_scores) / n_houses if n_houses > 0 else 0
    avg_seg = sum(seg_ratios) / n_houses if n_houses > 0 else 0
    avg_th_expl = np.mean(th_explanation_rates) if th_explanation_rates else 0
    std_th_expl = np.std(th_explanation_rates) if th_explanation_rates else 0

    # Count issues
    n_low_matching = sum(1 for a in valid if (a.get('flags', {}) or {}).get('low_matching_rate', False))
    n_negative = sum(1 for a in valid if (a.get('flags', {}) or {}).get('has_negative_values', False))
    n_low_seg = sum(1 for a in valid if (a.get('flags', {}) or {}).get('low_segmentation', False))

    # Color for threshold explanation rate
    th_color = '#28a745' if avg_th_expl >= 0.8 else '#ffc107' if avg_th_expl >= 0.5 else '#dc3545'

    # Classification metrics (dynamic threshold experiments only)
    classification_houses = [a for a in valid if a.get('classification')]
    has_classification = len(classification_houses) > 0
    classification_cards = ''
    if has_classification:
        classified_rates = [a['classification'].get('classified_rate', 0) for a in classification_houses]
        device_power_pcts = [a['classification'].get('device_power_pct', 0) for a in classification_houses]
        avg_classified = np.mean(classified_rates) if classified_rates else 0
        avg_device_power = np.mean(device_power_pcts) if device_power_pcts else 0
        cls_color = '#28a745' if avg_classified >= 0.7 else '#ffc107' if avg_classified >= 0.4 else '#dc3545'
        dev_color = '#28a745' if avg_device_power >= 0.5 else '#ffc107' if avg_device_power >= 0.3 else '#dc3545'
        classification_cards = f"""
        <div class="summary-card" style="border: 2px solid {cls_color}; background: linear-gradient(135deg, #fff 0%, {cls_color}22 100%);">
            <div class="summary-number" style="color: {cls_color};">{avg_classified:.1%}</div>
            <div class="summary-label">Classification Rate</div>
            <div style="font-size: 0.7em; color: #888;">classified / total matches</div>
        </div>
        <div class="summary-card" style="border: 2px solid {dev_color}; background: linear-gradient(135deg, #fff 0%, {dev_color}22 100%);">
            <div class="summary-number" style="color: {dev_color};">{avg_device_power:.1%}</div>
            <div class="summary-label">Device Power Segregated</div>
            <div style="font-size: 0.7em; color: #888;">classified device W / total W</div>
        </div>
        """

    return f"""
    <div class="summary-grid" id="summary-cards">
        <div class="summary-card">
            <div class="summary-number" id="summary-houses-count">{n_houses}</div>
            <div class="summary-label">Houses Analyzed</div>
        </div>
        <div class="summary-card" id="summary-th-card" style="border: 2px solid {th_color}; background: linear-gradient(135deg, #fff 0%, {th_color}22 100%);">
            <div class="summary-number" id="summary-th-explanation" style="color: {th_color};">{avg_th_expl:.1%}</div>
            <div class="summary-label">High-Power Energy Segregated</div>
            <div id="summary-th-std" style="font-size: 0.8em; color: #666;">(&gt;1300W) \u00b1{std_th_expl:.1%} std</div>
        </div>
        {classification_cards}
        <div class="summary-card">
            <div class="summary-number" id="summary-overall-score">{avg_overall:.0f}</div>
            <div class="summary-label">Avg Overall Score</div>
            <div style="font-size: 0.7em; color: #888;">70% match + 30% seg</div>
        </div>
        <div class="summary-card">
            <div class="summary-number" id="summary-matching-score">{avg_matching:.0f}%</div>
            <div class="summary-label">Avg Matching Score</div>
            <div style="font-size: 0.7em; color: #888;">matched / total events</div>
        </div>
        <div class="summary-card">
            <div class="summary-number" id="summary-seg-ratio">{avg_seg:.1%}</div>
            <div class="summary-label">Avg Segmentation</div>
            <div style="font-size: 0.7em; color: #888;">segregated W / original W</div>
        </div>
    </div>
    <p style="font-size: 0.8em; color: #666; margin: 10px 0;">
        <strong>Note:</strong> All phase metrics are averaged across 3 phases (w1, w2, w3).
        Damaged phases count as 0 (not excluded). Simple average across houses (not time-weighted).
    </p>
    <div class="summary-alerts" id="summary-alerts">
        <h3>Issues Summary</h3>
        <ul>
            <li class="{'alert' if n_low_matching > 0 else ''}">
                <strong id="summary-low-matching">{n_low_matching}</strong> houses with low matching rate (&lt;50%)
            </li>
            <li class="{'alert' if n_low_seg > 0 else ''}">
                <strong id="summary-low-seg">{n_low_seg}</strong> houses with low segmentation (&lt;30%)
            </li>
            <li class="{'alert' if n_negative > 0 else ''}">
                <strong id="summary-negative">{n_negative}</strong> houses with negative power values
            </li>
        </ul>
    </div>
    """


def _generate_comparison_table(analyses: List[Dict[str, Any]]) -> str:
    """Generate sortable comparison table HTML."""
    rows = []

    # Check if any house has classification data
    has_classification = any(a.get('classification') for a in analyses if a.get('status') != 'no_data')

    for a in analyses:
        if a.get('status') == 'no_data':
            continue

        house_id = a.get('house_id', 'unknown')
        iterations = a.get('iterations', {}) or {}
        scores = a.get('scores', {}) or {}
        first = a.get('first_iteration', {}) or {}
        matching = first.get('matching', {}) or {}
        seg = first.get('segmentation', {}) or {}
        flags = a.get('flags', {}) or {}
        patterns = first.get('patterns', {}) or {}

        # Device detection
        ac_detection = patterns.get('ac_detection', {}) or {}
        boiler_detection = patterns.get('boiler_detection', {}) or {}

        has_central_ac = ac_detection.get('has_central_ac', False)
        has_regular_ac = bool(ac_detection.get('regular_ac_by_phase', {}) or (ac_detection.get('regular_ac', {}) or {}).get('activations'))
        has_boiler = boiler_detection.get('has_boiler', False)

        # Device badges (colored text instead of emoji)
        central_ac_icon = '<span style="background:#A8D8EA;color:#2D5F7C;padding:2px 8px;border-radius:4px;font-size:0.85em;margin-right:3px;" title="Central AC">Central AC</span>' if has_central_ac else ''
        regular_ac_icon = '<span style="background:#B8E6C8;color:#2D6A3E;padding:2px 8px;border-radius:4px;font-size:0.85em;margin-right:3px;" title="Regular AC">Regular AC</span>' if has_regular_ac else ''
        boiler_icon = '<span style="background:#FFCBA4;color:#7A4A1E;padding:2px 8px;border-radius:4px;font-size:0.85em;margin-right:3px;" title="Boiler">Boiler</span>' if has_boiler else ''
        devices = f"{central_ac_icon}{regular_ac_icon}{boiler_icon}".strip() or '-'

        # Flag badges (colored text instead of emoji)
        flag_icons = []
        if flags.get('low_matching_rate'):
            flag_icons.append('<span style="background:#F5D8D8;color:#6A3030;padding:2px 6px;border-radius:4px;font-size:0.78em;" title="Low Matching Rate (<50%)">Low Match</span>')
        if flags.get('has_negative_values'):
            flag_icons.append('<span style="background:#F5ECD5;color:#6A5A2A;padding:2px 6px;border-radius:4px;font-size:0.78em;" title="Negative Power Values">Negatives</span>')
        if flags.get('low_segmentation'):
            flag_icons.append('<span style="background:#F5D8D8;color:#6A3030;padding:2px 6px;border-radius:4px;font-size:0.78em;" title="Low Segmentation (<30%)">Low Seg</span>')
        if flags.get('has_damaged_phases'):
            flag_icons.append('<span style="background:#E5D8F0;color:#5A3A7A;padding:2px 6px;border-radius:4px;font-size:0.78em;" title="Damaged Phase(s)">Damaged</span>')
        if flags.get('has_recurring_patterns'):
            flag_icons.append('<span style="background:#D0E4F4;color:#2A5A7A;padding:2px 6px;border-radius:4px;font-size:0.78em;" title="Has Recurring Patterns">Recurring</span>')
        flags_display = ' '.join(flag_icons) if flag_icons else '-'

        # Calculate minutes segmentation ratio (NOT x3 - real time only)
        matched_minutes = matching.get('total_matched_minutes', 0) or 0
        total_days = (patterns.get('daily_stats', {}) or {}).get('total_days', 0) or 0
        if total_days > 0 and matched_minutes > 0:
            total_available_minutes = total_days * 24 * 60  # Real time, NOT x3
            minutes_seg_ratio = matched_minutes / total_available_minutes
        else:
            minutes_seg_ratio = 0

        # Threshold explanation rate
        th_expl = a.get('threshold_explanation', {}) or {}
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
        tier = _assign_tier(pre_quality)
        if isinstance(pre_quality, str) and pre_quality.startswith('faulty'):
            _faulty_labels = {
                'faulty_dead_phase': ('Dead Phase', 'Phase with <2% of sisters avg'),
                'faulty_high_nan': ('High NaN', 'Phase with >=10% NaN values'),
                'faulty_both': ('Both', 'Dead phase + high NaN on other phases'),
            }
            _fl, _ft = _faulty_labels.get(pre_quality, ('Faulty', ''))
            pre_quality_html = f'<span style="color: #6f42c1; font-weight: bold;" title="{_ft}">{_fl}</span>'
        elif pre_quality is not None:
            pre_q_color = '#28a745' if pre_quality >= 75 else '#ffc107' if pre_quality >= 50 else '#dc3545'
            pre_quality_html = f'<span style="color: {pre_q_color}; font-weight: bold;">{pre_quality:.0f}</span>'
        else:
            pre_quality_html = '<span style="color: #999;">-</span>'

        # Classification columns (dynamic threshold experiments only)
        classification_cells = ''
        if has_classification:
            cls = a.get('classification', {})
            if cls:
                cls_rate = cls.get('classified_rate', 0)
                dev_pwr = cls.get('device_power_pct', 0)
                cls_color = '#28a745' if cls_rate >= 0.7 else '#ffc107' if cls_rate >= 0.4 else '#dc3545'
                dev_color = '#28a745' if dev_pwr >= 0.5 else '#ffc107' if dev_pwr >= 0.3 else '#dc3545'
                classification_cells = f"""
            <td style="color: {cls_color}; font-weight: bold;">{cls_rate:.1%}</td>
            <td style="color: {dev_color}; font-weight: bold;">{dev_pwr:.1%}</td>"""
            else:
                classification_cells = """
            <td style="color: #999;">-</td>
            <td style="color: #999;">-</td>"""

        rows.append(f"""
        <tr data-tier="{tier}">
            <td><a href="house_reports/house_{house_id}.html" target="_blank" style="text-decoration: none; color: #1976d2;"><strong>{house_id}</strong></a></td>
            <td>{pre_quality_html}</td>
            <td>{iterations.get('iterations_completed', 0) or 0}</td>
            <td>{(iterations.get('first_iter_matching_rate', 0) or 0):.1%}</td>
            <td>{(seg.get('segmentation_ratio', 0) or 0):.1%}</td>
            <td>{minutes_seg_ratio:.2%}</td>
            <td style="color: {th_color}; font-weight: bold;">{th_explanation_rate:.1%}</td>{classification_cells}
            <td style="font-size: 1.2em;">{devices}</td>
            <td>{score:.0f} {badge}</td>
            <td style="font-size: 1.2em;">{flags_display}</td>
        </tr>
        """)

    # Classification header columns
    classification_legend = ''
    classification_headers = ''
    if has_classification:
        col_offset = 7  # classification columns start after column index 6 (High-Power Segregated)
        classification_legend = """
        <strong>Classified</strong> = classified matches / total matches |
        <strong>Device Power</strong> = classified device watts / total watts |"""
        classification_headers = f"""
                <th onclick="sortTable({col_offset})">Classified<br><small>(rate)</small></th>
                <th onclick="sortTable({col_offset + 1})">Device Power<br><small>(segregated)</small></th>"""
        # Adjust sort indices for columns after classification
        devices_idx = col_offset + 2
        score_idx = col_offset + 3
        flags_idx = col_offset + 4
    else:
        devices_idx = 7
        score_idx = 8
        flags_idx = 9

    return f"""
    <p style="font-size: 0.85em; color: #666; margin-bottom: 10px;">
        <strong>Pre-Quality</strong> = quality score from pre-experiment analysis (coverage+days+data quality) |
        <strong>Match Rate</strong> = matched events / total events |
        <strong>Seg (power)</strong> = segregated watts / original watts |
        <strong>Seg (minutes)</strong> = matched minutes / total data minutes |
        <strong>High-Power Segregated</strong> = % of minutes &gt;1300W where segregation brought remaining below threshold |{classification_legend}
        <strong>Exp Score</strong> = 70% match + 30% seg (3-phase avg, damaged=0)
    </p>
    <p style="font-size: 0.85em; color: #7D7D92; margin-bottom: 10px;">
        <strong>Devices:</strong>
        <span style="background:#A8D8EA;color:#2D5F7C;padding:1px 6px;border-radius:4px;font-size:0.85em;">Central AC</span>
        <span style="background:#B8E6C8;color:#2D6A3E;padding:1px 6px;border-radius:4px;font-size:0.85em;">Regular AC</span>
        <span style="background:#FFCBA4;color:#7A4A1E;padding:1px 6px;border-radius:4px;font-size:0.85em;">Boiler</span> |
        <strong>Issues:</strong>
        <span style="background:#F5D8D8;color:#6A3030;padding:1px 6px;border-radius:4px;font-size:0.85em;">Low Match</span>
        <span style="background:#F5ECD5;color:#6A5A2A;padding:1px 6px;border-radius:4px;font-size:0.85em;">Negatives</span>
        <span style="background:#F5D8D8;color:#6A3030;padding:1px 6px;border-radius:4px;font-size:0.85em;">Low Seg</span>
        <span style="background:#E5D8F0;color:#5A3A7A;padding:1px 6px;border-radius:4px;font-size:0.85em;">Damaged</span>
        <span style="background:#D0E4F4;color:#2A5A7A;padding:1px 6px;border-radius:4px;font-size:0.85em;">Recurring</span>
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
                <th onclick="sortTable(6)">High-Power<br><small>Segregated</small></th>{classification_headers}
                <th onclick="sortTable({devices_idx})">Devices</th>
                <th onclick="sortTable({score_idx})">Exp Score</th>
                <th onclick="sortTable({flags_idx})">Flags</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    """


def _create_classification_overview_chart(classification_houses: List[Dict[str, Any]]) -> str:
    """Create a stacked bar chart showing device classification breakdown per house."""
    house_ids = []
    boiler_counts = []
    central_ac_counts = []
    regular_ac_counts = []
    unclassified_counts = []

    for a in classification_houses:
        house_id = str(a.get('house_id', 'unknown'))
        cls = a.get('classification', {})
        breakdown = cls.get('device_breakdown', {})

        house_ids.append(house_id)
        boiler_counts.append((breakdown.get('boiler', {}) or {}).get('count', 0))
        central_ac_counts.append((breakdown.get('central_ac', {}) or {}).get('count', 0))
        regular_ac_counts.append((breakdown.get('regular_ac', {}) or {}).get('count', 0))
        unclassified_counts.append((breakdown.get('unclassified', {}) or {}).get('count', 0))

    if not house_ids:
        return ''

    chart_id = 'classification-overview-chart'
    traces = [
        {'x': house_ids, 'y': boiler_counts, 'name': 'Boiler', 'type': 'bar',
         'marker': {'color': '#dc3545'}},
        {'x': house_ids, 'y': central_ac_counts, 'name': 'Central AC', 'type': 'bar',
         'marker': {'color': '#007bff'}},
        {'x': house_ids, 'y': regular_ac_counts, 'name': 'Regular AC', 'type': 'bar',
         'marker': {'color': '#28a745'}},
        {'x': house_ids, 'y': unclassified_counts, 'name': 'Unclassified', 'type': 'bar',
         'marker': {'color': '#6c757d'}},
    ]

    layout = {
        'title': 'Device Classification per House',
        'barmode': 'stack',
        'xaxis': {'title': 'House ID', 'type': 'category'},
        'yaxis': {'title': 'Number of Matches'},
        'legend': {'orientation': 'h', 'y': -0.2},
        'height': 400,
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', {json.dumps(traces)}, {json.dumps(layout)});
    </script>
    '''


def _generate_charts_section(analyses: List[Dict[str, Any]]) -> str:
    """Generate all charts HTML."""

    # Start with summary table (most important)
    summary_table = create_experiment_summary_table(analyses)

    # Organize charts by category (5 charts total + conditional classification)
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

    # Add classification chart if any house has classification data
    classification_houses = [a for a in analyses if a.get('classification') and a.get('status') != 'no_data']
    if classification_houses:
        cls_chart = _create_classification_overview_chart(classification_houses)
        if cls_chart:
            charts_by_category['Device Classification'] = [cls_chart]

    html_parts = []

    # Summary table first (full width)
    html_parts.append(f"""
    <div class="summary-section" id="summary-table-container">
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

