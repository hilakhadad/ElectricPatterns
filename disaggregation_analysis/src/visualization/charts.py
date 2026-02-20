"""
Chart generation for experiment analysis reports.

Creates meaningful visualizations that show distributions and aggregations,
not per-house comparisons with meaningless house ID ordering.
"""
from typing import List, Dict, Any
import json


def create_score_distribution_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create histogram of overall scores.

    Args:
        analyses: List of experiment analysis results

    Returns:
        HTML string with Plotly chart
    """
    scores = [a.get('scores', {}).get('overall_score', 0)
              for a in analyses if a.get('status') != 'no_data']

    if not scores:
        return '<p>No data available</p>'

    # Create bins for histogram
    bins = {'0-20': 0, '20-40': 0, '40-60': 0, '60-80': 0, '80-100': 0}
    for score in scores:
        if score < 20:
            bins['0-20'] += 1
        elif score < 40:
            bins['20-40'] += 1
        elif score < 60:
            bins['40-60'] += 1
        elif score < 80:
            bins['60-80'] += 1
        else:
            bins['80-100'] += 1

    chart_id = 'score-dist-chart'
    data = {
        'x': list(bins.keys()),
        'y': list(bins.values()),
        'type': 'bar',
        'marker': {
            'color': ['#dc3545', '#fd7e14', '#ffc107', '#20c997', '#28a745']
        }
    }

    layout = {
        'title': 'Overall Score Distribution',
        'xaxis': {'title': 'Score Range'},
        'yaxis': {'title': 'Number of Houses'},
        'showlegend': False
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_matching_rate_distribution_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create histogram showing distribution of matching rates across houses.

    Instead of showing each house as a bar (which implies order),
    this shows how many houses fall into each matching rate range.

    Args:
        analyses: List of experiment analysis results

    Returns:
        HTML string with Plotly chart
    """
    valid = [a for a in analyses if a.get('status') != 'no_data']

    if not valid:
        return '<p>No data available</p>'

    matching_rates = [a.get('iterations', {}).get('first_iter_matching_rate', 0) * 100
                      for a in valid]

    # Create bins for histogram
    bins = {
        '0-20%': 0, '20-40%': 0, '40-60%': 0, '60-80%': 0, '80-100%': 0
    }

    for rate in matching_rates:
        if rate < 20:
            bins['0-20%'] += 1
        elif rate < 40:
            bins['20-40%'] += 1
        elif rate < 60:
            bins['40-60%'] += 1
        elif rate < 80:
            bins['60-80%'] += 1
        else:
            bins['80-100%'] += 1

    chart_id = 'matching-dist-chart'

    # Calculate summary stats for display
    avg_rate = sum(matching_rates) / len(matching_rates) if matching_rates else 0
    min_rate = min(matching_rates) if matching_rates else 0
    max_rate = max(matching_rates) if matching_rates else 0

    data = {
        'x': list(bins.keys()),
        'y': list(bins.values()),
        'type': 'bar',
        'marker': {
            'color': ['#dc3545', '#fd7e14', '#ffc107', '#20c997', '#28a745']
        }
    }

    layout = {
        'title': f'Matching Rate Distribution<br><sub>Avg: {avg_rate:.1f}% | Range: {min_rate:.1f}% - {max_rate:.1f}%</sub>',
        'xaxis': {'title': 'Matching Rate Range'},
        'yaxis': {'title': 'Number of Houses'},
        'showlegend': False
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_segmentation_ratio_distribution_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create histogram showing distribution of segmentation ratios.

    Shows what portion of power was successfully segmented across houses.

    Args:
        analyses: List of experiment analysis results

    Returns:
        HTML string with Plotly chart
    """
    valid = [a for a in analyses if a.get('status') != 'no_data']

    if not valid:
        return '<p>No data available</p>'

    ratios = [a.get('first_iteration', {}).get('segmentation', {}).get('segmentation_ratio', 0) * 100
              for a in valid]

    # Create bins
    bins = {
        '0-10%': 0, '10-20%': 0, '20-30%': 0, '30-50%': 0, '50%+': 0
    }

    for ratio in ratios:
        if ratio < 10:
            bins['0-10%'] += 1
        elif ratio < 20:
            bins['10-20%'] += 1
        elif ratio < 30:
            bins['20-30%'] += 1
        elif ratio < 50:
            bins['30-50%'] += 1
        else:
            bins['50%+'] += 1

    chart_id = 'segmentation-dist-chart'

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0

    data = {
        'x': list(bins.keys()),
        'y': list(bins.values()),
        'type': 'bar',
        'marker': {
            'color': '#28a745'
        }
    }

    layout = {
        'title': f'Segmentation Ratio Distribution<br><sub>Average: {avg_ratio:.1f}% of power segmented</sub>',
        'xaxis': {'title': 'Segmentation Ratio'},
        'yaxis': {'title': 'Number of Houses'},
        'showlegend': False
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_minutes_segmentation_distribution_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create histogram showing distribution of minutes segmentation ratios.

    Shows what portion of time was successfully explained by matches across houses.
    Minutes ratio = matched_minutes / (total_days * 24 * 60) - NOT multiplied by 3

    Args:
        analyses: List of experiment analysis results

    Returns:
        HTML string with Plotly chart
    """
    valid = [a for a in analyses if a.get('status') != 'no_data']

    if not valid:
        return '<p>No data available</p>'

    # Calculate minutes ratio for each house (NOT x3 - real time only)
    ratios = []
    for a in valid:
        matching = a.get('first_iteration', {}).get('matching', {})
        patterns = a.get('first_iteration', {}).get('patterns', {})
        matched_minutes = matching.get('total_matched_minutes', 0)
        total_days = patterns.get('daily_stats', {}).get('total_days', 0)
        if total_days > 0 and matched_minutes > 0:
            total_available_minutes = total_days * 24 * 60  # Real time, NOT x3
            ratio = (matched_minutes / total_available_minutes) * 100
            ratios.append(ratio)
        else:
            ratios.append(0)

    if not ratios or all(r == 0 for r in ratios):
        return '<p>No minutes data available</p>'

    # Create bins (adjusted for higher percentages since NOT x3)
    bins = {
        '0-2%': 0, '2-5%': 0, '5-10%': 0, '10-20%': 0, '20%+': 0
    }

    for ratio in ratios:
        if ratio < 2:
            bins['0-2%'] += 1
        elif ratio < 5:
            bins['2-5%'] += 1
        elif ratio < 10:
            bins['5-10%'] += 1
        elif ratio < 20:
            bins['10-20%'] += 1
        else:
            bins['20%+'] += 1

    chart_id = 'minutes-segmentation-dist-chart'

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0
    min_ratio = min(ratios) if ratios else 0
    max_ratio = max(ratios) if ratios else 0

    data = {
        'x': list(bins.keys()),
        'y': list(bins.values()),
        'type': 'bar',
        'marker': {
            'color': '#17a2b8'
        }
    }

    layout = {
        'title': f'Minutes Segmentation Distribution<br><sub>Avg: {avg_ratio:.2f}% | Range: {min_ratio:.2f}% - {max_ratio:.2f}% of time explained</sub>',
        'xaxis': {'title': 'Minutes Segmentation Ratio'},
        'yaxis': {'title': 'Number of Houses'},
        'showlegend': False
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_tag_breakdown_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create pie chart showing match tag distribution across all houses.

    Args:
        analyses: List of experiment analysis results

    Returns:
        HTML string with Plotly chart
    """
    # Aggregate tag counts
    tag_totals = {'NON-M': 0, 'NOISY': 0, 'PARTIAL': 0, 'SPIKE': 0, 'Other': 0}

    for a in analyses:
        if a.get('status') == 'no_data':
            continue
        tags = a.get('first_iteration', {}).get('matching', {}).get('tag_breakdown', {})
        for tag, count in tags.items():
            if tag in tag_totals:
                tag_totals[tag] += count
            else:
                tag_totals['Other'] += count

    # Remove empty categories
    tag_totals = {k: v for k, v in tag_totals.items() if v > 0}

    if not tag_totals:
        return '<p>No data available</p>'

    chart_id = 'tag-breakdown-chart'

    colors = {
        'NON-M': '#28a745',
        'NOISY': '#ffc107',
        'PARTIAL': '#17a2b8',
        'SPIKE': '#6f42c1',
        'Other': '#6c757d'
    }

    total_matches = sum(tag_totals.values())

    data = {
        'values': list(tag_totals.values()),
        'labels': list(tag_totals.keys()),
        'type': 'pie',
        'marker': {'colors': [colors.get(k, '#6c757d') for k in tag_totals.keys()]},
        'textinfo': 'label+percent',
        'hole': 0.3
    }

    layout = {
        'title': f'Match Type Distribution<br><sub>Total: {total_matches:,} matches</sub>',
        'showlegend': True
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_iteration_contribution_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    DEPRECATED - Removed by user request.

    Was showing iteration contribution but deemed not useful.

    Returns:
        Empty string
    """
    return ''


def create_issues_summary_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create bar chart showing how many houses have each type of issue.

    Instead of a heatmap per house (which implies order), this shows
    aggregated counts: "X houses have low matching", "Y houses have negative values", etc.

    Args:
        analyses: List of experiment analysis results

    Returns:
        HTML string with Plotly chart
    """
    valid = [a for a in analyses if a.get('status') != 'no_data']

    if not valid:
        return '<p>No data available</p>'

    # Count issues across all houses
    issue_counts = {}

    for a in valid:
        flags = a.get('flags', {})
        for flag, has_issue in flags.items():
            if has_issue:
                formatted = flag.replace('_', ' ').title()
                issue_counts[formatted] = issue_counts.get(formatted, 0) + 1

    if not issue_counts:
        return '<p style="color: #28a745; font-weight: bold;">No issues detected across all houses!</p>'

    # Sort by count descending
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)

    chart_id = 'issues-summary-chart'

    data = {
        'x': [x[0] for x in sorted_issues],
        'y': [x[1] for x in sorted_issues],
        'type': 'bar',
        'marker': {'color': '#dc3545'}
    }

    total_houses = len(valid)

    layout = {
        'title': f'Issues Summary<br><sub>Out of {total_houses} houses analyzed</sub>',
        'xaxis': {'tickangle': -45},
        'yaxis': {'title': 'Number of Houses'},
        'margin': {'b': 120}
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_experiment_summary_table(analyses: List[Dict[str, Any]]) -> str:
    """
    Create an HTML summary table with key experiment statistics.

    This is often more useful than charts for understanding overall results.

    Args:
        analyses: List of experiment analysis results

    Returns:
        HTML string with summary table
    """
    valid = [a for a in analyses if a.get('status') != 'no_data']
    total = len(analyses)
    no_data = total - len(valid)

    if not valid:
        return f'<p>No valid data. {no_data} houses had no data.</p>'

    # Calculate aggregate statistics
    matching_rates = [a.get('iterations', {}).get('first_iter_matching_rate', 0) * 100 for a in valid]
    seg_ratios = [a.get('first_iteration', {}).get('segmentation', {}).get('segmentation_ratio', 0) * 100 for a in valid]

    # Calculate minutes segmentation ratio per house (NOT x3 - real time only)
    # minutes_ratio = matched_minutes / (total_days * 24 * 60)
    minutes_ratios = []
    for a in valid:
        matching = a.get('first_iteration', {}).get('matching', {})
        patterns = a.get('first_iteration', {}).get('patterns', {})
        matched_minutes = matching.get('total_matched_minutes', 0)
        total_days = patterns.get('daily_stats', {}).get('total_days', 0)
        if total_days > 0 and matched_minutes > 0:
            total_available_minutes = total_days * 24 * 60  # Real time, NOT x3
            minutes_ratio = (matched_minutes / total_available_minutes) * 100
            minutes_ratios.append(minutes_ratio)
        else:
            minutes_ratios.append(0)

    # Count houses with specific issues (for detailed breakdown)
    issue_counts = {}
    issue_labels = {
        'low_matching_rate': 'Low Matching Rate (<50%)',
        'has_negative_values': 'Negative Power Values',
        'low_segmentation': 'Low Segmentation (<30%)',
        'has_damaged_phases': 'Damaged Phase(s)',
        'few_matches': 'Few Matches (<10)',
        'high_partial_ratio': 'High Partial Ratio (>30%)',
        'many_remainders': 'Many Remainders (>20)',
        'low_improvement': 'Low Improvement (<5%)',
    }
    for a in valid:
        flags = a.get('flags', {})
        for flag, has_issue in flags.items():
            if has_issue and flag != 'has_recurring_patterns':  # Skip positive flag
                label = issue_labels.get(flag, flag.replace('_', ' ').title())
                issue_counts[label] = issue_counts.get(label, 0) + 1

    total_issues_houses = sum(1 for a in valid if any(
        v for k, v in a.get('flags', {}).items() if k != 'has_recurring_patterns'
    ))

    # Count recurring patterns
    houses_with_patterns = sum(1 for a in valid
        if a.get('first_iteration', {}).get('patterns', {}).get('recurring_matches', {}).get('pattern_count', 0) > 0)

    # Count device detections
    central_ac = sum(1 for a in valid
                     if a.get('first_iteration', {}).get('patterns', {}).get('ac_detection', {}).get('has_central_ac', False))
    regular_ac = sum(1 for a in valid
                     if a.get('first_iteration', {}).get('patterns', {}).get('ac_detection', {}).get('has_regular_ac', False))
    boiler = sum(1 for a in valid
                 if a.get('first_iteration', {}).get('patterns', {}).get('boiler_detection', {}).get('has_boiler', False))

    # Count duration categories (matching segmentation definitions)
    # short: <= 2 min, medium: 3-24 min, long: >= 25 min
    duration_totals = {'short': 0, 'medium': 0, 'long': 0}
    total_matches = 0
    for a in valid:
        matching = a.get('first_iteration', {}).get('matching', {})
        dur_breakdown = matching.get('duration_breakdown', {})
        duration_totals['short'] += dur_breakdown.get('short', 0)
        duration_totals['medium'] += dur_breakdown.get('medium', 0)
        duration_totals['long'] += dur_breakdown.get('long', 0)
        total_matches += matching.get('total_matches', 0)

    # Build issues detail HTML
    issues_detail_html = ""
    if issue_counts:
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        issues_items = [f"{label}: {count} ({100*count/len(valid):.1f}%)" for label, count in sorted_issues]
        issues_detail_html = "<br><small style='color:#856404;'>" + " | ".join(issues_items) + "</small>"

    table_html = f'''
    <table style="width:100%; border-collapse: collapse; margin: 20px 0;">
        <tr style="background: #667eea; color: white;">
            <th colspan="2" style="padding: 15px; text-align: left; font-size: 1.2em;">
                Experiment Summary
            </th>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; font-weight: 600; width: 40%;">Houses Analyzed</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">{len(valid)} (of {total}, {no_data} had no data)</td>
        </tr>
        <tr style="background: #f8f9fa;">
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; font-weight: 600;">Matching Rate (events)</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                <b>Avg: {sum(matching_rates)/len(matching_rates):.1f}%</b> |
                Range: {min(matching_rates):.1f}% - {max(matching_rates):.1f}%
            </td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; font-weight: 600;">Segmentation Ratio (power)</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                <b>Avg: {sum(seg_ratios)/len(seg_ratios):.1f}%</b> |
                Range: {min(seg_ratios):.1f}% - {max(seg_ratios):.1f}%
            </td>
        </tr>
        <tr style="background: #f8f9fa;">
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; font-weight: 600;">Segmentation Ratio (minutes)</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">
                <b>Avg: {sum(minutes_ratios)/len(minutes_ratios) if minutes_ratios else 0:.2f}%</b> |
                Range: {min(minutes_ratios) if minutes_ratios else 0:.2f}% - {max(minutes_ratios) if minutes_ratios else 0:.2f}%
                <br><small style="color:#666;">(matched_minutes from all phases / real time available)</small>
            </td>
        </tr>

        <tr style="background: #e8f4f8;">
            <td colspan="2" style="padding: 10px; font-weight: 600; color: #2c3e50;">Match Duration Breakdown</td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; padding-left: 25px;">Short (≤2 min)</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">{duration_totals['short']:,} ({100*duration_totals['short']/total_matches if total_matches > 0 else 0:.1f}%)</td>
        </tr>
        <tr style="background: #f8f9fa;">
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; padding-left: 25px;">Medium (3-24 min)</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">{duration_totals['medium']:,} ({100*duration_totals['medium']/total_matches if total_matches > 0 else 0:.1f}%)</td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; padding-left: 25px;">Long (≥25 min)</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">{duration_totals['long']:,} ({100*duration_totals['long']/total_matches if total_matches > 0 else 0:.1f}%)</td>
        </tr>

        <tr style="background: #e8f4f8;">
            <td colspan="2" style="padding: 10px; font-weight: 600; color: #2c3e50;">Pattern & Device Detection</td>
        </tr>
        <tr style="background: #f8f9fa;">
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; padding-left: 25px;">Recurring Patterns</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">{houses_with_patterns} houses ({100*houses_with_patterns/len(valid):.1f}%)</td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; padding-left: 25px;">Central AC (multi-phase)</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">{central_ac} houses ({100*central_ac/len(valid):.1f}%)</td>
        </tr>
        <tr style="background: #f8f9fa;">
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; padding-left: 25px;">Regular AC (single-phase)</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">{regular_ac} houses ({100*regular_ac/len(valid):.1f}%)</td>
        </tr>
        <tr>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6; padding-left: 25px;">Water Heater (Boiler)</td>
            <td style="padding: 12px; border-bottom: 1px solid #dee2e6;">{boiler} houses ({100*boiler/len(valid):.1f}%)</td>
        </tr>

        <tr style="background: #fff3cd;">
            <td style="padding: 12px; font-weight: 600;">Houses with Issues</td>
            <td style="padding: 12px;">{total_issues_houses} ({100*total_issues_houses/len(valid):.1f}%){issues_detail_html}</td>
        </tr>
    </table>
    '''

    return table_html


def create_duration_distribution_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create chart showing matches by duration category.

    Categories (as defined in segmentation):
    - Short: <= 2 minutes (spikes, brief switching)
    - Medium: 3-24 minutes (cooking, short appliance use)
    - Long: >= 25 minutes (water heater, AC, continuous loads)

    Args:
        analyses: List of experiment analysis results

    Returns:
        HTML string with Plotly chart
    """
    valid = [a for a in analyses if a.get('status') != 'no_data']

    if not valid:
        return '<p>No data available</p>'

    # Get duration counts from matching data
    duration_categories = {
        'short': 0,   # <= 2 min
        'medium': 0,  # 3-24 min
        'long': 0     # >= 25 min
    }

    for a in valid:
        matching = a.get('first_iteration', {}).get('matching', {})
        dur_breakdown = matching.get('duration_breakdown', {})

        duration_categories['short'] += dur_breakdown.get('short', 0)
        duration_categories['medium'] += dur_breakdown.get('medium', 0)
        duration_categories['long'] += dur_breakdown.get('long', 0)

    total = sum(duration_categories.values())
    if total == 0:
        return '<p>No duration data available</p>'

    chart_id = 'duration-dist-chart'

    # Labels with actual definitions
    labels = [
        'Short<br><sub>≤ 2 min</sub>',
        'Medium<br><sub>3-24 min</sub>',
        'Long<br><sub>≥ 25 min</sub>'
    ]

    values = [duration_categories['short'], duration_categories['medium'], duration_categories['long']]
    colors = ['#ffc107', '#17a2b8', '#6f42c1']

    # Calculate percentages
    percentages = [f'{100*v/total:.1f}%' for v in values]

    data = {
        'x': labels,
        'y': values,
        'type': 'bar',
        'marker': {'color': colors},
        'text': [f'{v:,}<br>({p})' for v, p in zip(values, percentages)],
        'textposition': 'outside'
    }

    layout = {
        'title': f'Matches by Duration<br><sub>Total: {total:,} matches</sub>',
        'xaxis': {'title': ''},
        'yaxis': {'title': 'Number of Matches'},
        'showlegend': False,
        'margin': {'b': 80}
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_pattern_detection_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create chart showing houses with/without recurring patterns.

    Args:
        analyses: List of experiment analysis results

    Returns:
        HTML string with Plotly chart
    """
    valid = [a for a in analyses if a.get('status') != 'no_data']

    if not valid:
        return '<p>No data available</p>'

    # Count houses with patterns
    houses_with_patterns = 0
    houses_without_patterns = 0
    total_patterns = 0

    for a in valid:
        patterns = a.get('first_iteration', {}).get('patterns', {})
        recurring = patterns.get('recurring_matches', {})
        pattern_count = recurring.get('pattern_count', 0)

        if pattern_count > 0:
            houses_with_patterns += 1
            total_patterns += pattern_count
        else:
            houses_without_patterns += 1

    chart_id = 'pattern-detection-chart'

    data = {
        'values': [houses_with_patterns, houses_without_patterns],
        'labels': [
            f'With Patterns<br>({houses_with_patterns} houses)',
            f'No Patterns<br>({houses_without_patterns} houses)'
        ],
        'type': 'pie',
        'marker': {'colors': ['#28a745', '#dc3545']},
        'textinfo': 'percent',
        'hole': 0.4
    }

    layout = {
        'title': f'Recurring Pattern Detection<br><sub>{total_patterns} patterns found in {houses_with_patterns} houses</sub>',
        'showlegend': True,
        'legend': {'orientation': 'h', 'y': -0.1}
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:400px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_device_detection_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create chart showing detected device types across houses.

    Device types:
    - Water Heater (Boiler): Long (≥25min) isolated high-power (≥1500W) events
      with no medium-duration events before/after (isolated long load)
    - Central AC: Multi-phase synchronized events
    - Regular AC: Single-phase high-power events (>800W)

    Args:
        analyses: List of experiment analysis results

    Returns:
        HTML string with Plotly chart
    """
    valid = [a for a in analyses if a.get('status') != 'no_data']

    if not valid:
        return '<p>No data available</p>'

    # Count device detections
    device_counts = {
        'boiler': {'houses': 0, 'activations': 0},
        'central_ac': {'houses': 0, 'activations': 0},
        'regular_ac': {'houses': 0, 'activations': 0},
    }

    for a in valid:
        patterns = a.get('first_iteration', {}).get('patterns', {})
        ac_detection = patterns.get('ac_detection', {})
        boiler_detection = patterns.get('boiler_detection', {})

        # Central AC
        if ac_detection.get('has_central_ac', False):
            device_counts['central_ac']['houses'] += 1
            device_counts['central_ac']['activations'] += ac_detection.get('central_ac', {}).get('total_count', 0)

        # Regular AC
        if ac_detection.get('has_regular_ac', False):
            device_counts['regular_ac']['houses'] += 1
            device_counts['regular_ac']['activations'] += ac_detection.get('regular_ac', {}).get('total_count', 0)

        # Boiler: Long isolated high-power events (no medium events nearby)
        if boiler_detection.get('has_boiler', False):
            device_counts['boiler']['houses'] += 1
            device_counts['boiler']['activations'] += boiler_detection.get('boiler', {}).get('total_count', 0)

    chart_id = 'device-detection-chart'

    labels = ['Water Heater<br>(Boiler)', 'Central AC<br>(Multi-phase)', 'Regular AC<br>(Single-phase)']
    house_counts = [
        device_counts['boiler']['houses'],
        device_counts['central_ac']['houses'],
        device_counts['regular_ac']['houses']
    ]
    activation_counts = [
        device_counts['boiler']['activations'],
        device_counts['central_ac']['activations'],
        device_counts['regular_ac']['activations']
    ]

    colors = ['#fd7e14', '#667eea', '#17a2b8']

    data = {
        'x': labels,
        'y': house_counts,
        'type': 'bar',
        'marker': {'color': colors},
        'text': [f'{h} houses<br>{a} activations' for h, a in zip(house_counts, activation_counts)],
        'textposition': 'outside'
    }

    total_houses = len(valid)

    layout = {
        'title': f'Device Detection Summary<br><sub>Out of {total_houses} houses analyzed</sub>',
        'xaxis': {'title': ''},
        'yaxis': {'title': 'Number of Houses'},
        'showlegend': False,
        'margin': {'b': 80}
    }

    return f'''
    <div id="{chart_id}" style="width:100%;height:450px;"></div>
    <script>
        Plotly.newPlot('{chart_id}', [{json.dumps(data)}], {json.dumps(layout)});
    </script>
    '''


def create_phase_distribution_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    DEPRECATED - Phase distribution is meaningless across houses.
    Phase 1 in one house might be Phase 3 in another.

    Returns empty string to remove this chart.
    """
    return ''


# Keep old function names for backwards compatibility but redirect to new versions
def create_matching_comparison_chart(analyses: List[Dict[str, Any]]) -> str:
    """Deprecated - use create_matching_rate_distribution_chart instead."""
    return create_matching_rate_distribution_chart(analyses)


def create_segmentation_chart(analyses: List[Dict[str, Any]]) -> str:
    """Deprecated - use create_segmentation_ratio_distribution_chart instead."""
    return create_segmentation_ratio_distribution_chart(analyses)


def create_iteration_progress_chart(analyses: List[Dict[str, Any]]) -> str:
    """Deprecated - use create_iteration_contribution_chart instead."""
    return create_iteration_contribution_chart(analyses)


def create_issues_heatmap(analyses: List[Dict[str, Any]]) -> str:
    """Deprecated - use create_issues_summary_chart instead."""
    return create_issues_summary_chart(analyses)


def create_events_reduction_chart(analyses: List[Dict[str, Any]]) -> str:
    """Deprecated - replaced by create_experiment_summary_table."""
    return create_experiment_summary_table(analyses)
