"""
Chart generation for house analysis reports.

Uses Plotly to create interactive charts that can be embedded in HTML.
"""
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from metrics.wave_behavior import WAVE_PRESENT_THRESHOLD, WAVE_DOMINANT_THRESHOLD

logger = logging.getLogger(__name__)


def create_days_distribution_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create histogram showing how many houses are in each days range.

    Args:
        analyses: List of house analysis results

    Returns:
        HTML string with embedded chart
    """
    logger.debug("create_days_distribution_chart: %d analyses", len(analyses))
    # Define day ranges (bins) - 6-month intervals from 0 to 4+ years
    # 6 months ≈ 182 days
    bins = [0, 182, 365, 548, 730, 913, 1095, 1278, 1460, float('inf')]
    bin_labels = ['0-6m', '6m-1y', '1-1.5y', '1.5-2y', '2-2.5y', '2.5-3y', '3-3.5y', '3.5-4y', '4y+']
    bin_colors = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#d4ac0d', '#27ae60', '#1abc9c', '#3498db', '#2980b9']

    # Count houses in each bin
    bin_counts = [0] * len(bin_labels)
    houses_per_bin = [[] for _ in bin_labels]

    for a in analyses:
        days = a.get('coverage', {}).get('days_span', 0)
        house_id = a.get('house_id', 'unknown')

        for i in range(len(bins) - 1):
            if bins[i] <= days < bins[i + 1]:
                bin_counts[i] += 1
                houses_per_bin[i].append(str(house_id))
                break

    # Create hover text with house IDs
    hover_texts = []
    for i, (label, count, houses) in enumerate(zip(bin_labels, bin_counts, houses_per_bin)):
        if count > 0:
            house_list = ', '.join(houses[:10])
            if len(houses) > 10:
                house_list += f'... (+{len(houses) - 10} more)'
            hover_texts.append(f'{label} days<br>{count} houses<br>{house_list}')
        else:
            hover_texts.append(f'{label} days<br>0 houses')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bin_labels,
        y=bin_counts,
        marker_color=bin_colors,
        text=bin_counts,
        textposition='outside',
        hovertext=hover_texts,
        hoverinfo='text',
        name='Houses'
    ))

    fig.update_layout(
        title='Houses by Data Duration (Days)',
        xaxis_title='Days Range',
        yaxis_title='Number of Houses',
        showlegend=False,
        height=400
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_quality_distribution_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create histogram of quality scores across all houses.

    Args:
        analyses: List of house analysis results

    Returns:
        HTML string with embedded chart
    """
    logger.debug("create_quality_distribution_chart: %d analyses", len(analyses))
    scores = []
    house_ids = []

    for a in analyses:
        score = a.get('data_quality', {}).get('quality_score', 0)
        house_id = a.get('house_id', 'unknown')
        scores.append(score)
        house_ids.append(house_id)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=10,
        marker_color='steelblue',
        opacity=0.7,
        name='Houses'
    ))

    # Add vertical lines for tier boundaries
    fig.add_vline(x=90, line_dash="dash", line_color="green",
                  annotation_text="Excellent (90+)")
    fig.add_vline(x=75, line_dash="dash", line_color="orange",
                  annotation_text="Good (75+)")
    fig.add_vline(x=50, line_dash="dash", line_color="red",
                  annotation_text="Fair (50+)")

    fig.update_layout(
        title="Quality Score Distribution",
        xaxis_title="Quality Score",
        yaxis_title="Number of Houses",
        showlegend=False,
        height=400
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_coverage_comparison_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create bar chart comparing coverage ratio across houses.

    Args:
        analyses: List of house analysis results

    Returns:
        HTML string with embedded chart
    """
    data = []
    for a in analyses:
        coverage = a.get('coverage', {})
        data.append({
            'house_id': a.get('house_id', 'unknown'),
            'coverage_ratio': coverage.get('coverage_ratio', 0) * 100,
            'days_span': coverage.get('days_span', 0)
        })

    df = pd.DataFrame(data).sort_values('coverage_ratio', ascending=True)

    fig = go.Figure()

    # Color based on coverage level
    colors = ['red' if x < 80 else 'orange' if x < 95 else 'green'
              for x in df['coverage_ratio']]

    fig.add_trace(go.Bar(
        y=df['house_id'].astype(str),
        x=df['coverage_ratio'],
        orientation='h',
        marker_color=colors,
        text=df['coverage_ratio'].round(1).astype(str) + '%',
        textposition='outside',
        hovertemplate='House %{y}<br>Coverage: %{x:.1f}%<br>Days: %{customdata}<extra></extra>',
        customdata=df['days_span']
    ))

    fig.add_vline(x=80, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_vline(x=95, line_dash="dash", line_color="green", opacity=0.5)

    fig.update_layout(
        title="Data Coverage by House",
        xaxis_title="Coverage (%)",
        yaxis_title="House ID",
        height=max(400, len(df) * 25),
        xaxis_range=[0, 105]
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_phase_balance_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create stacked bar chart showing phase distribution per house.

    Args:
        analyses: List of house analysis results

    Returns:
        HTML string with embedded chart
    """
    import plotly.graph_objects as go

    try:
        # Collect data per house
        houses = []
        w1_values = []
        w2_values = []
        w3_values = []

        for a in analyses:
            house_id = str(a.get('house_id', 'unknown'))
            power = a.get('power_statistics', {})

            w1 = power.get('phase_w1_mean', 0) or 0
            w2 = power.get('phase_w2_mean', 0) or 0
            w3 = power.get('phase_w3_mean', 0) or 0

            if w1 > 0 or w2 > 0 or w3 > 0:
                houses.append(house_id)
                w1_values.append(w1)
                w2_values.append(w2)
                w3_values.append(w3)

        if not houses:
            return "<p>No phase data available</p>"

        # Create stacked bar chart using graph_objects (more reliable than express)
        fig = go.Figure(data=[
            go.Bar(name='w1', x=houses, y=w1_values, marker_color='#1f77b4'),
            go.Bar(name='w2', x=houses, y=w2_values, marker_color='#ff7f0e'),
            go.Bar(name='w3', x=houses, y=w3_values, marker_color='#2ca02c')
        ])

        fig.update_layout(
            barmode='stack',
            title='Phase Balance by House',
            xaxis_title='House ID',
            yaxis_title='Mean Power (W)',
            height=400,
            xaxis_tickangle=-45
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    except Exception as e:
        return f"<p>Error creating phase balance chart: {e}</p>"


def create_day_night_scatter(analyses: List[Dict[str, Any]]) -> str:
    """
    Create scatter plot of day vs night power consumption.

    Args:
        analyses: List of house analysis results

    Returns:
        HTML string with embedded chart
    """
    data = []

    for a in analyses:
        house_id = a.get('house_id', 'unknown')
        temporal = a.get('temporal_patterns', {})

        day_mean = temporal.get('total_day_mean', 0)
        night_mean = temporal.get('total_night_mean', 0)

        if day_mean > 0 or night_mean > 0:
            data.append({
                'house_id': house_id,
                'day_mean': day_mean,
                'night_mean': night_mean,
                'ratio': night_mean / day_mean if day_mean > 0 else 0
            })

    if not data:
        return "<p>No temporal data available</p>"

    df = pd.DataFrame(data)

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=df['day_mean'],
        y=df['night_mean'],
        mode='markers+text',
        marker=dict(
            size=12,
            color=df['ratio'],
            colorscale='RdYlGn_r',
            colorbar=dict(title='Night/Day Ratio'),
            showscale=True
        ),
        text=df['house_id'],
        textposition='top center',
        hovertemplate='House %{text}<br>Day: %{x:.0f}W<br>Night: %{y:.0f}W<br>Ratio: %{marker.color:.2f}<extra></extra>'
    ))

    # Add diagonal line (equal day/night)
    max_val = max(df['day_mean'].max(), df['night_mean'].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Equal Day/Night',
        showlegend=True
    ))

    fig.update_layout(
        title='Day vs Night Power Consumption',
        xaxis_title='Day Mean Power (W)',
        yaxis_title='Night Mean Power (W)',
        height=500
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_issues_heatmap(analyses: List[Dict[str, Any]]) -> str:
    """
    Create heatmap showing issues/flags per house.

    Args:
        analyses: List of house analysis results

    Returns:
        HTML string with embedded chart
    """
    logger.debug("create_issues_heatmap: %d analyses", len(analyses))
    # Collect all flags
    house_ids = []
    all_flags = set()

    for a in analyses:
        house_ids.append(a.get('house_id', 'unknown'))
        flags = a.get('flags', {})
        all_flags.update(flags.keys())

    # Filter out non-boolean flags and flags that apply to all houses (not useful for comparison)
    excluded_flags = {'dead_phases_list', 'many_outliers', 'many_flat_segments'}
    all_flags = [f for f in all_flags if f not in excluded_flags]

    # Count how many houses have each flag
    flag_counts = {f: 0 for f in all_flags}
    for a in analyses:
        flags = a.get('flags', {})
        for f in all_flags:
            if flags.get(f, False):
                flag_counts[f] += 1

    # Sort flags by count (ascending order)
    all_flags = sorted(all_flags, key=lambda f: flag_counts[f])

    # Build matrix with sorted flags
    matrix = []
    for a in analyses:
        flags = a.get('flags', {})
        row = [1 if flags.get(f, False) else 0 for f in all_flags]
        matrix.append(row)

    # Clean up flag names for display
    flag_display_names = {
        'low_coverage': 'Coverage < 80%',
        'short_duration': 'Less Than 30 Days',
        'has_large_gaps': 'Gaps Over 1 Hour',
        'many_gaps': '>5% Gaps Over 2min',
        'has_negative_values': 'Negative Values',
        'many_large_jumps': '>500 Jumps Over 2kW',
        'low_quality_score': 'Quality < 70',
        'unbalanced_phases': 'Phase Ratio > 3',
        'single_active_phase': 'Single Active Phase',
        'very_high_power': 'Max Power > 20kW',
        'unusual_night_ratio': 'Night/Day > 3',
        'has_dead_phase': 'Dead Phase (<2% of sisters)',
        'low_sharp_entry': 'Low Sharp Entry Rate',
        'low_device_signature': 'Low Device Signature',
        'low_power_profile': 'Low Power Profile',
        'low_variability': 'Low Variability',
        'low_data_volume': 'Low Data Volume',
        'low_data_integrity': 'Low Data Integrity',
    }
    display_flags = [flag_display_names.get(f, f.replace('_', ' ').title()) for f in all_flags]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=display_flags,
        y=house_ids,
        colorscale=[[0, 'white'], [1, 'red']],
        showscale=False,
        hovertemplate='House %{y}<br>%{x}: %{z}<extra></extra>'
    ))

    fig.update_layout(
        title='Issues by House',
        xaxis_title='Issue Type',
        yaxis_title='House ID',
        height=max(400, len(house_ids) * 25),
        xaxis_tickangle=-45
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_power_distribution_chart(analyses: List[Dict[str, Any]],
                                    house_id: Optional[str] = None) -> str:
    """
    Create power range distribution chart.

    Args:
        analyses: List of house analysis results
        house_id: If provided, show only this house; otherwise show average

    Returns:
        HTML string with embedded chart
    """
    ranges = ['0-100W', '100-500W', '500-1000W', '1000-2000W', '2000W+']
    range_keys = ['share_0_100', 'share_100_500', 'share_500_1000',
                  'share_1000_2000', 'share_2000_plus']

    if house_id:
        # Single house
        analysis = next((a for a in analyses if a.get('house_id') == house_id), None)
        if not analysis:
            return f"<p>House {house_id} not found</p>"

        power = analysis.get('power_statistics', {})
        values = []
        for key in range_keys:
            # Try different phase naming conventions
            for phase in ['phase_w1_', 'phase_1_']:
                full_key = phase + key
                if full_key in power:
                    values.append(power[full_key] * 100)
                    break
            else:
                values.append(0)

        fig = go.Figure(data=[go.Pie(
            labels=ranges,
            values=values,
            hole=0.4,
            marker_colors=['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
        )])

        fig.update_layout(
            title=f'Power Distribution - House {house_id}',
            height=400
        )
    else:
        # Average across all houses
        avg_values = [0] * len(range_keys)
        count = 0

        for a in analyses:
            power = a.get('power_statistics', {})
            for i, key in enumerate(range_keys):
                for phase in ['phase_w1_', 'phase_1_']:
                    full_key = phase + key
                    if full_key in power:
                        avg_values[i] += power[full_key] * 100
                        if i == 0:
                            count += 1
                        break

        if count > 0:
            avg_values = [v / count for v in avg_values]

        fig = go.Figure(data=[go.Pie(
            labels=ranges,
            values=avg_values,
            hole=0.4,
            marker_colors=['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
        )])

        fig.update_layout(
            title=f'Average Power Distribution (n={count})',
            height=400
        )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_tier_score_comparison_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create grouped bar chart showing average score components per quality tier.

    Args:
        analyses: List of house analysis results

    Returns:
        HTML string with embedded chart
    """
    # Score component definitions
    components = [
        ('sharp_entry_score', 'Sharp Entry (20)', '#e74c3c'),
        ('device_signature_score', 'Device Sig. (15)', '#e67e22'),
        ('power_profile_score', 'Power Profile (20)', '#f39c12'),
        ('variability_score', 'Variability (20)', '#9b59b6'),
        ('data_volume_score', 'Data Volume (15)', '#3498db'),
        ('integrity_score', 'Integrity (10)', '#2ecc71'),
    ]

    # Group houses into tiers
    tier_order = ['Excellent', 'Good', 'Fair', 'Poor', 'Faulty']
    tier_houses = {t: [] for t in tier_order}

    for a in analyses:
        score = a.get('data_quality', {}).get('quality_score', 0)
        qlabel = a.get('flags', {}).get('quality_label')

        if qlabel and 'faulty' in qlabel:
            tier_houses['Faulty'].append(a)
        elif score >= 90:
            tier_houses['Excellent'].append(a)
        elif score >= 75:
            tier_houses['Good'].append(a)
        elif score >= 50:
            tier_houses['Fair'].append(a)
        else:
            tier_houses['Poor'].append(a)

    # Only include tiers with houses
    active_tiers = [t for t in tier_order if tier_houses[t]]

    if not active_tiers:
        return "<p>No data available for tier comparison</p>"

    fig = go.Figure()

    for comp_key, comp_label, color in components:
        avgs = []
        for tier in active_tiers:
            houses = tier_houses[tier]
            avg = sum(h.get('data_quality', {}).get(comp_key, 0) for h in houses) / len(houses)
            avgs.append(round(avg, 1))

        fig.add_trace(go.Bar(
            name=comp_label,
            x=active_tiers,
            y=avgs,
            marker_color=color,
            text=[f'{v:.1f}' for v in avgs],
            textposition='outside',
            textfont=dict(size=9),
        ))

    fig.update_layout(
        title='Average Score Components by Quality Tier',
        xaxis_title='Quality Tier',
        yaxis_title='Average Points',
        barmode='group',
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5,
                    font=dict(size=10)),
        margin=dict(b=100),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_hourly_pattern_chart(analysis: Dict[str, Any]) -> str:
    """
    Create hourly power pattern chart with confidence band.

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    logger.debug("create_hourly_pattern_chart: house=%s", analysis.get('house_id', 'unknown'))
    temporal = analysis.get('temporal_patterns', {})
    pattern = temporal.get('total_hourly_pattern', {})

    if not pattern:
        return "<p>No hourly pattern data available</p>"

    hours = pattern.get('hours', list(range(24)))
    mean_values = pattern.get('mean', [0] * 24)
    std_values = pattern.get('std', [0] * 24)

    # Calculate confidence band (mean ± std)
    upper = [m + s for m, s in zip(mean_values, std_values)]
    lower = [max(0, m - s) for m, s in zip(mean_values, std_values)]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=hours + hours[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 Std Dev',
        showlegend=True
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=hours,
        y=mean_values,
        mode='lines+markers',
        line=dict(color='#667eea', width=2),
        marker=dict(size=6),
        name='Average Power',
        hovertemplate='Hour %{x}:00<br>Power: %{y:.0f}W<extra></extra>'
    ))

    fig.update_layout(
        title='Hourly Power Pattern',
        xaxis_title='Hour of Day',
        yaxis_title='Power (W)',
        xaxis=dict(tickmode='array', tickvals=list(range(0, 24, 2)),
                   ticktext=[f'{h}:00' for h in range(0, 24, 2)]),
        height=400,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_phase_power_chart(analysis: Dict[str, Any]) -> str:
    """
    Create bar chart showing power by phase.

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    power = analysis.get('power_statistics', {})

    phases = []
    values = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for phase_name in ['w1', 'w2', 'w3', '1', '2', '3']:
        key = f'phase_{phase_name}_mean'
        if key in power:
            display_name = f'Phase {phase_name.replace("w", "")}'
            phases.append(display_name)
            values.append(power[key])

    if not phases:
        return "<p>No phase data available</p>"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=phases,
        y=values,
        marker_color=colors[:len(phases)],
        text=[f'{v:.0f}W' for v in values],
        textposition='outside',
        hovertemplate='%{x}<br>Average: %{y:.0f}W<extra></extra>'
    ))

    fig.update_layout(
        title='Average Power by Phase',
        xaxis_title='Phase',
        yaxis_title='Average Power (W)',
        height=350,
        showlegend=False
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_monthly_pattern_chart(analysis: Dict[str, Any]) -> str:
    """
    Create monthly/seasonal power pattern chart.

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    temporal = analysis.get('temporal_patterns', {})
    pattern = temporal.get('total_monthly_pattern', {})

    if not pattern:
        return "<p>No monthly pattern data available</p>"

    months = pattern.get('months', [])
    values = pattern.get('mean', [])

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    labels = [month_names[m-1] if 1 <= m <= 12 else str(m) for m in months]

    # Color based on season
    colors = []
    for m in months:
        if m in [12, 1, 2]:  # Winter
            colors.append('#3498db')
        elif m in [3, 4, 5]:  # Spring
            colors.append('#2ecc71')
        elif m in [6, 7, 8]:  # Summer
            colors.append('#e74c3c')
        else:  # Fall
            colors.append('#f39c12')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f'{v:.0f}W' for v in values],
        textposition='outside',
        hovertemplate='%{x}<br>Average: %{y:.0f}W<extra></extra>'
    ))

    fig.update_layout(
        title='Monthly Power Pattern',
        xaxis_title='Month',
        yaxis_title='Average Power (W)',
        height=350,
        showlegend=False
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_weekly_pattern_chart(analysis: Dict[str, Any]) -> str:
    """
    Create weekly power pattern chart.

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    temporal = analysis.get('temporal_patterns', {})
    pattern = temporal.get('total_weekly_pattern', {})

    if not pattern:
        return "<p>No weekly pattern data available</p>"

    days = pattern.get('days', list(range(7)))
    values = pattern.get('mean', [0] * 7)

    # Israeli week: Sunday=0 to Saturday=6
    day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    labels = [day_names[d] for d in days]

    # Highlight weekend (Friday-Saturday in Israel)
    colors = ['#667eea' if d not in [5, 6] else '#e74c3c' for d in days]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f'{v:.0f}W' for v in values],
        textposition='outside',
        hovertemplate='%{x}<br>Average: %{y:.0f}W<extra></extra>'
    ))

    fig.update_layout(
        title='Weekly Power Pattern',
        xaxis_title='Day of Week',
        yaxis_title='Average Power (W)',
        height=350,
        showlegend=False
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_power_heatmap_chart(analysis: Dict[str, Any]) -> str:
    """
    Create heatmap of power consumption (hour vs day of week).

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    temporal = analysis.get('temporal_patterns', {})
    heatmap = temporal.get('power_heatmap', {})

    if not heatmap:
        return "<p>No heatmap data available</p>"

    days = heatmap.get('days', list(range(7)))
    hours = heatmap.get('hours', list(range(24)))
    values = heatmap.get('values', [[0]*24]*7)

    day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    hour_labels = [f'{h}:00' for h in hours]

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=hour_labels,
        y=[day_names[d] for d in days],
        colorscale='YlOrRd',
        colorbar=dict(title='Power (W)'),
        hovertemplate='%{y} %{x}<br>Power: %{z:.0f}W<extra></extra>'
    ))

    fig.update_layout(
        title='Power Consumption Heatmap',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400,
        xaxis=dict(tickmode='array', tickvals=list(range(0, 24, 2)),
                   ticktext=[f'{h}:00' for h in range(0, 24, 2)])
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_power_histogram(analysis: Dict[str, Any]) -> str:
    """
    Create power distribution histogram.

    Note: This requires raw data which may not be in the analysis dict.
    Falls back to showing power range distribution from statistics.

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    power = analysis.get('power_statistics', {})

    # Build histogram from power range statistics
    ranges = []
    values = []
    colors = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']

    range_keys = [
        ('0-100W', 'share_0_100'),
        ('100-500W', 'share_100_500'),
        ('500-1000W', 'share_500_1000'),
        ('1000-2000W', 'share_1000_2000'),
        ('2000W+', 'share_2000_plus')
    ]

    for label, key in range_keys:
        for prefix in ['phase_w1_', 'phase_1_', 'total_']:
            full_key = prefix + key
            if full_key in power:
                ranges.append(label)
                values.append(power[full_key] * 100)
                break

    if not ranges:
        return "<p>No power distribution data available</p>"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=ranges,
        y=values,
        marker_color=colors[:len(ranges)],
        text=[f'{v:.1f}%' for v in values],
        textposition='outside',
        hovertemplate='%{x}<br>Percentage: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title='Power Distribution',
        xaxis_title='Power Range',
        yaxis_title='Percentage of Time (%)',
        height=350,
        showlegend=False
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_score_breakdown_chart(analysis: Dict[str, Any]) -> str:
    """
    Create a bar chart showing quality score breakdown.

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    logger.debug("create_score_breakdown_chart: house=%s", analysis.get('house_id', 'unknown'))
    quality = analysis.get('data_quality', {})

    # Get score components (6-component scoring system)
    sharp_entry = quality.get('sharp_entry_score', 0)
    device_sig = quality.get('device_signature_score', 0)
    power_profile = quality.get('power_profile_score', quality.get('balance_score', 0))
    variability = quality.get('variability_score', quality.get('noise_score', 0))
    data_volume = quality.get('data_volume_score', quality.get('completeness_score', 0))
    integrity = quality.get('integrity_score', quality.get('gap_score', 0))
    final_score = quality.get('quality_score', 0)

    categories = [
        'Sharp Entry<br>Rate<br>(max 20)',
        'Device<br>Signature<br>(max 15)',
        'Power<br>Profile<br>(max 20)',
        'Variability<br>(max 20)',
        'Data<br>Volume<br>(max 15)',
        'Data<br>Integrity<br>(max 10)',
        'Final<br>Score',
    ]
    values = [sharp_entry, device_sig, power_profile, variability, data_volume, integrity, final_score]
    colors = ['#e74c3c', '#e67e22', '#f39c12', '#9b59b6', '#3498db', '#2ecc71', '#1abc9c']

    sharp_rate = quality.get('sharp_entry_rate', 0)
    sustained = quality.get('sustained_high_power_density', 0)
    compressor = quality.get('compressor_cycle_density', 0)

    hover_texts = [
        f'Sharp Entry Rate: {sharp_entry:.1f}/20<br>'
        f'Rate: {sharp_rate:.2%} of threshold crossings are sharp jumps.<br>'
        f'Higher = more detectable ON/OFF events for the algorithm.',
        f'Device Signature: {device_sig:.1f}/15<br>'
        f'Sustained high power (boiler): {sustained:.1f}/10K min<br>'
        f'Compressor cycles (AC): {compressor:.1f}/10K min',
        f'Power Profile: {power_profile:.1f}/20<br>'
        f'Penalizes stuck 500-1000W range (below detection threshold).<br>'
        f'Rewards clear low-power baseline for sharp event contrast.',
        f'Variability: {variability:.1f}/20<br>'
        f'Coefficient of Variation (CV) of total power.<br>'
        f'Higher CV = more device activity = better for algorithm.',
        f'Data Volume: {data_volume:.1f}/15<br>'
        f'Days of data available + monthly coverage balance.<br>'
        f'More data = more patterns to detect.',
        f'Data Integrity: {integrity:.1f}/10<br>'
        f'NaN percentage, gap frequency, negative values.<br>'
        f'Basic data quality checks.',
        f'Final Score: {final_score:.1f}/100<br>'
        f'Sum of all components.',
    ]

    fig = go.Figure()

    # Component bars
    fig.add_trace(go.Bar(
        x=categories[:-1],
        y=values[:-1],
        marker_color=colors[:-1],
        text=[f'{v:.1f}' for v in values[:-1]],
        textposition='outside',
        name='Components',
        hovertext=hover_texts[:-1],
        hoverinfo='text',
    ))

    # Final score bar
    fig.add_trace(go.Bar(
        x=[categories[-1]],
        y=[values[-1]],
        marker_color=colors[-1],
        text=[f'{final_score:.1f}'],
        textposition='outside',
        name='Final Score',
        hovertext=[hover_texts[-1]],
        hoverinfo='text',
    ))

    fig.update_layout(
        title='Quality Score Breakdown',
        yaxis_title='Points',
        height=380,
        showlegend=False,
        yaxis=dict(range=[0, max(values) + 10])
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_quality_flags_chart(analysis: Dict[str, Any]) -> str:
    """
    Create a chart showing quality flag status for each scoring component.
    Green cells = sufficient, red cells = flagged (insufficient).

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    quality = analysis.get('data_quality', {})
    quality_flags = quality.get('quality_flags', [])

    # Define components with their scores, max values, and flag keys
    components = [
        ('Sharp Entry Rate', quality.get('sharp_entry_score', 0), 20, 'low_sharp_entry'),
        ('Device Signature', quality.get('device_signature_score', 0), 15, 'low_device_signature'),
        ('Power Profile', quality.get('power_profile_score', 0), 20, 'low_power_profile'),
        ('Variability', quality.get('variability_score', 0), 20, 'low_variability'),
        ('Data Volume', quality.get('data_volume_score', 0), 15, 'low_data_volume'),
        ('Data Integrity', quality.get('integrity_score', 0), 10, 'low_data_integrity'),
    ]

    names = [c[0] for c in components]
    scores = [c[1] for c in components]
    maxes = [c[2] for c in components]
    flagged = [c[3] in quality_flags for c in components]

    # Percentage of max for each component
    pcts = [s / m * 100 if m > 0 else 0 for s, m in zip(scores, maxes)]

    # Colors: red for flagged, green for sufficient
    bar_colors = ['#e74c3c' if f else '#27ae60' for f in flagged]
    border_colors = ['#c0392b' if f else '#1e8449' for f in flagged]

    # Category labels with score/max
    labels = [f'{n}<br>{s:.0f}/{m}' for n, s, m in zip(names, scores, maxes)]

    hover_texts = []
    for name, score, mx, is_flagged in zip(names, scores, maxes, flagged):
        status = 'Insufficient' if is_flagged else 'Sufficient'
        hover_texts.append(
            f'{name}: {score:.1f}/{mx} ({score/mx*100:.0f}%)<br>'
            f'Status: <b>{status}</b>'
        )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=pcts,
        marker_color=bar_colors,
        marker_line_color=border_colors,
        marker_line_width=2,
        text=[f'{"LOW" if f else "OK"}' for f in flagged],
        textposition='outside',
        textfont=dict(
            color=['#e74c3c' if f else '#27ae60' for f in flagged],
            size=13,
        ),
        hovertext=hover_texts,
        hoverinfo='text',
    ))

    # Add threshold line at 50%
    fig.add_hline(
        y=50, line_dash='dash', line_color='#888',
        annotation_text='Threshold',
        annotation_position='bottom right',
    )

    fig.update_layout(
        title='Quality Component Status',
        yaxis_title='% of Maximum Score',
        height=350,
        showlegend=False,
        yaxis=dict(range=[0, 115]),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_mini_hourly_chart(hourly_pattern: Dict[str, Any], title: str = "") -> str:
    """
    Create a compact hourly pattern chart for monthly view.

    Args:
        hourly_pattern: Dictionary with 'hours' and 'mean' keys
        title: Optional title for the chart

    Returns:
        HTML string with embedded mini chart
    """
    if not hourly_pattern:
        return "<p>No data</p>"

    hours = hourly_pattern.get('hours', list(range(24)))
    mean_values = hourly_pattern.get('mean', [0] * 24)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hours,
        y=mean_values,
        mode='lines',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)',
        hovertemplate='%{x}:00<br>%{y:.0f}W<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=12)) if title else None,
        height=120,
        margin=dict(l=30, r=10, t=25 if title else 5, b=20),
        xaxis=dict(
            tickmode='array',
            tickvals=[0, 6, 12, 18, 23],
            ticktext=['0', '6', '12', '18', '23'],
            showgrid=False,
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_year_hourly_chart(hourly_pattern: Dict[str, Any], year: int) -> str:
    """
    Create hourly pattern chart for a specific year.

    Args:
        hourly_pattern: Dictionary with 'hours', 'mean', and 'std' keys
        year: Year number

    Returns:
        HTML string with embedded chart
    """
    if not hourly_pattern:
        return "<p>No hourly pattern data available</p>"

    hours = hourly_pattern.get('hours', list(range(24)))
    mean_values = hourly_pattern.get('mean', [0] * 24)
    std_values = hourly_pattern.get('std', [0] * 24)

    # Calculate confidence band (mean ± std)
    upper = [m + s for m, s in zip(mean_values, std_values)]
    lower = [max(0, m - s) for m, s in zip(mean_values, std_values)]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=hours + hours[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='±1 Std Dev',
        showlegend=True
    ))

    # Mean line
    fig.add_trace(go.Scatter(
        x=hours,
        y=mean_values,
        mode='lines+markers',
        line=dict(color='#667eea', width=2),
        marker=dict(size=5),
        name='Average Power',
        hovertemplate='Hour %{x}:00<br>Power: %{y:.0f}W<extra></extra>'
    ))

    fig.update_layout(
        title=f'Hourly Power Pattern - {year}',
        xaxis_title='Hour of Day',
        yaxis_title='Power (W)',
        height=300,
        xaxis=dict(tickmode='array', tickvals=list(range(0, 24, 2)),
                   ticktext=[f'{h}:00' for h in range(0, 24, 2)]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_wave_monthly_chart(analysis: Dict[str, Any]) -> str:
    """
    Create grouped bar chart showing wave prevalence % by month per phase.

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    wave = analysis.get('wave_behavior', {})
    phases_data = wave.get('phases', {})

    if not phases_data:
        return "<p>No wave behavior data available</p>"

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    phase_colors = {'w1': '#1f77b4', 'w2': '#ff7f0e', 'w3': '#2ca02c',
                    '1': '#1f77b4', '2': '#ff7f0e', '3': '#2ca02c'}

    fig = go.Figure()

    for col, p_data in phases_data.items():
        monthly = p_data.get('monthly_minutes', {})
        if not monthly:
            continue

        total_minutes = sum(monthly.values())
        if total_minutes == 0:
            continue

        months = list(range(1, 13))
        values = [monthly.get(m, 0) for m in months]

        display_name = f'Phase {col.replace("w", "")}'
        fig.add_trace(go.Bar(
            x=month_names,
            y=values,
            name=display_name,
            marker_color=phase_colors.get(col, '#999'),
            hovertemplate=f'{display_name}<br>%{{x}}: %{{y}} min<extra></extra>'
        ))

    fig.update_layout(
        title='Wave Activity by Month (minutes)',
        xaxis_title='Month',
        yaxis_title='Wave Minutes',
        barmode='group',
        height=300,
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5),
        margin=dict(b=80),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_wave_comparison_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create scatter plot: quality score vs wave score, colored by classification.

    Args:
        analyses: List of house analysis results

    Returns:
        HTML string with embedded chart
    """
    cls_colors = {
        'wave_dominant': '#e67e22',
        'has_waves': '#3498db',
        'no_waves': '#95a5a6',
    }
    cls_labels = {
        'wave_dominant': 'Wave Dominant',
        'has_waves': 'Has Waves',
        'no_waves': 'No Waves',
    }

    fig = go.Figure()

    for cls_key, color in cls_colors.items():
        house_ids = []
        quality_scores = []
        wave_scores = []
        dominant_phases_list = []

        for a in analyses:
            wave = a.get('wave_behavior', {})
            if wave.get('wave_classification', 'no_waves') != cls_key:
                continue
            house_ids.append(str(a.get('house_id', '?')))
            quality_scores.append(a.get('data_quality', {}).get('quality_score', 0))
            wave_scores.append(wave.get('max_wave_score', 0))
            dominant_phases_list.append(', '.join(wave.get('dominant_phases', [])) or 'None')

        if not house_ids:
            continue

        fig.add_trace(go.Scatter(
            x=quality_scores,
            y=wave_scores,
            mode='markers+text',
            marker=dict(size=12, color=color, line=dict(width=1, color='white')),
            text=house_ids,
            textposition='top center',
            textfont=dict(size=9),
            name=cls_labels[cls_key],
            hovertemplate=(
                'House %{text}<br>'
                'Quality: %{x:.1f}<br>'
                'Wave Score: %{y:.3f}<br>'
                'Phases: %{customdata}<extra></extra>'
            ),
            customdata=dominant_phases_list,
        ))

    fig.update_layout(
        title='Wave Behavior vs Data Quality',
        xaxis_title='Quality Score',
        yaxis_title='Max Wave Score',
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        margin=dict(b=80),
    )

    # Threshold lines
    fig.add_hline(y=WAVE_PRESENT_THRESHOLD, line_dash='dot', line_color='#3498db',
                  opacity=0.5, annotation_text='Has Waves (0.25)')
    fig.add_hline(y=WAVE_DOMINANT_THRESHOLD, line_dash='dot', line_color='#e67e22',
                  opacity=0.5, annotation_text='Wave Dominant (0.50)')

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_phase_imbalance_chart(analysis: Dict[str, Any]) -> str:
    """
    Create per-house phase imbalance indicator with distribution breakdown.

    Shows mean imbalance score as a gauge and % time at each level.

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    imbalance = analysis.get('phase_imbalance', {})
    imb_mean = imbalance.get('imbalance_mean', 0)
    imb_median = imbalance.get('imbalance_median', 0)
    imb_p95 = imbalance.get('imbalance_p95', 0)
    high_pct = imbalance.get('high_imbalance_pct', 0)
    label = imbalance.get('imbalance_label', 'balanced')

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "bar"}]],
        column_widths=[0.4, 0.6],
    )

    # Gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=imb_mean,
        number=dict(valueformat=".3f"),
        gauge=dict(
            axis=dict(range=[0, 1.2], tickvals=[0, 0.2, 0.5, 1.0]),
            bar=dict(color="#667eea"),
            steps=[
                dict(range=[0, 0.2], color="#d4edda"),
                dict(range=[0.2, 0.5], color="#fff3cd"),
                dict(range=[0.5, 1.2], color="#f8d7da"),
            ],
            threshold=dict(line=dict(color="red", width=2), thickness=0.75, value=0.5),
        ),
        title=dict(text="Mean Imbalance"),
    ), row=1, col=1)

    # Summary bars
    categories = ['Mean', 'Median', 'P95', 'High %']
    values = [imb_mean, imb_median, imb_p95, high_pct / 100]
    colors = ['#667eea', '#3498db', '#e67e22', '#e74c3c']

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=colors,
        text=[f'{imb_mean:.3f}', f'{imb_median:.3f}', f'{imb_p95:.3f}', f'{high_pct:.1f}%'],
        textposition='outside',
        hovertemplate='%{x}: %{text}<extra></extra>',
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=f'Phase Imbalance — {label.title()}',
        height=300,
        margin=dict(t=60, b=40),
    )
    fig.update_yaxes(title_text="Score", row=1, col=2)

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_imbalance_comparison_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create horizontal bar chart comparing phase imbalance across houses.

    Args:
        analyses: List of house analysis results

    Returns:
        HTML string with embedded chart
    """
    data = []
    for a in analyses:
        imbalance = a.get('phase_imbalance', {})
        data.append({
            'house_id': str(a.get('house_id', 'unknown')),
            'imbalance_mean': imbalance.get('imbalance_mean', 0),
            'imbalance_label': imbalance.get('imbalance_label', 'balanced'),
            'high_pct': imbalance.get('high_imbalance_pct', 0),
        })

    if not data:
        return "<p>No imbalance data available</p>"

    df = pd.DataFrame(data).sort_values('imbalance_mean', ascending=True)

    colors = []
    for v in df['imbalance_mean']:
        if v < 0.2:
            colors.append('#28a745')
        elif v < 0.5:
            colors.append('#ffc107')
        else:
            colors.append('#dc3545')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['house_id'],
        x=df['imbalance_mean'],
        orientation='h',
        marker_color=colors,
        text=df['imbalance_mean'].round(3).astype(str),
        textposition='outside',
        hovertemplate=(
            'House %{y}<br>'
            'Mean Imbalance: %{x:.3f}<br>'
            'High Imbalance: %{customdata:.1f}% of time<extra></extra>'
        ),
        customdata=df['high_pct'],
    ))

    fig.add_vline(x=0.2, line_dash="dash", line_color="green", opacity=0.5,
                  annotation_text="Balanced")
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", opacity=0.5,
                  annotation_text="Imbalanced")

    fig.update_layout(
        title="Phase Imbalance by House",
        xaxis_title="Mean Imbalance Score (std/mean)",
        yaxis_title="House ID",
        height=max(400, len(df) * 25),
        xaxis_range=[0, max(df['imbalance_mean'].max() * 1.3, 0.6)],
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_year_heatmap(heatmap_data: Dict[str, Any], year: int) -> str:
    """
    Create power heatmap for a specific year.

    Args:
        heatmap_data: Dictionary with 'days', 'hours', and 'values' keys
        year: Year number

    Returns:
        HTML string with embedded chart
    """
    if not heatmap_data:
        return "<p>No heatmap data available</p>"

    days = heatmap_data.get('days', list(range(7)))
    hours = heatmap_data.get('hours', list(range(24)))
    values = heatmap_data.get('values', [[0]*24]*7)

    day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    hour_labels = [f'{h}:00' for h in hours]

    fig = go.Figure(data=go.Heatmap(
        z=values,
        x=hour_labels,
        y=[day_names[d] for d in days],
        colorscale='YlOrRd',
        colorbar=dict(title='Power (W)'),
        hovertemplate='%{y} %{x}<br>Power: %{z:.0f}W<extra></extra>'
    ))

    fig.update_layout(
        title=f'Power Consumption Heatmap - {year}',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=280,
        xaxis=dict(tickmode='array', tickvals=list(range(0, 24, 3)),
                   ticktext=[f'{h}:00' for h in range(0, 24, 3)])
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)
