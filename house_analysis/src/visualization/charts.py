"""
Chart generation for house analysis reports.

Uses Plotly to create interactive charts that can be embedded in HTML.
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


def create_quality_distribution_chart(analyses: List[Dict[str, Any]]) -> str:
    """
    Create histogram of quality scores across all houses.

    Args:
        analyses: List of house analysis results

    Returns:
        HTML string with embedded chart
    """
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
    data = []

    for a in analyses:
        house_id = a.get('house_id', 'unknown')
        power = a.get('power_statistics', {})

        # Try both naming conventions
        phases = {}
        for phase_name in ['w1', 'w2', 'w3', '1', '2', '3']:
            key = f'phase_{phase_name}_mean'
            if key in power:
                # Normalize phase name
                normalized = phase_name.replace('w', 'Phase ')
                if normalized.isdigit():
                    normalized = f'Phase {normalized}'
                phases[normalized] = power[key]

        if phases:
            total = sum(phases.values())
            for phase, value in phases.items():
                data.append({
                    'house_id': house_id,
                    'phase': phase,
                    'mean_power': value,
                    'percentage': (value / total * 100) if total > 0 else 0
                })

    if not data:
        return "<p>No phase data available</p>"

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x='house_id',
        y='mean_power',
        color='phase',
        title='Phase Balance by House',
        labels={'mean_power': 'Mean Power (W)', 'house_id': 'House ID'},
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
    )

    fig.update_layout(
        barmode='stack',
        height=400,
        xaxis_tickangle=-45
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


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
        'has_dead_phase': 'Dead Phase (<1%)',
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


def create_hourly_pattern_chart(analysis: Dict[str, Any]) -> str:
    """
    Create hourly power pattern chart with confidence band.

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
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
    colors = ['#667eea' if d not in [4, 5] else '#e74c3c' for d in days]

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
    Create a waterfall/bar chart showing quality score breakdown.

    Args:
        analysis: Single house analysis results

    Returns:
        HTML string with embedded chart
    """
    quality = analysis.get('data_quality', {})

    # Get score components
    coverage_score = quality.get('coverage_score_contribution', 0)
    days_score = quality.get('days_score_contribution', 0)
    secondary_score = quality.get('secondary_score_contribution', 0)
    balance_penalty = quality.get('balance_penalty', 0)
    final_score = quality.get('quality_score', 0)

    # Create waterfall chart
    categories = ['Coverage<br>(max 60)', 'Days<br>(max 30)', 'Secondary<br>(max 10)',
                  'Balance<br>Penalty', 'Final<br>Score']
    values = [coverage_score, days_score, secondary_score, -balance_penalty, final_score]
    colors = ['#667eea', '#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    # Use bar chart with annotations
    fig = go.Figure()

    # Show components as bars
    fig.add_trace(go.Bar(
        x=categories[:-1],
        y=values[:-1],
        marker_color=colors[:-1],
        text=[f'{v:.1f}' if v >= 0 else f'{v:.1f}' for v in values[:-1]],
        textposition='outside',
        name='Components',
        hovertemplate='%{x}<br>Points: %{y:.1f}<extra></extra>'
    ))

    # Add final score as separate bar with different styling
    fig.add_trace(go.Bar(
        x=[categories[-1]],
        y=[values[-1]],
        marker_color=colors[-1],
        text=[f'{final_score:.1f}'],
        textposition='outside',
        name='Final Score',
        hovertemplate='%{x}<br>Score: %{y:.1f}<extra></extra>'
    ))

    fig.update_layout(
        title='Quality Score Breakdown',
        yaxis_title='Points',
        height=350,
        showlegend=False,
        yaxis=dict(range=[min(0, min(values)) - 5, max(values) + 10])
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)
