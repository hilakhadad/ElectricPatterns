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

    all_flags = sorted(all_flags)

    # Build matrix
    matrix = []
    for a in analyses:
        flags = a.get('flags', {})
        row = [1 if flags.get(f, False) else 0 for f in all_flags]
        matrix.append(row)

    # Clean up flag names for display
    display_flags = [f.replace('_', ' ').title() for f in all_flags]

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
