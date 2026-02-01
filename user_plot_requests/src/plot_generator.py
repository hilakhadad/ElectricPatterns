"""
Plot generation for user requests.

Creates Plotly interactive charts showing power data and segmentation.
"""
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


_MODULE_DIR = Path(__file__).parent.parent.absolute()
_CACHE_DIR = _MODULE_DIR / "OUTPUT" / "plots"


def get_cached_plot_path(house_id: str, date: str, window_type: str) -> Path:
    """Get the path where a cached plot would be stored."""
    return _CACHE_DIR / f"house_{house_id}_{date}_{window_type}.html"


def plot_exists(house_id: str, date: str, window_type: str) -> bool:
    """Check if a plot already exists in cache."""
    return get_cached_plot_path(house_id, date, window_type).exists()


def generate_plot(
    df: pd.DataFrame,
    house_id: str,
    date: str,
    window_type: str,
    save_to_cache: bool = True,
    events_df: pd.DataFrame = None
) -> str:
    """
    Generate interactive Plotly plot for power data.

    Args:
        df: DataFrame with timestamp and power columns
        house_id: House ID
        date: Date string (YYYY-MM-DD)
        window_type: 'day' or 'night'
        save_to_cache: Whether to save HTML to cache
        events_df: Optional DataFrame with on/off events

    Returns:
        HTML string of the plot
    """
    phases = ['w1', 'w2', 'w3']
    phase_colors = {'w1': '#1f77b4', 'w2': '#2ca02c', 'w3': '#ff7f0e'}

    # Check what columns exist
    has_segmentation = any(f'remaining_{p}' in df.columns for p in phases)

    if has_segmentation:
        # Full segmentation view: 4 rows x 3 phases (added events row)
        fig = _create_segmentation_plot(df, phases, phase_colors, events_df)
        title = f"House {house_id} - Segmentation Results"
    else:
        # Simple view: just original power
        fig = _create_simple_plot(df, phases, phase_colors)
        title = f"House {house_id} - Power Data (No Segmentation)"

    window_label = "06:00-18:00" if window_type == 'day' else "18:00-06:00"
    fig.update_layout(
        title=f"{title}<br><sub>{date} ({window_label})</sub>",
        hovermode="x unified",
        showlegend=True,
        height=900 if events_df is not None else 800,
        template="plotly_white"
    )

    html = fig.to_html(full_html=True, include_plotlyjs=True)

    if save_to_cache:
        cache_path = get_cached_plot_path(house_id, date, window_type)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(html)

    return html


def _create_segmentation_plot(
    df: pd.DataFrame,
    phases: list,
    phase_colors: dict,
    events_df: pd.DataFrame = None
) -> go.Figure:
    """Create full segmentation plot with 3-4 rows."""
    # Determine subplot layout
    has_events = events_df is not None and not events_df.empty
    n_rows = 4 if has_events else 3

    if has_events:
        row_titles = ['Original Power', 'Remaining After Segmentation', 'Segmented Power', 'ON/OFF Events']
    else:
        row_titles = ['Original Power', 'Remaining After Segmentation', 'Segmented Power']

    fig = make_subplots(
        rows=n_rows, cols=len(phases),
        shared_xaxes=True,
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
        subplot_titles=[f"Phase {p}" for p in phases] + [''] * (len(phases) * (n_rows - 1)),
        row_titles=row_titles
    )

    for col_idx, phase in enumerate(phases, start=1):
        color = phase_colors[phase]

        # Row 1: Original data
        original_col = f'original_{phase}'
        if original_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=df[original_col],
                    mode='lines', name=f'Original {phase}',
                    line=dict(color=color, width=1),
                    showlegend=(col_idx == 1)
                ),
                row=1, col=col_idx
            )

        # Row 2: Remaining after segmentation
        remaining_col = f'remaining_{phase}'
        if remaining_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=df[remaining_col],
                    mode='lines', name=f'Remaining {phase}',
                    line=dict(color=color, width=1),
                    showlegend=(col_idx == 1)
                ),
                row=2, col=col_idx
            )

        # Row 3: Segmented data by duration
        duration_cols = [
            (f'short_duration_{phase}', 'Short', 'rgba(255, 99, 132, 0.7)'),
            (f'medium_duration_{phase}', 'Medium', 'rgba(54, 162, 235, 0.7)'),
            (f'long_duration_{phase}', 'Long', 'rgba(75, 192, 192, 0.7)')
        ]

        for col_name, label, seg_color in duration_cols:
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'], y=df[col_name],
                        mode='lines', name=f'{label} {phase}',
                        line=dict(width=1),
                        showlegend=(col_idx == 1)
                    ),
                    row=3, col=col_idx
                )

        # Row 4: ON/OFF Events
        if has_events:
            phase_events = events_df[events_df['phase'] == phase]

            # ON events (positive, green markers)
            on_events = phase_events[phase_events['event'] == 'on']
            if not on_events.empty:
                fig.add_trace(
                    go.Scatter(
                        x=on_events['start'],
                        y=on_events['magnitude'],
                        mode='markers',
                        name=f'ON {phase}',
                        marker=dict(color='green', size=8, symbol='triangle-up'),
                        text=[f"ON: {m:.0f}W" for m in on_events['magnitude']],
                        hovertemplate='%{text}<br>%{x}<extra></extra>',
                        showlegend=(col_idx == 1)
                    ),
                    row=4, col=col_idx
                )

            # OFF events (negative shown as positive, red markers)
            off_events = phase_events[phase_events['event'] == 'off']
            if not off_events.empty:
                fig.add_trace(
                    go.Scatter(
                        x=off_events['start'],
                        y=off_events['magnitude'].abs(),
                        mode='markers',
                        name=f'OFF {phase}',
                        marker=dict(color='red', size=8, symbol='triangle-down'),
                        text=[f"OFF: {abs(m):.0f}W" for m in off_events['magnitude']],
                        hovertemplate='%{text}<br>%{x}<extra></extra>',
                        showlegend=(col_idx == 1)
                    ),
                    row=4, col=col_idx
                )

    # Calculate y-axis range
    y_min, y_max = _calculate_y_range(df, phases)
    for row in range(1, 4):
        for col in range(1, len(phases) + 1):
            fig.update_yaxes(range=[y_min, y_max], row=row, col=col)

    # Events row has its own scale
    if has_events and not events_df.empty:
        event_max = events_df['magnitude'].abs().max() * 1.1
        for col in range(1, len(phases) + 1):
            fig.update_yaxes(range=[0, event_max], row=4, col=col)

    return fig


def _create_simple_plot(
    df: pd.DataFrame,
    phases: list,
    phase_colors: dict
) -> go.Figure:
    """Create simple plot with just original power."""
    # Map phase names to possible column names in raw data
    phase_mapping = {
        'w1': ['w1', '1', 'phase1', 'p1'],
        'w2': ['w2', '2', 'phase2', 'p2'],
        'w3': ['w3', '3', 'phase3', 'p3']
    }

    fig = make_subplots(
        rows=1, cols=len(phases),
        shared_xaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=[f"Phase {p}" for p in phases]
    )

    for col_idx, phase in enumerate(phases, start=1):
        color = phase_colors[phase]

        # Try different column names for raw data
        possible_cols = [f'original_{phase}', phase] + phase_mapping.get(phase, [])
        for col_name in possible_cols:
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'], y=df[col_name],
                        mode='lines', name=f'Power {phase}',
                        line=dict(color=color, width=1),
                        showlegend=(col_idx == 1)
                    ),
                    row=1, col=col_idx
                )
                break

    # Calculate y-axis range
    y_min, y_max = _calculate_simple_y_range(df, phases, phase_mapping)
    for col in range(1, len(phases) + 1):
        fig.update_yaxes(range=[y_min, y_max], row=1, col=col)

    return fig


def _calculate_simple_y_range(df: pd.DataFrame, phases: list, phase_mapping: dict) -> tuple:
    """Calculate y-axis range for simple plot."""
    y_min = float('inf')
    y_max = float('-inf')

    for phase in phases:
        possible_cols = [f'original_{phase}', phase] + phase_mapping.get(phase, [])
        for col in possible_cols:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if pd.notna(col_min):
                    y_min = min(y_min, col_min)
                if pd.notna(col_max):
                    y_max = max(y_max, col_max)
                break

    if y_min != float('inf') and y_max != float('-inf'):
        padding = (y_max - y_min) * 0.05
        return (y_min - padding, y_max + padding)

    return (0, 1000)


def _calculate_y_range(df: pd.DataFrame, phases: list) -> tuple:
    """Calculate y-axis range across all relevant columns."""
    y_min = float('inf')
    y_max = float('-inf')

    for phase in phases:
        columns = [
            f'original_{phase}', f'remaining_{phase}',
            f'short_duration_{phase}', f'medium_duration_{phase}', f'long_duration_{phase}'
        ]
        for col in columns:
            if col in df.columns:
                col_min = df[col].min()
                col_max = df[col].max()
                if pd.notna(col_min):
                    y_min = min(y_min, col_min)
                if pd.notna(col_max):
                    y_max = max(y_max, col_max)

    # Add some padding
    if y_min != float('inf') and y_max != float('-inf'):
        padding = (y_max - y_min) * 0.05
        return (y_min - padding, y_max + padding)

    return (0, 1000)


def generate_embedded_plot(
    df: pd.DataFrame,
    house_id: str,
    date: str,
    window_type: str
) -> str:
    """
    Generate plot HTML suitable for embedding in a page (no full HTML wrapper).

    Returns just the div and script for the plot.
    """
    phases = ['w1', 'w2', 'w3']
    phase_colors = {'w1': '#1f77b4', 'w2': '#2ca02c', 'w3': '#ff7f0e'}

    has_segmentation = any(f'remaining_{p}' in df.columns for p in phases)

    if has_segmentation:
        fig = _create_segmentation_plot(df, phases, phase_colors)
        title = f"House {house_id} - Segmentation Results"
    else:
        fig = _create_simple_plot(df, phases, phase_colors)
        title = f"House {house_id} - Power Data"

    window_label = "06:00-18:00" if window_type == 'day' else "18:00-06:00"
    fig.update_layout(
        title=f"{title}<br><sub>{date} ({window_label})</sub>",
        hovermode="x unified",
        showlegend=True,
        height=700,
        template="plotly_white"
    )

    # Use 'cdn' to reference the already-loaded Plotly library
    return fig.to_html(full_html=False, include_plotlyjs='cdn')
