"""
Interactive Plotly-based visualization.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional

from .utils import simplify_event_id, create_title, get_save_path

# Color scheme
COLORS = {
    'original': 'black',
    'remaining': 'blue',
    'short': 'green',
    'medium': 'orange',
    'long': 'purple',
    'matched': 'green',
    'unmatched_on': 'red',
    'unmatched_off': 'blue',
}


def plot_interactive(
    df: pd.DataFrame,
    phases: List[str],
    filepath: str,
    logger,
    events: Optional[pd.DataFrame] = None
) -> None:
    """
    Create interactive Plotly visualization.

    Args:
        df: DataFrame with segmented data
        phases: List of phase names (w1, w2, w3)
        filepath: Path for saving the plot
        logger: Logger instance
        events: Optional DataFrame with event markers
    """
    import os
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
    house_id = os.path.splitext(os.path.basename(filepath))[0]

    fig = make_subplots(
        rows=4, cols=len(phases),
        shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[f"Phase {phase}" for phase in phases]
    )

    _add_legend_entries(fig)
    _add_data_traces(fig, df, phases)

    if events is not None:
        _add_event_markers(fig, events, phases)

    _configure_layout(fig)

    title = create_title(df['timestamp'], phases, "interactive", house_id)
    filename = get_save_path(f'{title}.html', filepath)
    fig.write_html(filename)
    logger.info(f"Interactive plot saved to {filename}")


def _add_legend_entries(fig: go.Figure) -> None:
    """Add invisible traces for consistent legend."""
    legend_items = [
        ('Original', COLORS['original']),
        ('Remaining', COLORS['remaining']),
        ('Short duration', COLORS['short']),
        ('Medium duration', COLORS['medium']),
        ('Long duration', COLORS['long']),
    ]

    for name, color in legend_items:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            name=name, line=dict(color=color), showlegend=True
        ))

    # Event markers legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines+markers',
        name='Matched event', line=dict(dash='dash', color='green'), showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines+markers',
        name='Unmatched ON', line=dict(dash='dash', color='red'), showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines+markers',
        name='Unmatched OFF', line=dict(dash='dash', color='blue'), showlegend=True
    ))


def _add_data_traces(fig: go.Figure, df: pd.DataFrame, phases: List[str]) -> None:
    """Add data traces for all phases."""
    for col_idx, phase in enumerate(phases, start=1):
        # Row 1: Original data
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df[f'original_{phase}'],
                mode='lines', line=dict(color=COLORS['original']),
                showlegend=False
            ), row=1, col=col_idx
        )

        # Row 2: After segregation
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df[f'remaining_{phase}'],
                mode='lines', line=dict(color=COLORS['remaining']),
                showlegend=False
            ), row=2, col=col_idx
        )

        # Row 3: Segregated by duration
        for duration_type in ['short', 'medium', 'long']:
            col_name = f'{duration_type}_duration_{phase}'
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'], y=df[col_name],
                        mode='lines', line=dict(color=COLORS[duration_type]),
                        showlegend=False
                    ), row=3, col=col_idx
                )


def _add_event_markers(fig: go.Figure, events: pd.DataFrame, phases: List[str]) -> None:
    """Add event markers to row 4."""
    for col_idx, phase in enumerate(phases, start=1):
        phase_events = events[events['phase'] == phase]

        for _, event in phase_events.iterrows():
            color = _get_event_color(event)
            fig.add_trace(
                go.Scatter(
                    x=[event['start'], event['end']],
                    y=[0, event['magnitude']],
                    mode='lines+markers',
                    name=f"Event {simplify_event_id(event['event_id'])}",
                    line=dict(dash='dash', color=color),
                    showlegend=False
                ), row=4, col=col_idx
            )


def _get_event_color(event: pd.Series) -> str:
    """Determine event color based on matched status."""
    if event.get('matched', 0) == 1:
        return COLORS['matched']
    elif event['event_id'].startswith('on_'):
        return COLORS['unmatched_on']
    return COLORS['unmatched_off']


def _configure_layout(fig: go.Figure) -> None:
    """Configure figure layout."""
    row_titles = ["Original Data", "After Segregation", "Segregation Data", "Event Markers"]
    for row_idx, title in enumerate(row_titles, start=1):
        fig.update_yaxes(title_text=title, row=row_idx, col=1)

    fig.update_layout(
        title="Interactive Combined Phases Plot",
        hovermode="x unified",
        showlegend=True
    )
