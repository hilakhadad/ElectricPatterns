"""
Static Matplotlib-based visualization.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib.lines import Line2D
from typing import List, Optional

matplotlib.use('Agg')

from .utils import simplify_event_id, create_title, get_save_path

# Color scheme
COLORS = {
    'original': 'black',
    'remaining': 'blue',
    'short': 'green',
    'medium': 'orange',
    'long': 'purple',
}


def plot_static(
    df: pd.DataFrame,
    phases: List[str],
    filepath: str,
    events: Optional[pd.DataFrame] = None
) -> None:
    """
    Create static Matplotlib visualization.

    Args:
        df: DataFrame with segmented data
        phases: List of phase names (w1, w2, w3)
        filepath: Path for saving the plot
        events: Optional DataFrame with event markers
    """
    import os
    house_id = os.path.splitext(os.path.basename(filepath))[0]
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)

    sorted_phases = sorted(phases, reverse=True)
    y_min, y_max = _calculate_y_limits(df, sorted_phases)

    fig, axes = plt.subplots(
        nrows=4, ncols=len(phases),
        figsize=(18, 12), sharex=True, sharey=True,
        constrained_layout=True
    )

    time_formatter = mdates.DateFormatter('%H:%M')

    _set_titles(axes, sorted_phases)
    _plot_all_phases(axes, df, sorted_phases, time_formatter, y_min, y_max, events)
    _add_legend(fig, axes, events)

    title = create_title(df['timestamp'], sorted_phases, "combined", house_id)
    fig.suptitle(title, fontsize=16, y=1.02)

    filename = get_save_path(f'{title}.png', filepath)
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved combined plot: {filename}")
    plt.clf()
    plt.close()


def _calculate_y_limits(df: pd.DataFrame, phases: List[str]) -> tuple:
    """Calculate y-axis limits across all phases."""
    y_min, y_max = float('inf'), float('-inf')

    for phase in phases:
        cols = [f'original_{phase}', f'remaining_{phase}',
                f'short_duration_{phase}', f'medium_duration_{phase}', f'long_duration_{phase}']
        existing_cols = [c for c in cols if c in df.columns]
        all_values = pd.concat([df[c] for c in existing_cols])
        y_min = min(y_min, all_values.min())
        y_max = max(y_max, all_values.max())

    return y_min, y_max


def _set_titles(axes, phases: List[str]) -> None:
    """Set column and row titles."""
    for col_idx, phase in enumerate(phases):
        axes[0, col_idx].set_title(f"Phase {phase}", fontsize=12, fontweight='bold')

    row_titles = ["Original Data", "After Segregation", "Segregation Data", "Event Markers"]
    for row_idx, title in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(title, fontsize=12, fontweight='bold')


def _plot_all_phases(axes, df, phases, time_formatter, y_min, y_max, events) -> None:
    """Plot data for all phases."""
    for col_idx, phase in enumerate(phases):
        _plot_phase(axes, df, phase, col_idx, time_formatter, y_min, y_max, events)


def _plot_phase(axes, df, phase, col_idx, time_formatter, y_min, y_max, events) -> None:
    """Plot data for a single phase."""
    # Row 0: Original
    axes[0, col_idx].plot(
        df['timestamp'], df[f'original_{phase}'],
        label='Original' if col_idx == 0 else None,
        color=COLORS['original']
    )
    _configure_axis(axes[0, col_idx], time_formatter, y_min, y_max)

    # Row 1: Remaining
    axes[1, col_idx].plot(
        df['timestamp'], df[f'remaining_{phase}'],
        label='Remaining' if col_idx == 0 else None,
        color=COLORS['remaining']
    )
    _configure_axis(axes[1, col_idx], time_formatter, y_min, y_max)

    # Row 2: Duration categories
    for duration in ['short', 'medium', 'long']:
        col_name = f'{duration}_duration_{phase}'
        if col_name in df.columns:
            label = f'{duration.capitalize()} duration' if col_idx == 0 else None
            axes[2, col_idx].plot(
                df['timestamp'], df[col_name],
                label=label, color=COLORS[duration]
            )
    _configure_axis(axes[2, col_idx], time_formatter, y_min, y_max)

    # Row 3: Event markers
    if events is not None:
        _plot_event_markers(axes[3, col_idx], events, phase, y_max)
    _configure_axis(axes[3, col_idx], time_formatter, y_min, y_max)


def _configure_axis(ax, time_formatter, y_min, y_max) -> None:
    """Configure axis formatting."""
    ax.xaxis.set_major_formatter(time_formatter)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(y_min, y_max)


def _plot_event_markers(ax, events, phase, y_max) -> None:
    """Plot event markers for a phase."""
    phase_events = events[events['phase'] == phase]

    for idx, event in phase_events.iterrows():
        color = _get_event_color(event)
        end_y = min(abs(event['magnitude']), y_max)

        ax.plot([event['start'], event['start']], [0, end_y],
                color=color, linestyle=(0, (int(idx % 5), 3, 3, 3)))
        ax.plot([event['end'], event['end']], [0, end_y],
                color=color, linestyle=(0, (int(idx % 5), 3, 3, 3)))

        ax.text(event['start'], end_y + (y_max * 0.02),
                f"{simplify_event_id(event['event_id'])}_S",
                color=color, fontsize=8, rotation=45, ha='right')
        ax.text(event['end'], end_y + (y_max * 0.02),
                f"{simplify_event_id(event['event_id'])}_E",
                color=color, fontsize=8, rotation=45, ha='left')


def _get_event_color(event: pd.Series) -> str:
    """Determine event color based on matched status."""
    if event.get('matched', 0) == 1:
        return 'green'
    elif event['event_id'].startswith('on_'):
        return 'red'
    return 'blue'


def _add_legend(fig, axes, events) -> None:
    """Add legend to figure."""
    handles, labels = [], []
    for row_idx in range(3):
        ax_handles, ax_labels = axes[row_idx, 0].get_legend_handles_labels()
        for h, l in zip(ax_handles, ax_labels):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    if events is not None:
        handles.append(Line2D([0], [0], color='green', linestyle='--'))
        labels.append('Matched event')
        handles.append(Line2D([0], [0], color='red', linestyle='--'))
        labels.append('Unmatched ON')
        handles.append(Line2D([0], [0], color='blue', linestyle='--'))
        labels.append('Unmatched OFF')

    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.12, 0.98))
