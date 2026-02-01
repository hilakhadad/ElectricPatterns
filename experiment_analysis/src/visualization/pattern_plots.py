"""
Generate plots for recurring patterns.

Creates individual plots for each occurrence of a recurring pattern,
organized by house and pattern.
"""
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

# Try to import plotly for plot generation
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def generate_pattern_plots(
    experiment_dir: Path,
    house_id: str,
    patterns: List[Dict[str, Any]],
    output_dir: Path,
    run_number: int = 0,
    hours_before: float = 1.0,
    hours_after: float = 1.0
) -> Dict[str, Any]:
    """
    Generate plots for each occurrence of recurring patterns.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        patterns: List of pattern dictionaries from recurring_matches
        output_dir: Directory to save plots
        run_number: Run number to load data from
        hours_before: Hours to show before pattern start
        hours_after: Hours to show after pattern end

    Returns:
        Dictionary with generation statistics
    """
    if not PLOTLY_AVAILABLE:
        return {'error': 'Plotly not available', 'plots_generated': 0}

    result = {
        'plots_generated': 0,
        'patterns_processed': 0,
        'errors': [],
        'output_dir': str(output_dir),
    }

    # Create output directory for this house
    house_output_dir = output_dir / f"house_{house_id}"
    house_output_dir.mkdir(parents=True, exist_ok=True)

    # Load summarized data
    data = _load_summarized_data(experiment_dir, house_id, run_number)
    if data is None:
        result['error'] = f'Could not load data for house {house_id}'
        return result

    # Load matches for highlighting
    matches_df = _load_matches_data(experiment_dir, house_id, run_number)

    for pattern_idx, pattern in enumerate(patterns):
        pattern_phase = pattern.get('phase')  # Can be None
        pattern_mag = pattern.get('magnitude', 0) or 0
        pattern_time = pattern.get('avg_start_time', '00:00')
        pattern_duration = pattern.get('duration_minutes', 0) or 60
        dates = pattern.get('dates', [])

        if not dates:
            continue

        # Create subfolder for this pattern
        time_str = pattern_time.replace(':', '')
        if pattern_phase and pattern_mag:
            pattern_name = f"pattern_{pattern_idx+1}_{pattern_phase}_{pattern_mag}W_{time_str}"
        else:
            pattern_name = f"dates_{time_str}"
        pattern_dir = house_output_dir / pattern_name
        pattern_dir.mkdir(parents=True, exist_ok=True)

        result['patterns_processed'] += 1

        for date_str in dates:
            try:
                # Parse date and pattern time
                date = datetime.strptime(date_str, '%Y-%m-%d')
                time_parts = pattern_time.split(':')
                hour = int(time_parts[0])
                minute = int(time_parts[1]) if len(time_parts) > 1 else 0

                # Create window around pattern
                pattern_start = date.replace(hour=hour, minute=minute)
                window_start = pattern_start - timedelta(hours=hours_before)
                window_end = pattern_start + timedelta(minutes=pattern_duration) + timedelta(hours=hours_after)

                # Filter data to window
                mask = (data['timestamp'] >= window_start) & (data['timestamp'] <= window_end)
                window_data = data[mask].copy()

                if window_data.empty:
                    continue

                # Generate plot
                fig = _create_pattern_plot(
                    window_data,
                    pattern_phase,
                    pattern_mag,
                    pattern_start,
                    pattern_duration,
                    house_id,
                    date_str,
                    matches_df
                )

                # Save plot
                plot_filename = f"{date_str}.html"
                plot_path = pattern_dir / plot_filename
                fig.write_html(str(plot_path), include_plotlyjs='cdn')

                result['plots_generated'] += 1

            except Exception as e:
                result['errors'].append(f"{date_str}: {str(e)}")

    # Create index file for easy navigation
    _create_pattern_index(house_output_dir, patterns, house_id)

    return result


def _load_summarized_data(experiment_dir: Path, house_id: str, run_number: int) -> Optional[pd.DataFrame]:
    """Load summarized data for a house."""
    # Try old structure: experiment/house_{id}/run_N/house_{id}/summarized/
    house_dir = experiment_dir / f"house_{house_id}" / f"run_{run_number}" / f"house_{house_id}"
    summarized_dir = house_dir / "summarized"

    if summarized_dir.exists():
        files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.csv"))
        if files:
            dfs = []
            for f in files:
                df = pd.read_csv(f)
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)

    # Try new structure: experiment/run_N/house_{id}/summarized/
    run_dir = experiment_dir / f"run_{run_number}" / f"house_{house_id}"
    summarized_dir = run_dir / "summarized"

    if summarized_dir.exists():
        files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.csv"))
        if files:
            dfs = []
            for f in files:
                df = pd.read_csv(f)
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)

    return None


def _load_matches_data(experiment_dir: Path, house_id: str, run_number: int) -> Optional[pd.DataFrame]:
    """Load matches data for highlighting events."""
    # Try old structure
    house_dir = experiment_dir / f"house_{house_id}" / f"run_{run_number}" / f"house_{house_id}"
    matches_dir = house_dir / "matches"

    if matches_dir.exists():
        files = list(matches_dir.glob(f"matches_{house_id}_*.csv"))
        if files:
            dfs = []
            for f in files:
                df = pd.read_csv(f)
                for col in ['on_start', 'on_end', 'off_start', 'off_end']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M', errors='coerce')
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)

    # Try new structure
    run_dir = experiment_dir / f"run_{run_number}" / f"house_{house_id}"
    matches_dir = run_dir / "matches"

    if matches_dir.exists():
        files = list(matches_dir.glob(f"matches_{house_id}_*.csv"))
        if files:
            dfs = []
            for f in files:
                df = pd.read_csv(f)
                for col in ['on_start', 'on_end', 'off_start', 'off_end']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M', errors='coerce')
                dfs.append(df)
            return pd.concat(dfs, ignore_index=True)

    return None


def _create_pattern_plot(
    data: pd.DataFrame,
    phase: str,
    magnitude: int,
    pattern_start: datetime,
    duration_minutes: int,
    house_id: str,
    date_str: str,
    matches_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Create a plotly figure for a pattern occurrence.

    Layout: 3 columns (phases w1, w2, w3) x 3 rows (Original, Remaining, Segmented)
    All phases share the same Y-axis range.
    """
    phases = ['w1', 'w2', 'w3']
    phase_colors = {'w1': '#1f77b4', 'w2': '#2ca02c', 'w3': '#ff7f0e'}

    # Check if we have segmentation data
    has_segmentation = any(f'remaining_{p}' in data.columns for p in phases)
    has_duration_data = any(f'short_duration_{p}' in data.columns for p in phases)

    # Determine number of rows
    n_rows = 3 if has_segmentation else 1
    row_titles = ['Original Power', 'Remaining Power', 'Segmented Power'] if has_segmentation else ['Power']

    # Create subplot grid: rows x 3 phases
    fig = make_subplots(
        rows=n_rows, cols=len(phases),
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        subplot_titles=[f"Phase {p.upper()}" for p in phases] + [''] * (len(phases) * (n_rows - 1)),
        row_titles=row_titles
    )

    for col_idx, p in enumerate(phases, start=1):
        color = phase_colors[p]

        # Row 1: Original power
        original_col = f'original_{p}'
        if original_col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data[original_col],
                    mode='lines',
                    name=f'Original {p.upper()}',
                    line=dict(color=color, width=1.5),
                    showlegend=(col_idx == 1),
                ),
                row=1, col=col_idx
            )

        if has_segmentation:
            # Row 2: Remaining power
            remaining_col = f'remaining_{p}'
            if remaining_col in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data['timestamp'],
                        y=data[remaining_col],
                        mode='lines',
                        name=f'Remaining {p.upper()}',
                        line=dict(color=color, width=1.5),
                        showlegend=(col_idx == 1),
                    ),
                    row=2, col=col_idx
                )

            # Row 3: Segmented power by duration
            if has_duration_data:
                duration_cols = [
                    (f'short_duration_{p}', 'Short', 'rgba(255, 99, 132, 0.8)'),
                    (f'medium_duration_{p}', 'Medium', 'rgba(54, 162, 235, 0.8)'),
                    (f'long_duration_{p}', 'Long', 'rgba(75, 192, 192, 0.8)')
                ]

                for col_name, label, seg_color in duration_cols:
                    if col_name in data.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=data['timestamp'],
                                y=data[col_name],
                                mode='lines',
                                name=f'{label} {p.upper()}',
                                line=dict(width=1.5),
                                showlegend=(col_idx == 1),
                            ),
                            row=3, col=col_idx
                        )

        # Highlight the pattern time window (on all phases, or just the specified one)
        pattern_end = pattern_start + timedelta(minutes=duration_minutes)
        highlight_this_phase = (phase is None) or (p == phase)

        if highlight_this_phase:
            for row in range(1, n_rows + 1):
                fig.add_vrect(
                    x0=pattern_start, x1=pattern_end,
                    fillcolor="rgba(255, 0, 0, 0.15)",
                    layer="below",
                    line_width=0,
                    row=row, col=col_idx
                )

            # Add magnitude annotation on the specific phase (if provided)
            if phase and p == phase and magnitude:
                fig.add_annotation(
                    x=pattern_start,
                    y=magnitude,
                    text=f"{magnitude}W",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    font=dict(color="red", size=10),
                    row=1, col=col_idx
                )

    # Calculate unified Y-axis range across all phases and rows
    y_min, y_max = _calculate_y_range(data, phases)
    for row in range(1, n_rows + 1):
        for col in range(1, len(phases) + 1):
            fig.update_yaxes(range=[y_min, y_max], row=row, col=col)

    # Build title
    if phase and magnitude:
        title_text = f"House {house_id} - {date_str} - {phase.upper()} {magnitude}W @ {pattern_start.strftime('%H:%M')}"
    else:
        title_text = f"House {house_id} - {date_str} @ {pattern_start.strftime('%H:%M')}"

    # Update layout
    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        height=300 * n_rows,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        template="plotly_white"
    )

    # Add Y-axis labels
    for row in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Power (W)", row=row, col=1)

    # Add X-axis label to bottom row
    for col in range(1, len(phases) + 1):
        fig.update_xaxes(title_text="Time", row=n_rows, col=col)

    return fig


def _calculate_y_range(df: pd.DataFrame, phases: list) -> tuple:
    """Calculate unified y-axis range across all relevant columns."""
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

    # Add padding
    if y_min != float('inf') and y_max != float('-inf'):
        padding = (y_max - y_min) * 0.05
        return (y_min - padding, y_max + padding)

    return (0, 1000)


def _create_pattern_index(house_dir: Path, patterns: List[Dict], house_id: str):
    """Create an index HTML file for easy navigation of pattern plots."""
    html_parts = [f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pattern Plots - House {house_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .pattern-card {{
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .pattern-header {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #2c3e50;
        }}
        .pattern-details {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 10px;
        }}
        .dates-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 5px;
        }}
        .date-link {{
            display: block;
            padding: 5px 10px;
            background: #e3f2fd;
            border-radius: 4px;
            text-decoration: none;
            color: #1976d2;
            font-size: 0.85em;
            text-align: center;
        }}
        .date-link:hover {{
            background: #bbdefb;
        }}
    </style>
</head>
<body>
    <h1>Pattern Plots - House {house_id}</h1>
    <p>Click on a date to view the pattern plot.</p>
"""]

    for pattern_idx, pattern in enumerate(patterns):
        phase = pattern.get('phase', 'unknown')
        mag = pattern.get('magnitude', 0)
        time_str = pattern.get('avg_start_time', '00:00')
        duration = pattern.get('duration_minutes', 0)
        dates = pattern.get('dates', [])
        interval = pattern.get('interval_type', '')

        pattern_name = f"pattern_{pattern_idx+1}_{phase}_{mag}W_{time_str.replace(':', '')}"

        date_links = []
        for date_str in dates:
            link_path = f"{pattern_name}/{date_str}.html"
            date_links.append(f'<a href="{link_path}" class="date-link" target="_blank">{date_str}</a>')

        html_parts.append(f"""
    <div class="pattern-card">
        <div class="pattern-header">
            Pattern {pattern_idx + 1}: {phase.upper()} - {mag}W @ {time_str}
        </div>
        <div class="pattern-details">
            Duration: {duration} min | Frequency: {interval} | Occurrences: {len(dates)}
        </div>
        <div class="dates-grid">
            {''.join(date_links)}
        </div>
    </div>
""")

    html_parts.append("""
</body>
</html>
""")

    index_path = house_dir / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
