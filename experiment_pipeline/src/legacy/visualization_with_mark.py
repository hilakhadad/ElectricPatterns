from tqdm import tqdm
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
from matplotlib.lines import Line2D
matplotlib.use('Agg')  # Set the non-GUI backend for headless environments
import sys
from data_util import *
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def simplify_event_id(event_id):
    match = re.match(r"(on|off)_w\d+_(\d+)", event_id)
    if match:
        event_type, unique_id = match.groups()
        return f"{event_type}_{unique_id}"
    return event_id


# Utility functions
def create_seg_title(first_timestamp, columns, value, house_id):
    house_id = house_id.split('_')[1] if "summarized" in house_id else house_id
    date = first_timestamp.iloc[0].date()
    first_time = first_timestamp.iloc[0].time().strftime("%H_%M")
    columns = '_'.join([col for col in columns if col != 'timestamp'])
    title = f'{house_id}_{date}_{first_time}_{columns}_{value}'
    return title

def split_by_day_night_from_path(filepath, sessions_df, chunksize=10000):
    total_rows = sum(1 for _ in open(filepath)) - 1
    total_chunks = (total_rows // chunksize) + (1 if total_rows % chunksize > 0 else 0)

    segments = []
    points_by_segment = []
    segment_start = None

    for chunk in tqdm(pd.read_csv(filepath, chunksize=chunksize, dayfirst=True), total=total_chunks):
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='mixed', dayfirst=True)
        chunk = chunk.sort_values('timestamp').reset_index(drop=True)

        if segment_start is None:
            first_time = chunk.iloc[0]['timestamp']
            if 6 <= first_time.hour < 18:
                segment_start = first_time.replace(hour=6, minute=0, second=0)
            else:
                segment_start = first_time.replace(hour=18, minute=0, second=0)
                if first_time.hour < 6:
                    segment_start -= timedelta(days=1)
                if segment_start < first_time:
                    segment_start += timedelta(hours=12)

        while segment_start <= chunk['timestamp'].max():
            segment_end = segment_start + timedelta(hours=12)
            segment = chunk[(chunk['timestamp'] >= segment_start) & (chunk['timestamp'] < segment_end)]

            # Keep only segments containing relevant sessions
            relevant_sessions = sessions_df[
                (sessions_df['start'] < segment_end) &
                (sessions_df['end'] > segment_start)
            ]

            if not segment.empty and not relevant_sessions.empty:
                segments.append(segment)

                # Ensure we also pass relevant events to plotting function
                points_in_segment = sessions_df[
                    (sessions_df['start'] < segment_end) &
                    (sessions_df['end'] > segment_start)
                ]
                points_by_segment.append(points_in_segment)

            segment_start = segment_end

    return segments, points_by_segment

def check_or_create_house_directory(path):
    house_id = os.path.splitext(os.path.basename(path))[0]
    plot_dir_path = os.path.join(os.path.dirname(os.path.dirname(path)), "plots")
    house_plot_dir_path = os.path.join(plot_dir_path, house_id)
    os.makedirs(house_plot_dir_path, exist_ok=True)
    return house_plot_dir_path

def generate_save_filename_with_title(title, filepath):
    dir_of_plots = check_or_create_house_directory(filepath)
    return os.path.join(dir_of_plots, title)


def plot_combined_phases_interactive(df, phases, filepath, logger, additional_points=None):
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)

    house_id = os.path.splitext(os.path.basename(filepath))[0]

    # Create subplot grid with correct titles
    fig = make_subplots(
        rows=4, cols=len(phases),
        shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[f"Phase {phase}" for phase in phases]  # Titles only for the top row
    )

    # Define colors - same for all phases
    original_color = "black"  # For original data
    main_color = "blue"  # For remaining data
    duration_colors = {
        "short": "green",
        "medium": "orange",
        "long": "purple"
    }

    row_titles = ["Original Data", "After Segregation", "Segregation Data", "Event Markers"]

    # Track which legend entries we've already added
    legend_added = {'original': False, 'remaining': False, 'short': False, 'medium': False, 'long': False, 'on': False, 'off': False}

    # Add invisible traces for legend items that should always appear
    # This ensures consistent legend even when some data types are missing
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Original',
                             line=dict(color=original_color), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Remaining',
                             line=dict(color=main_color), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Short duration',
                             line=dict(color=duration_colors["short"]), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Medium duration',
                             line=dict(color=duration_colors["medium"]), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Long duration',
                             line=dict(color=duration_colors["long"]), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+markers', name='Matched event',
                             line=dict(dash='dash', color='green'), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+markers', name='Unmatched ON',
                             line=dict(dash='dash', color='red'), showlegend=True))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+markers', name='Unmatched OFF',
                             line=dict(dash='dash', color='blue'), showlegend=True))

    # Mark all as added since we created the legend entries above
    legend_added = {k: True for k in legend_added}

    for col_idx, phase in enumerate(phases, start=1):
        original_values = f'original_{phase}'
        cols_to_plot_after_seg = f'remaining_{phase}'
        cols_to_plot_seg = [f'short_duration_{phase}', f'medium_duration_{phase}', f'long_duration_{phase}']

        # Row 1: Original Data
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df[original_values],
                mode='lines', name='Original', line=dict(color=original_color),
                showlegend=False
            ), row=1, col=col_idx
        )

        # Row 2: After Segregation
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df[cols_to_plot_after_seg],
                mode='lines', name='Remaining', line=dict(color=main_color),
                showlegend=False
            ), row=2, col=col_idx
        )

        # Row 3: Segregation Data
        for col in cols_to_plot_seg:
            if col in df.columns:
                # Determine duration type from column name
                if 'short' in col:
                    color = duration_colors["short"]
                elif 'medium' in col:
                    color = duration_colors["medium"]
                else:
                    color = duration_colors["long"]
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'], y=df[col],
                        mode='lines', line=dict(color=color),
                        showlegend=False
                    ), row=3, col=col_idx
                )

        # Row 4: Additional Points (if any)
        if additional_points is not None:
            for _, point in additional_points[additional_points['phase'] == phase].iterrows():
                # Determine color based on matched status and event type
                # Matched events = green, Unmatched ON = red, Unmatched OFF = blue
                if point.get('matched', 0) == 1:
                    event_color = 'green'  # Matched event
                elif point['event_id'].startswith('on_'):
                    event_color = 'red'    # Unmatched ON
                else:
                    event_color = 'blue'   # Unmatched OFF
                fig.add_trace(
                    go.Scatter(
                        x=[point['start'], point['end']], y=[0, point['magnitude']],
                        mode='lines+markers',
                        name=f"Event {simplify_event_id(point['event_id'])}",
                        line=dict(dash='dash', color=event_color),
                        showlegend=False  # Don't show in legend
                    ), row=4, col=col_idx
                )

    # Add Y-axis labels per row
    for row_idx, title in enumerate(row_titles, start=1):
        fig.update_yaxes(title_text=title, row=row_idx, col=1)

    # Update layout for interactivity
    fig.update_layout(
        title="Interactive Combined Phases Plot",
        hovermode="x unified",
        showlegend=True
    )

    title = create_seg_title(
        first_timestamp=df['timestamp'],
        columns=phases,
        value="interactive",
        house_id=house_id
    )

    filename = generate_save_filename_with_title(f'{title}.html', filepath)
    fig.write_html(filename)  # Save the plot as an HTML file
    logger.info(f"Interactive plot saved to {filename}")



def plot_combined_phases(df, phases, filepath, additional_points=None):
    house_id = os.path.splitext(os.path.basename(filepath))[0]

    # Define colors - same for all phases
    original_color = "black"  # For original and remaining data
    main_color = "blue"  
    duration_colors = {
        "short": "green",
        "medium": "orange",
        "long": "purple"
    }

    fig, axes = plt.subplots(nrows=4, ncols=len(phases), figsize=(18, 12), sharex=True, sharey=True,
                             constrained_layout=True)
    time_formatter = mdates.DateFormatter('%H:%M')

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)

    # Print phases order to debug alignment issues
    print("Phases order in plots:", phases)

    # Ensure correct order of phases if needed (manually adjust if necessary)
    sorted_phases = sorted(phases, reverse=True)  # Reverse if order is incorrect

    # Compute y-axis limits across all phases
    y_min, y_max = float('inf'), float('-inf')
    for phase in sorted_phases:
        original_values = f'original_{phase}'
        cols_to_plot_after_seg = f'remaining_{phase}'
        cols_to_plot_seg = [f'short_duration_{phase}', f'medium_duration_{phase}', f'long_duration_{phase}']

        all_values = pd.concat([
            df[original_values],
            df[cols_to_plot_after_seg],
            *(df[col] for col in cols_to_plot_seg if col in df.columns)
        ])
        phase_y_min, phase_y_max = all_values.min(), all_values.max()
        y_min, y_max = min(y_min, phase_y_min), max(y_max, phase_y_max)

    # Set correct column titles (phases) at the top
    for col_idx, phase in enumerate(sorted_phases):
        axes[0, col_idx].set_title(f"Phase {phase}", fontsize=12, fontweight='bold')

    # Set row labels for data categories on the left-most populated column
    row_titles = [
        "Original Data",
        "After Segregation",
        "Segregation Data",
        "Event Markers"
    ]
    leftmost_col = 0  # Adjust if needed
    for row_idx, title in enumerate(row_titles):
        axes[row_idx, leftmost_col].set_ylabel(title, fontsize=12, fontweight='bold')

    # Plot data for each phase in separate columns
    for col_idx, phase in enumerate(sorted_phases):
        original_values = f'original_{phase}'
        cols_to_plot_after_seg = f'remaining_{phase}'
        cols_to_plot_seg = [f'short_duration_{phase}', f'medium_duration_{phase}', f'long_duration_{phase}']

        # Row 0: Original data
        axes[0, col_idx].plot(
            df['timestamp'],
            df[original_values],
            label='Original' if col_idx == 0 else None,
            color=original_color
        )
        axes[0, col_idx].xaxis.set_major_formatter(time_formatter)
        axes[0, col_idx].tick_params(axis='x', rotation=45)
        axes[0, col_idx].set_ylim(y_min, y_max)

        # Row 1: After segregation
        axes[1, col_idx].plot(
            df['timestamp'],
            df[cols_to_plot_after_seg],
            label='Remaining' if col_idx == 0 else None,
            color=main_color
        )
        axes[1, col_idx].xaxis.set_major_formatter(time_formatter)
        axes[1, col_idx].tick_params(axis='x', rotation=45)
        axes[1, col_idx].set_ylim(y_min, y_max)

        # Row 2: Segregated events
        for col in cols_to_plot_seg:
            if col in df.columns:
                # Determine duration type from column name
                if 'short' in col:
                    color = duration_colors["short"]
                    label = 'Short duration' if col_idx == 0 else None
                elif 'medium' in col:
                    color = duration_colors["medium"]
                    label = 'Medium duration' if col_idx == 0 else None
                else:
                    color = duration_colors["long"]
                    label = 'Long duration' if col_idx == 0 else None
                axes[2, col_idx].plot(df['timestamp'], df[col], label=label, color=color)
        axes[2, col_idx].xaxis.set_major_formatter(time_formatter)
        axes[2, col_idx].tick_params(axis='x', rotation=45)
        axes[2, col_idx].set_ylim(y_min, y_max)

        # Row 3: Event markers - color by matched status
        # Matched = green, Unmatched ON = red, Unmatched OFF = blue
        if additional_points is not None:
            for idx, point in additional_points.iterrows():
                if phase == point['phase']:
                    # Determine color based on matched status
                    if point.get('matched', 0) == 1:
                        event_color = 'green'  # Matched event
                    elif point['event_id'].startswith('on_'):
                        event_color = 'red'    # Unmatched ON
                    else:
                        event_color = 'blue'   # Unmatched OFF

                    start_y = 0
                    end_y = min(abs(point['magnitude']), y_max)
                    axes[3, col_idx].plot(
                        [point['start'], point['start']], [start_y, end_y],
                        color=event_color, linestyle=(0, (int(idx % 5), 3, 3, 3))
                    )
                    axes[3, col_idx].plot(
                        [point['end'], point['end']], [start_y, end_y],
                        color=event_color, linestyle=(0, (int(idx % 5), 3, 3, 3))
                    )
                    axes[3, col_idx].text(
                        point['start'], end_y + (y_max * 0.02), f"{simplify_event_id(point['event_id'])}_S",
                        color=event_color, fontsize=8, rotation=45, ha='right'
                    )
                    axes[3, col_idx].text(
                        point['end'], end_y + (y_max * 0.02), f"{simplify_event_id(point['event_id'])}_E",
                        color=event_color, fontsize=8, rotation=45, ha='left'
                    )
            axes[3, col_idx].xaxis.set_major_formatter(time_formatter)
            axes[3, col_idx].tick_params(axis='x', rotation=45)

    # Create a single legend for the entire figure
    # Collect handles and labels from the first column only (where we set labels)
    handles, labels = [], []
    for row_idx in range(3):  # Rows 0, 1, 2 have the data plots
        ax_handles, ax_labels = axes[row_idx, 0].get_legend_handles_labels()
        for h, l in zip(ax_handles, ax_labels):
            if l not in labels:  # Avoid duplicates
                handles.append(h)
                labels.append(l)

    # Add event legend entries manually
    handles.append(Line2D([0], [0], color='green', linestyle='--', label='Matched event'))
    labels.append('Matched event')
    handles.append(Line2D([0], [0], color='red', linestyle='--', label='Unmatched ON'))
    labels.append('Unmatched ON')
    handles.append(Line2D([0], [0], color='blue', linestyle='--', label='Unmatched OFF'))
    labels.append('Unmatched OFF')

    # Place legend outside the plot area
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.12, 0.98))

    # Set main title
    first_timestamp = df['timestamp'].iloc[0]
    title = create_seg_title(
        first_timestamp=df['timestamp'],
        columns=sorted_phases,
        value="combined",
        house_id=house_id
    )
    fig.suptitle(title, fontsize=16, y=1.02)

    # Save plot
    filename = generate_save_filename_with_title(f'{title}.png', filepath)
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved combined plot: {filename}")
    plt.clf()
    plt.close()

def create_changed_plots_from_path_for_seg_combined(filepath, original_cols, logger, sessions_df):
    logger.info(f"Splitting data into segments for file: {filepath}")
    segments, points_by_segment = split_by_day_night_from_path(filepath, sessions_df)

    logger.info(f"Number of valid segments: {len(segments)}")
    check_or_create_house_directory(filepath)

    for i, (segment, points) in tqdm(enumerate(zip(segments, points_by_segment)), desc=f"Plotting valid segments for {os.path.basename(filepath)}"):
        logger.info(f"Plotting segment {i + 1} out of {len(segments)}")
        try:
            if not points.empty:
                plot_combined_phases_interactive(segment, original_cols, filepath, logger, additional_points=points)
            else:
                logger.info(f"Skipping segment {i + 1} as it has no relevant events")
        except Exception as e:
            logger.error(f"Error while plotting segment {i + 1} in file {filepath}: {e}")

    logger.info(f"### Finished processing file: {filepath} ###")



# Main process
def process_visualization(house_id, run_number, threshold = DEFAULT_THRESHOLD):
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Visualization process for house {house_id}, run {run_number}...")

    seg_seq_directory = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    additional_data = pd.read_csv(
        f"{seg_seq_directory}/on_off_{threshold}.csv"
    )
    additional_data['start'] = pd.to_datetime(additional_data['start'], format='mixed', dayfirst=True)
    additional_data['end'] = pd.to_datetime(additional_data['end'], format='mixed', dayfirst=True)

    file_path = f"{seg_seq_directory}/summarized_{house_id}.csv"
    if not os.path.exists(file_path):
        logger.error(f"Segregation file {file_path} does not exist.")
        return

    try:
        original_cols = ['w1', 'w2', 'w3']
        create_changed_plots_from_path_for_seg_combined(file_path, original_cols, logger, additional_data)
        logger.info(f"Visualization for house {house_id}, run {run_number} completed successfully.")
    except Exception as e:
        logger.error(f"Error during visualization for house {house_id}, run {run_number}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualization.py <house_id> <run_number>")
    else:
        house_id = sys.argv[1]
        run_number = int(sys.argv[2])
        threshold = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_THRESHOLD

        process_visualization(house_id, run_number, threshold)
