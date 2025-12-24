from tqdm import tqdm
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
matplotlib.use('Agg')  # Set the non-GUI backend for headless environments
from data_util import *

import sys
import os


def create_seg_title(first_timestamp, columns, value, house_id):
    house_id = house_id.split('_')[1] if "summarized" in house_id else house_id
    date = first_timestamp.date.iloc[0]
    first_time = first_timestamp.time.iloc[0].strftime("%H_%M")
    columns = '_'.join([col for col in columns if col != 'timestamp'])
    title = f'{house_id}_{date}_{first_time}_{columns}_{value}'
    return title


def split_by_day_night_from_path(filepath, chunksize=10000):
    # Get total number of rows in the CSV file
    total_rows = sum(1 for _ in open(filepath)) - 1  # Subtract 1 for the header
    total_chunks = (total_rows // chunksize) + (1 if total_rows % chunksize > 0 else 0)

    segments = []
    segment_start = None

    # Read the CSV in chunks for efficiency
    for chunk in tqdm(pd.read_csv(filepath, chunksize=chunksize, dayfirst=True), total=total_chunks):
        # Ensure the timestamp is in datetime format
        chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], format='mixed', dayfirst=True)
        chunk = chunk.sort_values('timestamp').reset_index(drop=True)

        # Set the start time for day or night segment based on the first timestamp
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

        # Process each 12-hour day or night segment without iterating row-by-row
        while segment_start <= chunk['timestamp'].max():
            segment_end = segment_start + timedelta(hours=12)
            segment = chunk[(chunk['timestamp'] >= segment_start) & (chunk['timestamp'] < segment_end)]

            if not segment.empty:
                segments.append(segment)

            # Move to the next 12-hour segment
            segment_start = segment_end

    return segments


def check_or_create_house_directory(path):
    house_id = os.path.splitext(os.path.basename(path))[0]
    plot_dir_path = os.path.join(os.path.dirname(os.path.dirname(path)), "plots")
    house_plot_dir_path = os.path.join(plot_dir_path, house_id)
    os.makedirs(house_plot_dir_path, exist_ok=True)
    return house_plot_dir_path


def generate_save_filename_with_title(title, filepath):
    dir_of_plots = check_or_create_house_directory(filepath)
    return os.path.join(dir_of_plots, title)


def plot_combined_phases(df, phases, filepath):
    """
    Creates a single plot with three subplots (side by side): Original Data, After Segregation Data, and Segregation Data
    for all given phases from a single DataFrame. Ensures the same Y-axis scale for all plots.
    """
    house_id = os.path.splitext(os.path.basename(filepath))[0]

    phase_colors = {
        "w1": "blue",
        "w2": "green",
        "w3": "orange"
    }

    fig, axes = plt.subplots(nrows=3, ncols=len(phases), figsize=(18, 12), sharex=True, sharey=True,
                             constrained_layout=True)
    time_formatter = mdates.DateFormatter('%H:%M')

    # Ensure 'timestamp' is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)

    # Calculate global Y-axis limits across all phases and columns
    y_min, y_max = float('inf'), float('-inf')
    for phase in phases:
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

    # Iterate over phases to create subplots
    for col_idx, phase in enumerate(phases):
        # Column definitions
        original_values = f'original_{phase}'
        cols_to_plot_after_seg = f'remaining_{phase}'
        cols_to_plot_seg = [f'short_duration_{phase}', f'medium_duration_{phase}', f'long_duration_{phase}']

        # Plot Original Data
        axes[0, col_idx].plot(
            df['timestamp'],
            df[original_values],
            label=original_values,
            color=phase_colors.get(phase, "black")  # Default to black if phase not in colors
        )
        axes[0, col_idx].set_title(f"Original Data (Phase {phase})")
        axes[0, col_idx].legend()
        axes[0, col_idx].xaxis.set_major_formatter(time_formatter)
        axes[0, col_idx].tick_params(axis='x', rotation=45)
        axes[0, col_idx].set_ylim(y_min, y_max)

        # Plot After Segregation Data
        axes[1, col_idx].plot(
            df['timestamp'],
            df[cols_to_plot_after_seg],
            label=cols_to_plot_after_seg,
            color=phase_colors.get(phase, "black")  # Default to black if phase not in colors
        )
        axes[1, col_idx].set_title(f"After Segregation Data (Phase {phase})")
        axes[1, col_idx].legend()
        axes[1, col_idx].xaxis.set_major_formatter(time_formatter)
        axes[1, col_idx].tick_params(axis='x', rotation=45)
        axes[1, col_idx].set_ylim(y_min, y_max)

        # Plot Segregation Data
        for col in cols_to_plot_seg:
            if col in df.columns:
                axes[2, col_idx].plot(df['timestamp'], df[col], label=col)
        axes[2, col_idx].set_title(f"Segregation Data (Phase {phase})")
        axes[2, col_idx].legend()
        axes[2, col_idx].xaxis.set_major_formatter(time_formatter)
        axes[2, col_idx].tick_params(axis='x', rotation=45)
        axes[2, col_idx].set_ylim(y_min, y_max)

    # Generate meaningful title for the file
    first_timestamp = df['timestamp'].iloc[0]
    title = create_seg_title(
        first_timestamp=df['timestamp'].dt,
        columns=phases,
        value="combined",
        house_id=house_id
    )

    # Add a global title for the figure
    fig.suptitle(title, fontsize=16, y=1.02)

    filename = generate_save_filename_with_title(f'{title}.png', filepath)
    plt.savefig(filename, bbox_inches="tight")
    print(f"Saved combined plot: {filename}")
    plt.clf()
    plt.close()


def create_plots_from_path_for_seg_combined(filepath, original_cols, logger):
    """
    Creates plots for segmented combined data from a given file path.
    """
    logger.info(f"Splitting data into segments for file: {filepath}")
    segments = split_by_day_night_from_path(filepath)

    logger.info(f"Number of segments found: {len(segments)}")
    check_or_create_house_directory(filepath)

    for i, segment in tqdm(enumerate(segments), desc=f"Plotting segments for {os.path.basename(filepath)}"):
        logger.info(f"Plotting segment {i + 1} out of {len(segments)}")
        try:
            plot_combined_phases(segment, original_cols, filepath)
        except Exception as e:
            logger.error(f"Error while plotting segment {i + 1} in file {filepath}: {e}")

    logger.info(f"### Finished processing file: {filepath} ###")


def filter_segments_with_changes(segments, columns_to_check):
    """
    Filters segments where specified columns contain values different from 0.

    Parameters:
        segments (list): A list of DataFrames, each representing a segment.
        columns_to_check (list): List of column names to check for non-zero values.

    Returns:
        list: Segments where at least one of the specified columns contains non-zero values.
    """
    filtered_segments = []

    for segment in segments:
        # Check if any of the specified columns contain non-zero values
        if not segment[columns_to_check].eq(0).all().all():
            filtered_segments.append(segment)

    return filtered_segments


def create_changed_plots_from_path_for_seg_combined(filepath, original_cols, logger):
    """
    Creates plots for segmented combined data from a given file path.
    """
    logger.info(f"Splitting data into segments for file: {filepath}")
    segments = split_by_day_night_from_path(filepath)

    # columns_to_check = [
    #     "short_duration_w1", "medium_duration_w1", "long_duration_w1",
    #     "short_duration_w2", "medium_duration_w2", "long_duration_w2",
    #     "short_duration_w3", "medium_duration_w3", "long_duration_w3"
    # ]
    columns_to_check = [
        "long_duration_w1",
        "long_duration_w2",
        "long_duration_w3"
    ]
    filtered_segments = filter_segments_with_changes(segments, columns_to_check)

    logger.info(f"Number of segments with changes: {len(filtered_segments)}")

    check_or_create_house_directory(filepath)

    for i, segment in tqdm(enumerate(filtered_segments), desc=f"Plotting segments for {os.path.basename(filepath)}"):
        logger.info(f"Plotting segment {i + 1} out of {len(filtered_segments)}")
        try:
            plot_combined_phases(segment, original_cols, filepath)
        except Exception as e:
            logger.error(f"Error while plotting segment {i + 1} in file {filepath}: {e}")

    logger.info(f"### Finished processing file: {filepath} ###")

def check_or_create_plots_directory(path):
    plot_dir_path = os.path.join(os.path.dirname(path), "plots")
    os.makedirs(plot_dir_path, exist_ok=True)
    return plot_dir_path


def process_visualization(house_id, run_number):
    logger = setup_logging(house_id, run_number, LOGS_DIRECTORY)
    logger.info(f"Processing visualization for house {house_id}, run {run_number}...")

    expected_env = "/sise/home/hilakese/.conda/envs/nilm_stat_env/bin/python"
    if sys.executable != expected_env:
        raise EnvironmentError(f"Expected Python executable: {expected_env}, but got: {sys.executable}")

    seg_seq_directory = f"{OUTPUT_BASE_PATH}/run_{run_number}/house_{house_id}"
    file_path = f"{seg_seq_directory}/summarized_{house_id}.csv"
    if not os.path.exists(file_path):
        logger.error(f"Segregation file {file_path} does not exist.")
        return

    try:
        original_cols = ['w1', 'w2', 'w3']
        # if run_number == 0:
        #     create_plots_from_path_for_seg_combined(file_path, original_cols)
        # else:
        create_changed_plots_from_path_for_seg_combined(file_path, original_cols, logger)
        logger.info(f"Visualization for house {house_id}, run {run_number} completed successfully.")
    except Exception as e:
        logger.error(f"Error during visualization for house {house_id}, run {run_number}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python visualization.py <house_id> <run_number>")
    else:
        house_id = sys.argv[1]
        run_number = int(sys.argv[2])
        process_visualization(house_id, run_number)

