import pandas as pd
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from data_util import *

PIPE_LINE_DIR = "/sise/shanigu-group/hilakese-dorins/SequenceData/with_marks_pipeline"

def get_available_houses(run_number):
    base_path = f"{PIPE_LINE_DIR}/run_{run_number}"
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Run directory not found: {base_path}")

    house_files = [
        f.split("_")[1].split(".")[0]
        for f in os.listdir(base_path)
        if f.startswith("house_")
    ]
    return sorted(house_files)


def load_data(house_id, run_number):
    base_path = f"{PIPE_LINE_DIR}/run_{run_number}/house_{house_id}"
    file_path = os.path.join(base_path, f"summarized_{house_id}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
    return df


def filter_data_by_date(df, target_date, hour):
    target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
    target_datetime = target_datetime.replace(hour=int(hour))
    start_time = target_datetime - timedelta(hours=6)
    end_time = target_datetime + timedelta(hours=6)

    filtered_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

    if filtered_df.empty:
        raise ValueError(f"No data found for the 12-hour window around {target_datetime}")

    return filtered_df


def calculate_y_axis_range(df, phases):
    y_min = float('inf')
    y_max = float('-inf')

    for phase in phases:
        columns = [f'original_{phase}', f'remaining_{phase}', f'short_duration_{phase}', f'medium_duration_{phase}',
                   f'long_duration_{phase}']
        for col in columns:
            if col in df.columns:
                y_min = min(y_min, df[col].min())
                y_max = max(y_max, df[col].max())

    return y_min, y_max


def plot_combined_phases_interactive(df, phases, output_path, y_range):
    fig = make_subplots(
        rows=4, cols=len(phases),
        shared_xaxes=True, shared_yaxes=True,
        subplot_titles=[f"Phase {phase}" for phase in phases for _ in range(4)]
    )

    phase_colors = {"w1": "blue", "w2": "green", "w3": "orange"}

    for col_idx, phase in enumerate(phases, start=1):
        original_values = f'original_{phase}'
        cols_to_plot_after_seg = f'remaining_{phase}'
        cols_to_plot_seg = [f'short_duration_{phase}', f'medium_duration_{phase}', f'long_duration_{phase}']

        # Row 1: Original Data
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df[original_values],
                mode='lines', name=f'Original {phase}', line=dict(color=phase_colors.get(phase, 'black'))
            ), row=1, col=col_idx
        )

        # Row 2: After Segregation
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df[cols_to_plot_after_seg],
                mode='lines', name=f'Remaining {phase}', line=dict(color=phase_colors.get(phase, 'black'))
            ), row=2, col=col_idx
        )

        # Row 3: Segregation Data
        for col in cols_to_plot_seg:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'], y=df[col],
                        mode='lines', name=col
                    ), row=3, col=col_idx
                )

    fig.update_layout(
        title="12-Hour Window Plot",
        xaxis_title="Timestamp",
        yaxis_title="Values",
        yaxis_range=y_range,
        hovermode="x unified",
        showlegend=True
    )

    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")
    fig.show()


def main():
    while True:
        print("Welcome to the CLI-based Plot Generator")
        run_number = input("Enter Run Number: ")

        try:
            houses = get_available_houses(run_number)
        except FileNotFoundError as e:
            print(e)
            return

        if not houses:
            print(f"No houses found for Run {run_number}")
            return

        print(f"Available houses: {', '.join(houses)}")
        house_id = input("Enter House ID from the available list: ")

        if house_id not in houses:
            print(f"House ID {house_id} is not available. Please select from the available list.")
            return

        try:
            df = load_data(house_id, run_number)
        except FileNotFoundError as e:
            print(e)
            return

        print(f"Data loaded for House {house_id}, Run {run_number}.")
        print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

        target_date = input("Enter target date (YYYY-MM-DD): ")
        target_hour = input("Enter hour (0-23): ")

        try:
            filtered_df = filter_data_by_date(df, target_date, target_hour)
        except ValueError as e:
            print(e)
            return

        phases = ['w1', 'w2', 'w3']
        y_range = calculate_y_axis_range(filtered_df, phases)

        output_path = f"{PLOT_DIR}/house_{house_id}_run_{run_number}_{target_date}_{target_hour}.html"

        try:
            plot_combined_phases_interactive(filtered_df, phases, output_path, y_range)
        except Exception as e:
            print(f"Error creating plot: {e}")


if __name__ == "__main__":
    os.makedirs(f"{PLOT_DIR}", exist_ok=True)
    main()
