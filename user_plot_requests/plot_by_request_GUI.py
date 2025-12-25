"""
GUI tool for generating interactive plots from segmentation results.

NOTE: This script uses server paths by default.
For local use, update the base_path in load_data() and get_available_houses()
to point to your local OUTPUT directory.
"""
import pandas as pd
from datetime import datetime, timedelta
import os
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkcalendar import Calendar


# Load data from file
def load_data(house_id, run_number):
    # Server path - update this for local use
    # Example: base_path = f"./experiment_pipeline/OUTPUT/run_{run_number}/house_{house_id}"
    base_path = f"/sise/shanigu-group/hilakese-dorins/SequenceData/new_pipeline/run_{run_number}/house_{house_id}"
    file_path = os.path.join(base_path, f"summarized_{house_id}.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', dayfirst=True)
    return df


# Filter data to 12-hour window around target date
def filter_data_by_date(df, target_date):
    target_datetime = datetime.strptime(target_date, "%Y-%m-%d %H:%M:%S")
    start_time = target_datetime - timedelta(hours=6)
    end_time = target_datetime + timedelta(hours=6)

    filtered_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

    if filtered_df.empty:
        raise ValueError(f"No data found for the 12-hour window around {target_date}")

    return filtered_df


# Create interactive plot
def plot_combined_phases_interactive(df, phases, output_path):
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
        hovermode="x unified",
        showlegend=True
    )

    fig.write_html(output_path)
    print(f"Interactive plot saved to {output_path}")


# Process and generate plot
def process_plot(house_id, run_number, target_datetime):
    try:
        # Load data
        df = load_data(house_id, run_number)

        # Filter by date
        filtered_df = filter_data_by_date(df, target_datetime)

        # Create plot
        output_path = f"./plots/house_{house_id}_run_{run_number}_{target_datetime.replace(':', '-')}.html"
        phases = ['w1', 'w2', 'w3']
        plot_combined_phases_interactive(filtered_df, phases, output_path)
        messagebox.showinfo("Success", f"Plot saved to: {output_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Find available house numbers in directory
def get_available_houses(run_number):
    # Server path - update this for local use
    # Example: base_path = f"./experiment_pipeline/OUTPUT/run_{run_number}"
    base_path = f"/sise/shanigu-group/hilakese-dorins/SequenceData/new_pipeline/run_{run_number}"
    house_files = [
        f.split("_")[1].split(".")[0]  # Extract house number from filename
        for f in os.listdir(base_path)
        if f.startswith("summarized_") and f.endswith(".csv")
    ]
    return sorted(house_files)


# Create GUI
def create_gui():
    def on_submit():
        house_id = house_id_combobox.get()
        run_number = run_number_entry.get()
        target_date = cal.get_date()
        target_time = time_entry.get()

        if not house_id or not run_number or not target_time:
            messagebox.showerror("Error", "Please fill all fields.")
            return

        target_datetime = f"{target_date} {target_time}:00"

        try:
            process_plot(house_id, int(run_number), target_datetime)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_load():
        run_number = run_number_entry.get()

        if not run_number:
            messagebox.showerror("Error", "Please provide a run number.")
            return

        try:
            houses = get_available_houses(int(run_number))
            house_id_combobox['values'] = houses
            house_id_combobox.set(houses[0] if houses else "")
            messagebox.showinfo("Success", "House list loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    root = tk.Tk()
    root.title("Interactive Plot Generator")

    tk.Label(root, text="Run Number:").grid(row=0, column=0)
    run_number_entry = ttk.Entry(root)
    run_number_entry.grid(row=0, column=1)

    tk.Label(root, text="House ID:").grid(row=1, column=0)
    house_id_combobox = ttk.Combobox(root)
    house_id_combobox.grid(row=1, column=1)

    tk.Label(root, text="Date:").grid(row=2, column=0)
    cal = Calendar(root, date_pattern="yyyy-mm-dd")
    cal.grid(row=2, column=1)

    tk.Label(root, text="Time (HH:MM):").grid(row=3, column=0)
    time_entry = ttk.Entry(root)
    time_entry.grid(row=3, column=1)

    ttk.Button(root, text="Load House List", command=on_load).grid(row=4, column=0, columnspan=2, pady=5)
    ttk.Button(root, text="Generate Plot", command=on_submit).grid(row=5, column=0, columnspan=2, pady=5)

    root.mainloop()


if __name__ == "__main__":
    os.makedirs("./plots", exist_ok=True)
    create_gui()
