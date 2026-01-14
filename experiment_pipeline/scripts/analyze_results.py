"""
Analyze results from an experiment across multiple houses.
Creates summary reports showing cumulative metrics for all houses.
"""
import sys
import os
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set this to the experiment folder you want to analyze
EXPERIMENT_PATH = None  # Will be set via command line or manually below

# Example:
# EXPERIMENT_PATH = "c:/Users/hilak/PycharmProjects/role_based_segregation_dev/experiment_pipeline/OUTPUT/experiments/exp003_progressive_search_20260114_123106"

# ============================================================================


def find_latest_evaluation(house_path):
    """Find the latest evaluation_history file for a house"""
    house_id = os.path.basename(house_path).replace("house_", "")

    # Find all run folders, sorted by number
    all_items = os.listdir(house_path)
    run_folders = sorted([f for f in all_items if f.startswith("run_")],
                        key=lambda x: int(x.replace("run_", "")))

    if not run_folders:
        return None, None

    # Search from latest run backwards to find one with evaluation
    for latest_run in reversed(run_folders):
        latest_run_num = int(latest_run.replace("run_", ""))

        # Find evaluation file
        eval_path = os.path.join(house_path, latest_run, f"house_{house_id}", f"evaluation_history_{house_id}.csv")

        if os.path.exists(eval_path):
            return eval_path, latest_run_num

    return None, None


def analyze_experiment(experiment_path):
    """Analyze all houses in an experiment and create summary reports"""

    print(f"\n{'='*60}")
    print(f"Analyzing experiment: {os.path.basename(experiment_path)}")
    print(f"{'='*60}\n")

    # Find all house folders
    house_folders = [f for f in os.listdir(experiment_path)
                     if f.startswith("house_") and os.path.isdir(os.path.join(experiment_path, f))]

    if not house_folders:
        print("No house folders found!")
        return

    print(f"Found {len(house_folders)} houses: {', '.join(house_folders)}\n")

    # Collect all evaluation data
    all_detailed = []
    all_summary = []

    for house_folder in sorted(house_folders):
        house_path = os.path.join(experiment_path, house_folder)
        eval_path, latest_run = find_latest_evaluation(house_path)

        if eval_path is None:
            print(f"  {house_folder}: No evaluation data found")
            continue

        # Read evaluation history
        df = pd.read_csv(eval_path)
        all_detailed.append(df)

        # Get the final cumulative row for each phase (latest run)
        final_rows = df[df['run_number'] == df['run_number'].max()]
        all_summary.append(final_rows)

        # Print summary for this house
        house_id = house_folder.replace("house_", "")
        print(f"  {house_folder} (run {latest_run}):")
        for _, row in final_rows.iterrows():
            phase = row['phase']
            power_pct = row['explained_power_cumulative_pct']
            time_pct = row['minutes_explained_cumulative_pct']
            neg_min = row.get('minutes_negative', 0)
            print(f"    {phase}: Power {power_pct:.1f}%, Time {time_pct:.1f}%", end="")
            if neg_min > 0:
                print(f" [WARNING: {neg_min} negative minutes]", end="")
            print()

    if not all_detailed:
        print("\nNo evaluation data found for any house!")
        return

    # Create detailed report (all iterations, all houses)
    detailed_df = pd.concat(all_detailed, ignore_index=True)
    detailed_path = os.path.join(experiment_path, "detailed_report.csv")
    detailed_df.to_csv(detailed_path, index=False)
    print(f"\nDetailed report saved to: {detailed_path}")

    # Create summary report (final cumulative only)
    summary_df = pd.concat(all_summary, ignore_index=True)

    # Add totals row for each phase
    totals = []
    for phase in ['w1', 'w2', 'w3']:
        phase_data = summary_df[summary_df['phase'] == phase]
        if len(phase_data) == 0:
            continue

        total_row = {
            'house_id': 'TOTAL',
            'run_number': '-',
            'threshold': phase_data['threshold'].iloc[0],
            'phase': phase,
            'total_power_all': phase_data['total_power_all'].sum(),
            'total_power_above_th': phase_data['total_power_above_th'].sum(),
            'explained_power': phase_data['explained_power'].sum(),
            'explained_power_pct': '-',
            'explained_power_cumulative': phase_data['explained_power_cumulative'].sum(),
            'explained_power_cumulative_pct': round(
                phase_data['explained_power_cumulative'].sum() / phase_data['total_power_above_th'].sum() * 100, 2
            ) if phase_data['total_power_above_th'].sum() > 0 else 0,
            'minutes_above_th': phase_data['minutes_above_th'].sum(),
            'minutes_explained': phase_data['minutes_explained'].sum(),
            'minutes_explained_pct': '-',
            'minutes_explained_cumulative': phase_data['minutes_explained_cumulative'].sum(),
            'minutes_explained_cumulative_pct': round(
                phase_data['minutes_explained_cumulative'].sum() / phase_data['minutes_above_th'].sum() * 100, 2
            ) if phase_data['minutes_above_th'].sum() > 0 else 0,
            'minutes_negative': phase_data['minutes_negative'].sum() if 'minutes_negative' in phase_data.columns else 0,
            'power_negative': phase_data['power_negative'].sum() if 'power_negative' in phase_data.columns else 0,
            'minutes_missing': phase_data['minutes_missing'].sum() if 'minutes_missing' in phase_data.columns else 0,
        }
        totals.append(total_row)

    summary_with_totals = pd.concat([summary_df, pd.DataFrame(totals)], ignore_index=True)
    summary_path = os.path.join(experiment_path, "summary_report.csv")
    summary_with_totals.to_csv(summary_path, index=False)
    print(f"Summary report saved to: {summary_path}")

    # Print overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")

    for total in totals:
        phase = total['phase']
        power_pct = total['explained_power_cumulative_pct']
        time_pct = total['minutes_explained_cumulative_pct']
        neg_min = total['minutes_negative']

        print(f"\n{phase}:")
        print(f"  Power explained: {power_pct}%")
        print(f"  Time explained: {time_pct}%")
        if neg_min > 0:
            print(f"  WARNING: {neg_min} total negative minutes across all houses!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_path = sys.argv[1]
    elif EXPERIMENT_PATH:
        exp_path = EXPERIMENT_PATH
    else:
        # Default to latest experiment
        exp_base = Path(__file__).parent.parent / "OUTPUT" / "experiments"
        experiments = sorted([f for f in os.listdir(exp_base) if os.path.isdir(os.path.join(exp_base, f))])
        if experiments:
            exp_path = os.path.join(exp_base, experiments[-1])
            print(f"Using latest experiment: {experiments[-1]}")
        else:
            print("No experiments found. Please specify an experiment path.")
            sys.exit(1)

    if not os.path.exists(exp_path):
        print(f"Error: Experiment path not found: {exp_path}")
        sys.exit(1)

    analyze_experiment(exp_path)
