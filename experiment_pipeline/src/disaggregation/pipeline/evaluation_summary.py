"""
Dynamic evaluation summary generation.

Provides threshold-independent evaluation metrics using the original power
as a fixed denominator, making cumulative percentages comparable across iterations.
"""
import pandas as pd
from pathlib import Path


def generate_dynamic_evaluation_summary(
    output_path: str,
    house_id: str,
    threshold_schedule: list,
    iterations_completed: int,
    logger,
) -> None:
    """
    Generate threshold-independent evaluation summary for dynamic threshold runs.

    The standard evaluation_history CSV uses `total_power_above_th` as denominator,
    which changes with each threshold — making cumulative percentages misleading.

    This summary uses total original power (fixed) as denominator, so cumulative
    percentages grow monotonically and are directly comparable across iterations
    and with other experiments.

    Saves: evaluation_summaries/dynamic_evaluation_summary_{house_id}.csv
    """
    output_dir = Path(output_path)

    # Load run_0 summarized data (has original power)
    run0_summarized = output_dir / "run_0" / f"house_{house_id}" / "summarized"
    if not run0_summarized.is_dir():
        logger.warning("Cannot generate dynamic evaluation summary: run_0 summarized not found")
        return

    run0_files = sorted(run0_summarized.glob(f"summarized_{house_id}_*.pkl"))
    if not run0_files:
        logger.warning("Cannot generate dynamic evaluation summary: no run_0 summarized files")
        return

    baseline = pd.concat([pd.read_pickle(f) for f in run0_files], ignore_index=True)
    baseline['timestamp'] = pd.to_datetime(baseline['timestamp'])
    baseline = baseline.drop_duplicates(subset=['timestamp'], keep='first')

    # Calculate total original power per phase (fixed denominator)
    original_power = {}
    for phase in ['w1', 'w2', 'w3']:
        col = f'original_{phase}'
        if col in baseline.columns:
            original_power[phase] = baseline[col].clip(lower=0).sum()
        else:
            original_power[phase] = 0

    rows = []
    prev_remaining = {phase: original_power[phase] for phase in ['w1', 'w2', 'w3']}

    for run_number in range(iterations_completed):
        threshold = threshold_schedule[run_number] if run_number < len(threshold_schedule) else 0

        # Load this run's summarized data
        run_summarized = output_dir / f"run_{run_number}" / f"house_{house_id}" / "summarized"
        if not run_summarized.is_dir():
            break

        run_files = sorted(run_summarized.glob(f"summarized_{house_id}_*.pkl"))
        if not run_files:
            break

        run_data = pd.concat([pd.read_pickle(f) for f in run_files], ignore_index=True)
        run_data['timestamp'] = pd.to_datetime(run_data['timestamp'])
        run_data = run_data.drop_duplicates(subset=['timestamp'], keep='first')

        for phase in ['w1', 'w2', 'w3']:
            remaining_col = f'remaining_{phase}'
            if remaining_col not in run_data.columns:
                continue

            current_remaining = run_data[remaining_col].clip(lower=0).sum()
            orig = original_power[phase]

            # Power explained by THIS iteration
            iteration_explained = max(0, prev_remaining[phase] - current_remaining)
            # Cumulative explained from original
            cumulative_explained = max(0, orig - current_remaining)

            rows.append({
                'run_number': run_number,
                'threshold': threshold,
                'phase': phase,
                'original_power': round(orig, 1),
                'remaining_power': round(current_remaining, 1),
                'iteration_explained': round(iteration_explained, 1),
                'iteration_explained_pct': round(iteration_explained / orig * 100, 2) if orig > 0 else 0,
                'cumulative_explained': round(cumulative_explained, 1),
                'cumulative_explained_pct': round(cumulative_explained / orig * 100, 2) if orig > 0 else 0,
            })

            prev_remaining[phase] = current_remaining

    if not rows:
        logger.warning("No data for dynamic evaluation summary")
        return

    summary_df = pd.DataFrame(rows)
    summaries_dir = output_dir / "evaluation_summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summaries_dir / f"dynamic_evaluation_summary_{house_id}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Dynamic evaluation summary saved to {summary_path}")

    # Log final cumulative results
    last_run = summary_df[summary_df['run_number'] == summary_df['run_number'].max()]
    for _, row in last_run.iterrows():
        logger.info(
            f"  {row['phase']}: {row['cumulative_explained_pct']:.1f}% of total power explained "
            f"({row['cumulative_explained']:.0f}W / {row['original_power']:.0f}W)"
        )
