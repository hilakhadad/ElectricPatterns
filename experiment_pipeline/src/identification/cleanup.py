"""
Post-pipeline cleanup for minimal output mode.

Removes intermediate pkl files after unified JSON is built,
keeping only run_0 and last-run summarized data.
"""
import shutil
from pathlib import Path


def cleanup_intermediate_files(
    experiment_dir: Path,
    house_id: str,
    iterations_completed: int,
    logger,
) -> None:
    """
    Delete intermediate pkl files after unified JSON is built.

    Deletes:
    - on_off/, matches/, unmatched_on/, unmatched_off/ (all runs)
    - summarized/ for intermediate runs (keep run_0 + last run)

    Keeps:
    - summarized/ for run_0 (baseline for evaluation) and last run (final remaining)
    - evaluation CSV files, logs, device_sessions JSON, device_activations JSON
    """
    logger.info("Starting cleanup of intermediate files...")

    for run_number in range(iterations_completed):
        run_dir = experiment_dir / f"run_{run_number}"
        if not run_dir.exists():
            continue

        house_dir = run_dir / f"house_{house_id}"
        if not house_dir.exists():
            continue

        # Always delete event-level pkl directories
        for subdir in ['on_off', 'matches', 'unmatched_on', 'unmatched_off']:
            dir_path = house_dir / subdir
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"  Deleted {dir_path.relative_to(experiment_dir)}")

        # Delete intermediate summarized dirs (keep run_0 and last run)
        if 0 < run_number < iterations_completed - 1:
            summarized_dir = house_dir / "summarized"
            if summarized_dir.exists():
                shutil.rmtree(summarized_dir)
                logger.info(f"  Deleted intermediate {summarized_dir.relative_to(experiment_dir)}")

    logger.info("Cleanup completed")
