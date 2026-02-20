"""
Post-pipeline cleanup for minimal output mode.

Removes only unmatched_on/ and unmatched_off/ pkl directories
(derivable from on_off minus matches).

Preserves on_off/, matches/, and summarized/ for ALL iterations
to ensure full visibility into every pipeline step.
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
    Delete derivable intermediate pkl files after unified JSON is built.

    Deletes:
    - unmatched_on/, unmatched_off/ (all runs) — derivable from on_off minus matches

    Keeps:
    - on_off/ (detected events per iteration)
    - matches/ (matched pairs per iteration)
    - summarized/ (segmentation results per iteration)
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

        # Only delete derivable unmatched directories
        for subdir in ['unmatched_on', 'unmatched_off']:
            dir_path = house_dir / subdir
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"  Deleted {dir_path.relative_to(experiment_dir)}")

    logger.info("Cleanup completed")
