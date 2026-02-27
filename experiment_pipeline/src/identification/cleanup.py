"""
Post-pipeline cleanup utilities.

Two levels:
  1. cleanup_intermediate_files() — conservative, runs inside pipeline.
     Deletes only unmatched_on/off (derivable from on_off - matches).

  2. cleanup_after_reports() — aggressive, runs AFTER per-house reports.
     Deletes ALL pkl subdirectories (on_off, matches, summarized, unmatched).
     Safe because reports have already been generated as HTML.
     Keeps: evaluation CSVs, logs, device_sessions JSON, experiment_metadata.
"""
import shutil
from pathlib import Path


# pkl subdirectories created by the pipeline per run_N/house_{id}/
_PKL_SUBDIRS = ['on_off', 'matches', 'summarized', 'unmatched_on', 'unmatched_off']


def cleanup_intermediate_files(
    experiment_dir: Path,
    house_id: str,
    iterations_completed: int,
    logger,
) -> None:
    """
    Conservative cleanup: delete only derivable pkl files.

    Runs inside the pipeline (after M2, before reports).

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

        for subdir in ['unmatched_on', 'unmatched_off']:
            dir_path = house_dir / subdir
            if dir_path.exists():
                shutil.rmtree(dir_path)
                logger.info(f"  Deleted {dir_path.relative_to(experiment_dir)}")

    logger.info("Cleanup completed")


def cleanup_after_reports(
    experiment_dir: Path,
    house_id: str,
) -> dict:
    """
    Aggressive cleanup: delete ALL pkl subdirectories for a house.

    Call AFTER per-house reports (segregation + identification) have been
    generated, so the HTML reports are already on disk. The aggregate
    reports only combine per-house HTMLs and read JSONs/CSVs — they
    do NOT re-read pkl files.

    Deletes from ALL run_*/house_{id}/ and run_post/house_{id}/:
    - on_off/, matches/, summarized/, unmatched_on/, unmatched_off/

    Preserves (at experiment level, outside run_*/):
    - device_sessions/*.json
    - device_activations/*.json
    - evaluation_summaries/*.csv
    - house_timing.csv
    - experiment_metadata.json
    - reports/ (HTML)
    - logs/

    Returns:
        dict with 'dirs_deleted' count and 'bytes_freed' estimate
    """
    experiment_dir = Path(experiment_dir)
    dirs_deleted = 0

    # Clean run_0 through run_N
    for run_dir in sorted(experiment_dir.glob("run_*")):
        if not run_dir.is_dir():
            continue
        house_dir = run_dir / f"house_{house_id}"
        if not house_dir.is_dir():
            continue

        for subdir_name in _PKL_SUBDIRS:
            dir_path = house_dir / subdir_name
            if dir_path.exists():
                shutil.rmtree(dir_path)
                dirs_deleted += 1

    return {'dirs_deleted': dirs_deleted}
