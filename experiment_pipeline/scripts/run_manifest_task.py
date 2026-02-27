"""
Run a single (experiment, house) task from a manifest file.

Each task includes: pre-analysis (with normalization) + pipeline + per-house reports.
Used by sbatch_run_all_experiments.sh to dispatch SLURM array tasks.

Manifest CSV format (no header):
    config_name,house_id,output_dir,norm_method

Usage:
    python scripts/run_manifest_task.py \\
        --manifest /path/to/manifest.csv \\
        --task-index 42 \\
        --input-dir /path/to/HouseholdData
"""
import sys
import os
import time
import json
import csv
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = PROJECT_DIR.parent
SRC_DIR = PROJECT_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))


def main():
    parser = argparse.ArgumentParser(description='Run one (experiment, house) task from manifest')
    parser.add_argument('--manifest', required=True, help='Path to manifest CSV')
    parser.add_argument('--task-index', type=int, required=True,
                        help='0-based task index (SLURM_ARRAY_TASK_ID)')
    parser.add_argument('--input-dir', required=True, help='Input data directory')
    parser.add_argument('--reports-base', required=False, default=None,
                        help='Base reports dir (default: {experiment_output}/reports)')
    parser.add_argument('--skip_visualization', action='store_true', default=True)

    args = parser.parse_args()

    # Read manifest line
    with open(args.manifest, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)

    if args.task_index >= len(lines):
        print(f"ERROR: task index {args.task_index} >= manifest size {len(lines)}")
        sys.exit(1)

    config_name, house_id, experiment_output, norm_method = lines[args.task_index]

    # Reports directory: default inside experiment output, or custom base
    if args.reports_base:
        reports_dir = os.path.join(args.reports_base, os.path.basename(experiment_output))
    else:
        reports_dir = os.path.join(experiment_output, 'reports')

    print("=" * 60)
    print(f"Task {args.task_index}: {config_name} / house {house_id}")
    print(f"Output:  {experiment_output}")
    print(f"Reports: {reports_dir}")
    print(f"Norm:    {norm_method}")
    print(f"Start:   {datetime.now()}")
    print("=" * 60)

    os.makedirs(reports_dir, exist_ok=True)

    # ---- Step 1: Pre-analysis with normalization ----
    print(f"\n--- Pre-analysis (normalize={norm_method}) ---")
    pre_start = time.time()

    norm_flag = f"--normalize {norm_method}" if norm_method != 'none' else ""
    pre_cmd = (
        f'"{sys.executable}" scripts/run_analysis.py '
        f'--houses {house_id} '
        f'--input-dir {args.input_dir} '
        f'--output-dir {reports_dir} '
        f'--publish house '
        f'{norm_flag}'
    )
    os.chdir(str(PROJECT_ROOT / 'house_analysis'))
    pre_exit = os.system(pre_cmd)
    print(f"  Pre-analysis: exit {pre_exit >> 8} ({time.time() - pre_start:.0f}s)")

    # ---- Step 2: Pipeline (M1 + M2) ----
    print(f"\n--- Pipeline ({config_name}) ---")
    pipe_start = time.time()
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    os.chdir(str(PROJECT_DIR))
    pipe_cmd = (
        f'"{sys.executable}" -u scripts/test_single_house.py '
        f'--house_id {house_id} '
        f'--experiment_name {config_name} '
        f'--output_path {experiment_output} '
        f'--skip_visualization '
        f'--minimal_output'
    )
    pipe_exit = os.system(pipe_cmd) >> 8  # Get actual exit code

    elapsed = time.time() - pipe_start
    end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    secs = int(elapsed % 60)
    elapsed_human = f"{hours}h {mins}m {secs}s"

    status = "OK" if pipe_exit == 0 else f"FAIL(exit={pipe_exit})"

    # Append to timing CSV (atomic-ish via append mode)
    timing_file = os.path.join(experiment_output, 'house_timing.csv')
    with open(timing_file, 'a') as f:
        f.write(f"{house_id},,{start_time},{end_time},{int(elapsed)},{elapsed_human},{status}\n")

    print(f"\n  Pipeline: {status} ({elapsed_human})")

    # ---- Step 3: Per-house reports (only on success) ----
    if pipe_exit == 0:
        print(f"\n--- Per-house reports ---")
        report_start = time.time()

        os.chdir(str(PROJECT_ROOT / 'disaggregation_analysis'))
        os.system(
            f'"{sys.executable}" scripts/run_dynamic_report.py '
            f'--experiment {experiment_output} '
            f'--houses {house_id} '
            f'--output-dir {reports_dir} '
            f'--publish segregation'
        )

        os.chdir(str(PROJECT_ROOT / 'identification_analysis'))
        os.system(
            f'"{sys.executable}" scripts/run_identification_report.py '
            f'--experiment {experiment_output} '
            f'--houses {house_id} '
            f'--output-dir {reports_dir} '
            f'--publish identification'
        )

        print(f"  Reports generated in {time.time() - report_start:.0f}s")

        # ---- Step 4: Aggressive cleanup (pkl files no longer needed) ----
        print(f"\n--- Cleanup ---")
        try:
            os.chdir(str(PROJECT_DIR))
            sys.path.insert(0, str(SRC_DIR))
            from identification.cleanup import cleanup_after_reports
            r = cleanup_after_reports(Path(experiment_output), house_id)
            print(f"  Cleanup: {r['dirs_deleted']} directories removed")
        except Exception as e:
            print(f"  Cleanup warning: {e}")

    total = time.time() - pre_start
    print(f"\nTotal task time: {total:.0f}s")
    print(f"End: {datetime.now()}")
    sys.exit(pipe_exit)


if __name__ == '__main__':
    main()
