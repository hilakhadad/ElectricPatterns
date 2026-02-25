"""
Local batch runner -replicates sbatch_run_houses.sh for local execution.

Runs the full pipeline + reports for a list of houses, sequentially:
  Phase 1 — House pre-analysis (uses raw data, no pipeline needed):
    1a. Per-house house pre-analysis
    1b. Aggregate house pre-analysis report
  Phase 2 — Pipeline + per-house reports:
    2a. Pipeline (M1 + M2) for each house
    2b. Per-house segregation report (after each successful house)
    2c. Per-house identification report (after each successful house)
  Phase 3 — Aggregate pipeline reports:
    3a. Aggregate segregation report
    3b. Aggregate identification report

Output structure (same as server):
  {experiment}/reports/
    ├── house_report.html           -House pre-analysis aggregate
    ├── house_reports/              -House pre-analysis per-house
    ├── segregation_report.html     -M1 disaggregation aggregate
    ├── segregation_reports/        -M1 per-house
    ├── identification_report.html  -M2 identification aggregate
    └── identification_reports/     -M2 per-house

Usage:
    python scripts/run_local_batch.py
    python scripts/run_local_batch.py --houses 5080,344,3021
    python scripts/run_local_batch.py --shortest 10
    python scripts/run_local_batch.py --experiment_name exp015_hole_repair
"""
import sys
import os
import time
import subprocess
import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Fix encoding for Windows console
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
EXPERIMENT_PIPELINE = PROJECT_ROOT / "experiment_pipeline"
DATA_DIR = PROJECT_ROOT / "INPUT" / "HouseholdData"

PYTHON = sys.executable


def get_houses_sorted_by_months():
    """Get house IDs sorted by number of monthly pkl files (fewest first)."""
    houses = []
    for folder in DATA_DIR.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            pkl_files = list(folder.glob("*.pkl"))
            if pkl_files:
                total_size = sum(f.stat().st_size for f in pkl_files)
                houses.append((folder.name, len(pkl_files), total_size))
    houses.sort(key=lambda x: (x[1], x[2]))
    return houses


def run_command(args, cwd, label=""):
    """Run a subprocess and stream output. Returns exit code."""
    print(f"\n{'-'*60}", flush=True)
    print(f"  {label}", flush=True)
    print(f"  Command: {' '.join(str(a) for a in args)}", flush=True)
    print(f"  CWD: {cwd}", flush=True)
    print(f"{'-'*60}", flush=True)

    result = subprocess.run(
        args, cwd=str(cwd),
        # Stream output directly to console
    )
    return result.returncode


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Local batch runner (replicates server SH script)")
    parser.add_argument("--houses", type=str, default=None,
                        help="Comma-separated house IDs")
    parser.add_argument("--shortest", type=int, default=None,
                        help="Run N shortest houses (by month count)")
    parser.add_argument("--experiment_name", type=str, default="exp015_hole_repair",
                        help="Experiment name (default: exp015_hole_repair)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory name (defaults to experiment_name)")
    parser.add_argument("--skip_house_analysis", action="store_true",
                        help="Skip house pre-analysis step")
    args = parser.parse_args()

    # Determine house list
    if args.houses:
        house_ids = [h.strip() for h in args.houses.split(',')]
    elif args.shortest:
        all_houses = get_houses_sorted_by_months()
        house_ids = [h[0] for h in all_houses[:args.shortest]]
        print(f"Selected {args.shortest} shortest houses:")
        for name, n_months, size in all_houses[:args.shortest]:
            print(f"  House {name}: {n_months} months, {size/1024/1024:.1f} MB")
    else:
        print("ERROR: Specify --houses or --shortest")
        sys.exit(1)

    experiment_name = args.experiment_name
    output_dir_name = args.output_dir or experiment_name
    experiment_output = EXPERIMENT_PIPELINE / "OUTPUT" / "experiments" / output_dir_name
    reports_dir = experiment_output / "reports"

    os.makedirs(experiment_output, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Timing CSV
    timing_file = experiment_output / "house_timing.csv"
    with open(timing_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["house_id", "n_months", "start_time", "end_time",
                          "elapsed_seconds", "elapsed_human", "status"])

    total_start = time.time()

    print("\n" + "=" * 60)
    print("LOCAL BATCH RUNNER (replicating server SH script)")
    print("=" * 60)
    print(f"Experiment:  {experiment_name}")
    print(f"Houses:      {len(house_ids)} -{house_ids}")
    print(f"Output:      {experiment_output}")
    print(f"Reports:     {reports_dir}")
    print(f"Timing log:  {timing_file}")
    print("=" * 60)

    # ── Phase 1: House pre-analysis (raw data, before pipeline) ─────
    if not args.skip_house_analysis:
        print(f"\n{'='*60}")
        print(f"  PHASE 1: HOUSE PRE-ANALYSIS ({len(house_ids)} houses)")
        print(f"{'='*60}")

        for house_id in tqdm(house_ids, desc="Pre-analysis", unit="house"):
            run_command(
                [PYTHON, "scripts/run_analysis.py",
                 "--input-dir", str(DATA_DIR),
                 "--houses", house_id,
                 "--output-dir", str(reports_dir),
                 "--publish", "house"],
                cwd=PROJECT_ROOT / "house_analysis",
                label=f"House pre-analysis: house {house_id}",
            )

        if len(house_ids) > 1:
            run_command(
                [PYTHON, "scripts/run_analysis.py",
                 "--input-dir", str(DATA_DIR),
                 "--houses", ",".join(house_ids),
                 "--output-dir", str(reports_dir),
                 "--publish", "house"],
                cwd=PROJECT_ROOT / "house_analysis",
                label="Aggregate house pre-analysis report",
            )

    # ── Phase 2: Pipeline + per-house reports ────────────────────────
    successful_houses = []
    failed_houses = []

    house_pbar = tqdm(house_ids, desc="Pipeline", unit="house")
    for house_id in house_pbar:
        house_pbar.set_postfix(house=house_id)
        print(f"\n{'='*60}")
        print(f"  HOUSE {house_id}")
        print(f"{'='*60}")

        start_time = time.time()
        start_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Step 1: Run pipeline
        exit_code = run_command(
            [PYTHON, "-u", "scripts/test_single_house.py",
             "--house_id", house_id,
             "--experiment_name", experiment_name,
             "--output_path", str(experiment_output),
             "--skip_visualization"],
            cwd=EXPERIMENT_PIPELINE,
            label=f"Pipeline: house {house_id}",
        )

        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        elapsed_human = f"{hours}h {mins}m {secs}s"
        end_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if exit_code == 0:
            status = "OK"
            successful_houses.append(house_id)
        else:
            status = f"FAIL(exit={exit_code})"
            failed_houses.append(house_id)

        # Write timing
        with open(timing_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([house_id, "", start_str, end_str,
                             f"{elapsed:.0f}", elapsed_human, status])

        print(f"\n  House {house_id}: {status} ({elapsed_human})")

        # Per-house reports (only on success)
        if exit_code == 0:
            print(f"\n  Generating reports for house {house_id}...")
            report_start = time.time()

            # Segregation report
            run_command(
                [PYTHON, "scripts/run_dynamic_report.py",
                 "--experiment", str(experiment_output),
                 "--houses", house_id,
                 "--output-dir", str(reports_dir),
                 "--publish", "segregation"],
                cwd=PROJECT_ROOT / "disaggregation_analysis",
                label=f"Segregation report: house {house_id}",
            )

            # Identification report
            run_command(
                [PYTHON, "scripts/run_identification_report.py",
                 "--experiment", str(experiment_output),
                 "--houses", house_id,
                 "--output-dir", str(reports_dir),
                 "--publish", "identification"],
                cwd=PROJECT_ROOT / "identification_analysis",
                label=f"Identification report: house {house_id}",
            )

            report_elapsed = time.time() - report_start
            print(f"  Reports generated in {report_elapsed:.0f}s")

    # ── Phase 3: Aggregate pipeline reports ─────────────────────────
    if len(successful_houses) > 1:
        print(f"\n{'='*60}")
        print(f"  AGGREGATE REPORTS ({len(successful_houses)} houses)")
        print(f"{'='*60}")

        # Aggregate segregation report
        run_command(
            [PYTHON, "scripts/run_dynamic_report.py",
             "--experiment", str(experiment_output),
             "--output-dir", str(reports_dir),
             "--publish", "segregation"],
            cwd=PROJECT_ROOT / "disaggregation_analysis",
            label="Aggregate segregation report",
        )

        # Aggregate identification report
        run_command(
            [PYTHON, "scripts/run_identification_report.py",
             "--experiment", str(experiment_output),
             "--output-dir", str(reports_dir),
             "--publish", "identification"],
            cwd=PROJECT_ROOT / "identification_analysis",
            label="Aggregate identification report",
        )

    # ── Final summary ───────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    total_hours = int(total_elapsed // 3600)
    total_mins = int((total_elapsed % 3600) // 60)
    total_secs = int(total_elapsed % 60)

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"Total time:  {total_hours}h {total_mins}m {total_secs}s")
    print(f"Successful:  {len(successful_houses)}/{len(house_ids)}")
    if failed_houses:
        print(f"Failed:      {len(failed_houses)} -{failed_houses}")
    print(f"")
    print(f"Output:      {experiment_output}")
    print(f"Reports:     {reports_dir}")
    print(f"Timing:      {timing_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
