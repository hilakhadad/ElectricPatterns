"""
Generate dynamic threshold HTML reports.

Usage:
    python run_dynamic_report.py                              # Latest exp010 experiment
    python run_dynamic_report.py --experiment <path>          # Specific experiment directory
    python run_dynamic_report.py --houses 305,1234            # Specific houses only

Examples:
    python run_dynamic_report.py --experiment ../experiment_pipeline/OUTPUT/experiments/exp010_dynamic_threshold_20260215
    python run_dynamic_report.py --experiment ../experiment_pipeline/OUTPUT/experiments/exp010_dynamic_threshold_20260215 --houses 305
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import argparse

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from visualization.dynamic_html_report import (
    generate_dynamic_house_report,
    generate_dynamic_aggregate_report,
)


def find_latest_dynamic_experiment() -> Path:
    """Find the most recent exp010 dynamic threshold output directory."""
    project_root = script_dir.parent.parent
    experiments_dir = project_root / "experiment_pipeline" / "OUTPUT" / "experiments"

    if not experiments_dir.exists():
        return None

    # Look for exp010 directories, sorted by modification time (newest first)
    exp010_dirs = [
        d for d in experiments_dir.iterdir()
        if d.is_dir() and 'exp010' in d.name
    ]

    if not exp010_dirs:
        return None

    return max(exp010_dirs, key=lambda d: d.stat().st_mtime)


def discover_houses(experiment_dir: Path) -> list:
    """Discover house IDs from run_0 directory structure."""
    run_0 = experiment_dir / "run_0"
    if not run_0.exists():
        # Try run_0_th* pattern
        run_dirs = list(experiment_dir.glob("run_0*"))
        if not run_dirs:
            return []
        run_0 = run_dirs[0]

    houses = []
    for d in sorted(run_0.iterdir()):
        if d.is_dir() and d.name.startswith("house_"):
            house_id = d.name.replace("house_", "")
            houses.append(house_id)

    return houses


def main():
    parser = argparse.ArgumentParser(
        description="Generate dynamic threshold HTML reports"
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Path to experiment output directory"
    )
    parser.add_argument(
        "--houses", type=str, default=None,
        help="Comma-separated house IDs (default: all houses)"
    )
    args = parser.parse_args()

    # Find experiment directory
    if args.experiment:
        experiment_dir = Path(args.experiment)
    else:
        experiment_dir = find_latest_dynamic_experiment()

    if experiment_dir is None or not experiment_dir.exists():
        print("ERROR: No experiment directory found.")
        print("Use --experiment <path> to specify the experiment output directory.")
        sys.exit(1)

    print(f"Experiment directory: {experiment_dir}", flush=True)

    # Discover houses
    if args.houses:
        house_ids = [h.strip() for h in args.houses.split(',')]
    else:
        house_ids = discover_houses(experiment_dir)

    if not house_ids:
        print("ERROR: No houses found in experiment directory.")
        sys.exit(1)

    print(f"Houses to analyze: {', '.join(house_ids)}")
    print(flush=True)

    # Output directory
    output_dir = experiment_dir / "reports"
    os.makedirs(output_dir, exist_ok=True)

    # ── Phase 1: Per-house reports ──────────────────────────────────
    start_time = time.time()
    successful = 0
    failed = 0
    failed_houses = []

    houses_iter = house_ids
    if HAS_TQDM:
        houses_iter = tqdm(house_ids, desc="Per-house reports", unit="house")

    for house_id in houses_iter:
        try:
            out_path = str(output_dir / f"dynamic_report_{house_id}.html")
            if HAS_TQDM:
                houses_iter.set_postfix(house=house_id, ok=successful, fail=failed)
            else:
                print(f"  [{successful+failed+1}/{len(house_ids)}] house {house_id}...", end=" ", flush=True)

            generate_dynamic_house_report(
                str(experiment_dir), house_id, out_path
            )

            successful += 1
            if HAS_TQDM:
                houses_iter.set_postfix(house=house_id, ok=successful, fail=failed)
            else:
                print("OK", flush=True)
        except Exception as e:
            failed += 1
            failed_houses.append((house_id, str(e)))
            if HAS_TQDM:
                houses_iter.set_postfix(house=house_id, ok=successful, fail=failed)
            else:
                print(f"FAILED: {e}", flush=True)

    phase1_time = time.time() - start_time
    print(f"\nPer-house reports: {successful} OK, {failed} failed ({phase1_time:.1f}s)", flush=True)

    if failed_houses:
        print(f"\nFailed houses ({failed}):")
        for hid, err in failed_houses:
            print(f"  house {hid}: {err}")
        print(flush=True)

    # ── Phase 2: Aggregate report ───────────────────────────────────
    if len(house_ids) > 1 and successful > 0:
        try:
            agg_path = str(output_dir / "dynamic_report_aggregate.html")
            print("Generating aggregate report...", flush=True)
            agg_start = time.time()
            generate_dynamic_aggregate_report(
                str(experiment_dir), house_ids, agg_path
            )
            print(f"Aggregate report: OK ({time.time() - agg_start:.1f}s)", flush=True)
        except Exception as e:
            print(f"Aggregate report: FAILED: {e}", flush=True)

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}min). {successful} reports generated, {failed} failed.")
    print(f"Reports saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
