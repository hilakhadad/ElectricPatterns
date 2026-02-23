"""
Generate dynamic threshold HTML reports.

Output structure (follows disaggregation_analysis conventions):
    disaggregation_analysis/OUTPUT/analysis_{experiment_name}_{timestamp}/
    ├── dynamic_report_aggregate.html       # Aggregate report
    └── house_reports/                      # Per-house reports
        ├── dynamic_report_305.html
        ├── dynamic_report_1.html
        └── ...

Usage:
    python run_dynamic_report.py                              # Latest exp010 experiment
    python run_dynamic_report.py --experiment <path>          # Specific experiment directory
    python run_dynamic_report.py --houses 305,1234            # Specific houses only
    python run_dynamic_report.py --pre-analysis <path>        # Specify house_analysis output
    python run_dynamic_report.py --resume <analysis_dir>      # Resume: only process new houses

Examples:
    python run_dynamic_report.py --experiment ../experiment_pipeline/OUTPUT/experiments/exp010_dynamic_threshold_20260215
    python run_dynamic_report.py --houses 305
    python run_dynamic_report.py --resume ../OUTPUT/analysis_exp010_20260218_120000
"""
import sys
import os
import json
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
from reports.aggregate_report import load_pre_analysis_scores

# Output base: disaggregation_analysis/OUTPUT/
_ANALYSIS_OUTPUT_DIR = script_dir.parent / "OUTPUT"


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


def find_latest_house_analysis() -> Path:
    """Auto-detect latest house_analysis output directory."""
    project_root = script_dir.parent.parent
    house_analysis_output = project_root / "house_analysis" / "OUTPUT"

    if not house_analysis_output.exists():
        return None

    run_dirs = sorted(
        [d for d in house_analysis_output.iterdir()
         if d.is_dir() and d.name.startswith("run_")],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


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


def count_iterations(experiment_dir: Path) -> int:
    """Count total iteration (run_*) directories in experiment."""
    return len(list(experiment_dir.glob("run_*")))


def house_iterations(experiment_dir: Path, house_id: str) -> int:
    """Count how many run directories contain this house."""
    count = 0
    for run_dir in experiment_dir.glob("run_*"):
        if (run_dir / f"house_{house_id}").exists():
            count += 1
    return count


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
    parser.add_argument(
        "--pre-analysis", type=str, default=None,
        dest="pre_analysis",
        help="Path to house_analysis output (default: auto-detect)"
    )
    parser.add_argument(
        "--skip-activations", action="store_true",
        dest="skip_activations",
        help="Omit Device Activations Detail section (saves disk space)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        dest="output_dir",
        help="Output directory (flat: all HTML files go here directly)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to existing analysis output dir — only process new houses, regenerate aggregate"
    )
    parser.add_argument(
        "--fast-mode", action="store_true",
        dest="fast_mode",
        help="Skip expensive pattern analysis (faster reports)"
    )
    args = parser.parse_args()

    # ── Resume mode: load metadata from previous run ────────────
    resume_mode = False
    existing_houses = set()

    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"ERROR: Resume directory not found: {resume_dir}")
            sys.exit(1)

        resume_mode = True
        metadata_file = resume_dir / "_metadata.json"

        # Load experiment_dir from metadata (unless --experiment overrides)
        if not args.experiment and metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            args.experiment = metadata.get('experiment_dir')
            if not args.pre_analysis and metadata.get('pre_analysis_path'):
                args.pre_analysis = metadata['pre_analysis_path']

        # Detect existing house reports
        reports_subdir = resume_dir / "house_reports"
        scan_dir = reports_subdir if reports_subdir.exists() else resume_dir
        for html_file in scan_dir.glob("dynamic_report_*.html"):
            house_id = html_file.stem.replace("dynamic_report_", "")
            if house_id != "aggregate":
                existing_houses.add(house_id)

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

    # In resume mode, filter out already-processed houses
    # Also regenerate reports for houses whose experiment has more iterations now
    all_house_ids = list(house_ids)  # keep full list for aggregate
    if resume_mode:
        total_iters = count_iterations(experiment_dir)
        new_houses = [h for h in house_ids if h not in existing_houses]

        # Stale: houses with a report but fewer iterations than expected
        stale_houses = [
            h for h in existing_houses
            if h in set(all_house_ids) and house_iterations(experiment_dir, h) < total_iters
        ]

        house_ids = new_houses + stale_houses
        print(f"Resume mode: {len(existing_houses)} existing reports, "
              f"{len(new_houses)} new + {len(stale_houses)} incomplete to (re)process "
              f"(total iterations: {total_iters})", flush=True)

    if not resume_mode:
        print(f"Houses to analyze: {len(all_house_ids)}", flush=True)

    # Load pre-analysis quality scores
    pre_analysis_scores = {}
    if args.pre_analysis:
        pre_analysis_path = Path(args.pre_analysis).resolve()
    else:
        pre_analysis_path = find_latest_house_analysis()
        if pre_analysis_path:
            print(f"Auto-detected house_analysis output: {pre_analysis_path.name}", flush=True)

    if pre_analysis_path and pre_analysis_path.exists():
        pre_analysis_scores = load_pre_analysis_scores(pre_analysis_path)
    elif pre_analysis_path:
        print(f"Warning: Pre-analysis path not found: {pre_analysis_path}", flush=True)

    if pre_analysis_scores:
        print(f"Pre-analysis scores loaded: {len(pre_analysis_scores)} houses", flush=True)
    print(flush=True)

    # Output directory
    if resume_mode:
        output_dir = Path(args.resume)
        reports_subdir = output_dir / "house_reports"
        if reports_subdir.exists():
            house_reports_dir = reports_subdir
            house_reports_subdir = "house_reports"
        else:
            house_reports_dir = output_dir
            house_reports_subdir = None
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        house_reports_dir = output_dir  # Flat: per-house reports go directly here
        house_reports_subdir = None     # No subdirectory in aggregate links
    else:
        experiment_name = experiment_dir.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _ANALYSIS_OUTPUT_DIR / f"analysis_{experiment_name}_{timestamp}"
        house_reports_dir = output_dir / "house_reports"
        house_reports_subdir = "house_reports"

    os.makedirs(house_reports_dir, exist_ok=True)

    # Save metadata for future --resume
    metadata_file = output_dir / "_metadata.json"
    metadata = {
        'experiment_dir': str(experiment_dir),
        'created': datetime.now().isoformat(),
        'pre_analysis_path': str(pre_analysis_path) if pre_analysis_path else None,
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Output directory: {output_dir}", flush=True)
    print(flush=True)

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
            out_path = str(house_reports_dir / f"dynamic_report_{house_id}.html")
            if HAS_TQDM:
                houses_iter.set_postfix(house=house_id, ok=successful, fail=failed)
            else:
                print(f"  [{successful+failed+1}/{len(house_ids)}] house {house_id}...", end=" ", flush=True)

            house_pre = pre_analysis_scores.get(house_id, {})
            pre_quality = house_pre.get('quality_score') if isinstance(house_pre, dict) else house_pre
            generate_dynamic_house_report(
                str(experiment_dir), house_id, out_path,
                pre_quality=pre_quality,
                skip_activations_detail=args.skip_activations,
                show_timing=True,
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
    # Use all_house_ids (old + new) so aggregate covers everything
    agg_house_ids = all_house_ids if resume_mode else house_ids
    if len(agg_house_ids) > 1 and (successful > 0 or resume_mode):
        try:
            agg_path = str(output_dir / "dynamic_report_aggregate.html")
            print(f"Generating aggregate report ({len(agg_house_ids)} houses)...", flush=True)
            agg_start = time.time()
            generate_dynamic_aggregate_report(
                str(experiment_dir), agg_house_ids, agg_path,
                pre_analysis_scores=pre_analysis_scores,
                house_reports_subdir=house_reports_subdir,
                show_progress=True,
            )
            print(f"Aggregate report: OK ({time.time() - agg_start:.1f}s)", flush=True)
        except Exception as e:
            print(f"Aggregate report: FAILED: {e}", flush=True)

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    if resume_mode:
        total_reports = len(existing_houses) + successful
        print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}min). "
              f"{successful} new + {len(existing_houses)} existing = {total_reports} total reports. "
              f"{failed} failed.")
    else:
        print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f}min). {successful} reports generated, {failed} failed.")
    print(f"Reports saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
