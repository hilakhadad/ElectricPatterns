"""
Generate device identification (Module 2) HTML reports.

Output structure:
    identification_analysis/OUTPUT/identification_{experiment_name}_{timestamp}/
    ├── identification_report_aggregate.html
    └── house_reports/
        ├── identification_report_305.html
        └── ...

Usage:
    python run_identification_report.py                              # Latest exp010 experiment
    python run_identification_report.py --experiment <path>          # Specific experiment
    python run_identification_report.py --houses 305,1234            # Specific houses
    python run_identification_report.py --resume <dir>               # Resume: only new houses
    python run_identification_report.py --skip-activations           # Faster (skip detail)
"""
import sys
import os
import json
import logging
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

from visualization.identification_html_report import (
    generate_identification_report,
    generate_identification_aggregate_report,
    _load_pre_analysis_scores,
)

# Output base
_ANALYSIS_OUTPUT_DIR = script_dir.parent / "OUTPUT"


def find_latest_dynamic_experiment() -> Path:
    """Find the most recent exp010 dynamic threshold output directory."""
    project_root = script_dir.parent.parent
    experiments_dir = project_root / "experiment_pipeline" / "OUTPUT" / "experiments"

    if not experiments_dir.exists():
        return None

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
    """Discover house IDs from device_sessions JSON files or run_0 directory structure."""
    # Try device_sessions directory first (most direct for M2)
    sessions_dir = experiment_dir / "device_sessions"
    if sessions_dir.exists():
        houses = []
        for f in sorted(sessions_dir.iterdir()):
            if f.name.startswith("device_sessions_") and f.name.endswith(".json"):
                house_id = f.stem.replace("device_sessions_", "")
                houses.append(house_id)
        if houses:
            return houses

    # Fallback: run_0 directory structure (same as M1 report)
    run_0 = experiment_dir / "run_0"
    if not run_0.exists():
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    parser = argparse.ArgumentParser(
        description="Generate device identification (Module 2) HTML reports"
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
        "--skip-activations", action="store_true",
        dest="skip_activations",
        help="Omit Device Activations Detail section"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        dest="output_dir",
        help="Output directory (flat: all HTML files go here directly)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from existing output dir — only process new houses"
    )
    parser.add_argument(
        "--pre-analysis", type=str, default=None,
        dest="pre_analysis",
        help="Path to house_analysis output (default: auto-detect)"
    )
    parser.add_argument(
        "--publish", type=str, default=None, metavar="NAME",
        help="Publish mode: output {NAME}_report.html + {NAME}_reports/house_{id}.html "
             "(requires --output-dir)"
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        dest="aggregate_only",
        help="Skip per-house reports, generate only cross-house + aggregate report"
    )
    args = parser.parse_args()

    if args.publish and not args.output_dir:
        print("ERROR: --publish requires --output-dir")
        sys.exit(1)

    # ── Resume mode ───────────────────────────────────────────────
    resume_mode = False
    existing_houses = set()

    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"ERROR: Resume directory not found: {resume_dir}")
            sys.exit(1)

        resume_mode = True
        metadata_file = resume_dir / "_metadata.json"

        if not args.experiment and metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            args.experiment = metadata.get('experiment_dir')

        # Detect existing reports
        if args.publish:
            scan_dir = resume_dir / f"{args.publish}_reports"
        else:
            reports_subdir = resume_dir / "house_reports"
            scan_dir = reports_subdir if reports_subdir.exists() else resume_dir
        if scan_dir.exists():
            for html_file in scan_dir.glob("identification_report_*.html"):
                house_id = html_file.stem.replace("identification_report_", "")
                if house_id != "aggregate":
                    existing_houses.add(house_id)
            # Also detect publish-mode naming (house_*.html)
            for html_file in scan_dir.glob("house_*.html"):
                house_id = html_file.stem.replace("house_", "")
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
        print("ERROR: No houses found with device session data.")
        sys.exit(1)

    all_house_ids = list(house_ids)

    if resume_mode:
        new_houses = [h for h in house_ids if h not in existing_houses]
        house_ids = new_houses
        print(f"Resume mode: {len(existing_houses)} existing, "
              f"{len(new_houses)} new to process", flush=True)
    else:
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
        pre_analysis_scores = _load_pre_analysis_scores(pre_analysis_path)
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
        if args.publish:
            house_reports_dir = output_dir / f"{args.publish}_reports"
            house_reports_subdir = f"{args.publish}_reports"
        else:
            house_reports_dir = output_dir
            house_reports_subdir = None
    else:
        experiment_name = experiment_dir.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _ANALYSIS_OUTPUT_DIR / f"identification_{experiment_name}_{timestamp}"
        house_reports_dir = output_dir / "house_reports"
        house_reports_subdir = "house_reports"

    os.makedirs(house_reports_dir, exist_ok=True)

    # Save metadata
    metadata_file = output_dir / "_metadata.json"
    metadata = {
        'experiment_dir': str(experiment_dir),
        'created': datetime.now().isoformat(),
        'report_type': 'identification',
        'pre_analysis_path': str(pre_analysis_path) if pre_analysis_path else None,
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Output directory: {output_dir}", flush=True)
    print(flush=True)

    # ── Phase 1: Per-house reports ────────────────────────────────
    start_time = time.time()
    successful = 0
    failed = 0
    failed_houses = []
    cached_metrics = {}  # house_id -> {'quality': ..., 'confidence': ...}

    if args.aggregate_only:
        print("Aggregate-only mode: skipping per-house reports", flush=True)
        house_ids = []  # skip per-house loop

    houses_iter = house_ids
    if HAS_TQDM:
        houses_iter = tqdm(house_ids, desc="Per-house M2 reports", unit="house")

    for house_id in houses_iter:
        try:
            if args.publish:
                out_path = str(house_reports_dir / f"house_{house_id}.html")
            else:
                out_path = str(house_reports_dir / f"identification_report_{house_id}.html")
            if HAS_TQDM:
                houses_iter.set_postfix(house=house_id, ok=successful, fail=failed)
            else:
                print(f"  [{successful+failed+1}/{len(house_ids)}] house {house_id}...",
                      flush=True)

            result = generate_identification_report(
                str(experiment_dir), house_id, out_path,
                skip_activations_detail=args.skip_activations,
                show_timing=True,
            )
            # Cache quality/confidence for aggregate phase
            cached_metrics[house_id] = {
                'quality': result.get('quality'),
                'confidence': result.get('confidence'),
            }
            successful += 1

            if HAS_TQDM:
                houses_iter.set_postfix(house=house_id, ok=successful, fail=failed)
        except Exception as e:
            failed += 1
            failed_houses.append((house_id, str(e)))
            if HAS_TQDM:
                houses_iter.set_postfix(house=house_id, ok=successful, fail=failed)
            else:
                print(f"  FAILED: {e}", flush=True)

    phase1_time = time.time() - start_time
    print(f"\nPer-house M2 reports: {successful} OK, {failed} failed ({phase1_time:.1f}s)",
          flush=True)

    if failed_houses:
        print(f"\nFailed houses ({failed}):")
        for hid, err in failed_houses:
            print(f"  house {hid}: {err}")
        print(flush=True)

    # ── Phase 1.5: Cross-house pattern matching ─────────────────────
    cross_house_result = None
    agg_house_ids = all_house_ids if (resume_mode or args.aggregate_only) else house_ids
    if len(agg_house_ids) > 1 and (successful > 0 or resume_mode or args.aggregate_only):
        try:
            from metrics.cross_house_patterns import (
                match_patterns_across_houses,
                update_house_jsons,
                save_cross_house_summary,
            )
            print("Cross-house pattern matching...", flush=True)
            cross_house_start = time.time()
            cross_house_result = match_patterns_across_houses(
                experiment_dir, agg_house_ids,
            )
            n_global = cross_house_result['summary']['total_global_patterns']
            if n_global > 0:
                n_updated = update_house_jsons(
                    experiment_dir, cross_house_result['house_updates'],
                )
                save_cross_house_summary(experiment_dir, cross_house_result)
                print(f"  Found {n_global} cross-house patterns, "
                      f"updated {n_updated} sessions ({time.time() - cross_house_start:.1f}s)",
                      flush=True)
            else:
                print(f"  No cross-house patterns found ({time.time() - cross_house_start:.1f}s)",
                      flush=True)
        except Exception as e:
            print(f"  Cross-house pattern matching failed: {e}", flush=True)

    # ── Phase 2: Aggregate report ─────────────────────────────────
    if len(agg_house_ids) > 1 and (successful > 0 or resume_mode or args.aggregate_only):
        try:
            if args.publish:
                agg_path = str(output_dir / f"{args.publish}_report.html")
            else:
                agg_path = str(output_dir / "identification_report_aggregate.html")
            print(f"Generating aggregate M2 report ({len(agg_house_ids)} houses)...",
                  flush=True)
            agg_start = time.time()
            generate_identification_aggregate_report(
                str(experiment_dir), agg_house_ids, agg_path,
                house_reports_subdir=house_reports_subdir,
                show_progress=True,
                precomputed_metrics=cached_metrics,
                show_timing=True,
                per_house_filename_pattern="house_{house_id}.html" if args.publish else None,
                pre_analysis_scores=pre_analysis_scores,
                cross_house_result=cross_house_result,
            )
            print(f"Aggregate M2 report: OK ({time.time() - agg_start:.1f}s)", flush=True)
        except Exception as e:
            print(f"Aggregate M2 report: FAILED: {e}", flush=True)

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - start_time
    if resume_mode:
        total_reports = len(existing_houses) + successful
        print(f"\nDone in {elapsed:.1f}s. "
              f"{successful} new + {len(existing_houses)} existing = {total_reports} total. "
              f"{failed} failed.")
    else:
        print(f"\nDone in {elapsed:.1f}s. {successful} reports generated, {failed} failed.")
    print(f"Reports saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
