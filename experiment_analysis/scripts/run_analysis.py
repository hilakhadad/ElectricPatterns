"""
Run experiment analysis for all houses.

By default, analyzes the most recent experiment run in FAST mode (skips pattern analysis).

Usage:
    python run_analysis.py                    # Analyze latest experiment (fast mode)
    python run_analysis.py --full             # Full mode with pattern analysis
    python run_analysis.py --houses 1234,5678 # Analyze specific houses from latest
    python run_analysis.py --experiment <path> # Analyze specific experiment

Examples:
    python run_analysis.py
    python run_analysis.py --full
    python run_analysis.py --houses 1234,5678
    python run_analysis.py --experiment ../experiment_pipeline/OUTPUT/experiment_20240115_120000
"""
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from reports.experiment_report import analyze_experiment_house, generate_experiment_report
from reports.aggregate_report import aggregate_experiment_results, generate_summary_report, create_comparison_table
from visualization.html_report import generate_html_report, generate_house_html_report


def _save_analysis_json(analysis: dict, output_path: Path):
    """Save analysis data to JSON for pattern plot generation."""
    # Convert datetime objects to strings for JSON serialization
    def serialize(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        return str(obj)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, default=serialize, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  Warning: Could not save JSON for house {analysis.get('house_id')}: {e}")


def find_latest_experiment() -> Path:
    """Find the most recent experiment output directory."""
    # Look in experiment_pipeline/OUTPUT
    project_root = script_dir.parent.parent
    pipeline_output = project_root / "experiment_pipeline" / "OUTPUT"

    if not pipeline_output.exists():
        return None

    experiment_dirs = []

    # Search in OUTPUT directly and in OUTPUT/experiments subdirectory
    search_dirs = [pipeline_output]
    experiments_subdir = pipeline_output / "experiments"
    if experiments_subdir.exists():
        search_dirs.append(experiments_subdir)

    for search_dir in search_dirs:
        for item in search_dir.iterdir():
            if item.is_dir():
                # Check if it looks like an experiment directory
                # Support two structures:
                # 1. New structure: experiment_dir/run_0/, run_1/, etc.
                # 2. Old structure: experiment_dir/house_X/run_0/...

                # Check for new structure (run_X directories at top level)
                has_run_dirs = any(
                    d.is_dir() and d.name.startswith("run_")
                    for d in item.iterdir()
                )

                # Check for old structure (house_X/run_0)
                has_house_dirs = any(
                    (item / d.name / "run_0").exists()
                    for d in item.iterdir()
                    if d.is_dir() and d.name.startswith("house_")
                )

                if has_run_dirs or has_house_dirs:
                    experiment_dirs.append(item)

    if not experiment_dirs:
        return None

    # Sort by modification time, get most recent
    experiment_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return experiment_dirs[0]


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Path to experiment output directory (default: latest)')
    parser.add_argument('--houses', type=str, default=None,
                        help='Comma-separated list of house IDs (default: auto-detect)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: experiment_analysis/OUTPUT)')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum iterations to analyze per house (default: 10)')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Fast mode: skip expensive pattern analysis for quicker results')
    parser.add_argument('--full', action='store_true',
                        help='Full mode: include pattern analysis (slower but complete)')

    args = parser.parse_args()

    # Find experiment directory
    if args.experiment:
        experiment_dir = Path(args.experiment).resolve()
    else:
        experiment_dir = find_latest_experiment()
        if experiment_dir is None:
            print("Error: No experiment found in experiment_pipeline/OUTPUT/")
            print("Run an experiment first or specify --experiment <path>")
            sys.exit(1)
        print(f"Auto-detected latest experiment: {experiment_dir.name}")

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    # Parse house IDs if provided
    house_ids = None
    if args.houses:
        house_ids = [h.strip() for h in args.houses.split(',')]

    # Set output directory
    if args.output:
        output_dir = Path(args.output).resolve()
    else:
        output_dir = script_dir.parent / "OUTPUT"

    # Create timestamped output folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = experiment_dir.name
    run_output_dir = output_dir / f"analysis_{experiment_name}_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment Analysis")
    print(f"=" * 60)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Output directory: {run_output_dir}")
    print()

    # Run aggregate analysis
    # --full overrides --fast (which is now default)
    fast_mode = args.fast and not args.full
    mode_str = " (FAST MODE)" if fast_mode else " (FULL MODE - with patterns)"
    print(f"Analyzing houses...{mode_str}")
    analysis_start = time.time()
    analyses = aggregate_experiment_results(
        experiment_dir,
        house_ids=house_ids,
        max_iterations=args.max_iterations,
        fast_mode=fast_mode
    )
    analysis_time = time.time() - analysis_start

    valid_analyses = [a for a in analyses if a.get('status') != 'no_data']
    print(f"Found data for {len(valid_analyses)} houses (took {analysis_time:.1f}s)")
    print()

    if not valid_analyses:
        print("No experiment data found. Exiting.")
        sys.exit(1)

    # Generate text summary
    print("Generating summary report...")
    t0 = time.time()
    summary_text = generate_summary_report(analyses)
    summary_path = run_output_dir / "summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"  Saved: {summary_path} ({time.time()-t0:.1f}s)")

    # Generate comparison CSV
    print("Generating comparison table...")
    t0 = time.time()
    comparison_df = create_comparison_table(analyses)
    csv_path = run_output_dir / "comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({time.time()-t0:.1f}s)")

    # Generate per-house reports (text and HTML)
    print(f"Generating per-house reports ({len(valid_analyses)} houses)...")
    t0 = time.time()
    house_reports_dir = run_output_dir / "house_reports"
    house_reports_dir.mkdir(exist_ok=True)

    for i, analysis in enumerate(valid_analyses, 1):
        house_id = analysis.get('house_id', 'unknown')

        # Text report
        report_text = generate_experiment_report(analysis)
        report_path = house_reports_dir / f"house_{house_id}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        # HTML report
        html_report_path = house_reports_dir / f"house_{house_id}.html"
        generate_house_html_report(analysis, str(html_report_path))

        # JSON data (for pattern plot generation) - save in same folder as txt/html
        json_path = house_reports_dir / f"house_{house_id}_analysis.json"
        _save_analysis_json(analysis, json_path)

        if i % 10 == 0:
            print(f"  Progress: {i}/{len(valid_analyses)} house reports generated...")

    print(f"  Saved {len(valid_analyses)} reports (txt + html) to {house_reports_dir} ({time.time()-t0:.1f}s)")

    # Generate HTML report
    print("Generating HTML report...")
    html_path = run_output_dir / "report.html"
    generate_html_report(
        analyses,
        str(html_path),
        title=f"Experiment Analysis: {experiment_name}"
    )
    print(f"  Saved: {html_path}")

    # Print quick summary
    print()
    print("=" * 60)
    print("Quick Summary")
    print("=" * 60)

    overall_scores = [a.get('scores', {}).get('overall_score', 0) for a in valid_analyses]
    avg_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0

    print(f"Houses analyzed: {len(valid_analyses)}")
    print(f"Average overall score: {avg_score:.0f}/100")

    # Count by tier
    excellent = sum(1 for s in overall_scores if s >= 80)
    good = sum(1 for s in overall_scores if 60 <= s < 80)
    fair = sum(1 for s in overall_scores if 40 <= s < 60)
    poor = sum(1 for s in overall_scores if s < 40)

    print(f"Score distribution: {excellent} excellent, {good} good, {fair} fair, {poor} poor")

    # Flag summary
    flag_counts = {}
    for a in valid_analyses:
        for flag, value in a.get('flags', {}).items():
            if value:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    if flag_counts:
        print(f"\nTop issues:")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1])[:3]:
            print(f"  - {flag.replace('_', ' ').title()}: {count} houses")

    print()
    print(f"Full report: {html_path}")


if __name__ == "__main__":
    main()
