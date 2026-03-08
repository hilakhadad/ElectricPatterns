"""
Generate cross-house insights report.

Scans all houses' identification results and produces an interactive HTML
report with population-level insights: device prevalence, temporal patterns,
classification quality, segregation effectiveness, anomalies, and device
signatures.

Usage:
    python run_cross_house_insights.py                                  # Latest experiment
    python run_cross_house_insights.py --experiment <path>              # Specific experiment
    python run_cross_house_insights.py --houses 305,344,263             # Specific houses
    python run_cross_house_insights.py --output-dir <path>              # Custom output location
    python run_cross_house_insights.py --house-reports <path>           # Include links to per-house reports
    python run_cross_house_insights.py --no-segregation                 # Skip segregation data
"""
import sys
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

from metrics.cross_house_insights import (
    load_all_house_data,
    load_segregation_data,
    compute_all_insights,
)
from visualization.insights_html_report import generate_insights_report

# Output base
_ANALYSIS_OUTPUT_DIR = script_dir.parent / "OUTPUT"


def find_latest_experiment() -> Path:
    """Find the most recent experiment output directory."""
    project_root = script_dir.parent.parent
    experiments_dir = project_root / "experiment_pipeline" / "OUTPUT" / "experiments"

    if not experiments_dir.exists():
        return None

    # Look for any experiment directories
    exp_dirs = [
        d for d in experiments_dir.iterdir()
        if d.is_dir() and (d / 'device_sessions').exists()
    ]

    if not exp_dirs:
        return None

    return max(exp_dirs, key=lambda d: d.stat().st_mtime)


def discover_houses(experiment_dir: Path) -> list:
    """Discover house IDs from device_sessions JSON files."""
    sessions_dir = experiment_dir / "device_sessions"
    if not sessions_dir.exists():
        return []

    houses = []
    for f in sorted(sessions_dir.iterdir()):
        if f.name.startswith("device_sessions_") and f.name.endswith(".json"):
            house_id = f.stem.replace("device_sessions_", "")
            houses.append(house_id)
    return houses


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    parser = argparse.ArgumentParser(
        description="Generate cross-house insights report from identification results"
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
        "--output-dir", type=str, default=None,
        dest="output_dir",
        help="Output directory for the HTML report"
    )
    parser.add_argument(
        "--house-reports", type=str, default=None,
        dest="house_reports",
        help="Path to per-house report directory (for clickable links)"
    )
    parser.add_argument(
        "--no-segregation", action="store_true",
        dest="no_segregation",
        help="Skip loading segregation data (faster)"
    )
    args = parser.parse_args()

    # ── Find experiment ────────────────────────────────────────────
    if args.experiment:
        experiment_dir = Path(args.experiment)
    else:
        experiment_dir = find_latest_experiment()

    if experiment_dir is None or not experiment_dir.exists():
        print("ERROR: No experiment directory found. Use --experiment <path>.")
        sys.exit(1)

    experiment_name = experiment_dir.name
    print(f"Experiment: {experiment_dir}")

    # ── Discover houses ────────────────────────────────────────────
    if args.houses:
        house_ids = [h.strip() for h in args.houses.split(',')]
    else:
        house_ids = discover_houses(experiment_dir)

    if not house_ids:
        print("ERROR: No houses found. Check experiment directory.")
        sys.exit(1)

    print(f"Houses: {len(house_ids)}")

    # ── Load data ──────────────────────────────────────────────────
    house_data = load_all_house_data(experiment_dir, house_ids, show_progress=True)

    if not house_data:
        print("ERROR: No house data loaded.")
        sys.exit(1)

    segregation_data = None
    if not args.no_segregation:
        segregation_data = load_segregation_data(experiment_dir, house_ids)

    # ── Compute insights ───────────────────────────────────────────
    print("Computing insights...")
    insights = compute_all_insights(house_data, segregation_data)

    # ── Generate report ────────────────────────────────────────────
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / 'cross_house_insights.html')
    else:
        output_dir = _ANALYSIS_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = str(output_dir / f'cross_house_insights_{ts}.html')

    report_path = generate_insights_report(
        insights=insights,
        experiment_dir=str(experiment_dir),
        output_path=output_path,
        experiment_name=experiment_name,
        house_reports_dir=args.house_reports,
    )

    print(f"\nInsights report saved to: {report_path}")


if __name__ == '__main__':
    main()
