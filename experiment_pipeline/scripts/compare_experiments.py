"""
Compare results across multiple experiment runs.

Produces per-house and aggregate CSVs + an HTML comparison report.

Usage:
    # Scan all experiments in OUTPUT/experiments/
    python scripts/compare_experiments.py --scan

    # Compare specific experiment directories
    python scripts/compare_experiments.py --experiments OUTPUT/experiments/exp015_xxx OUTPUT/experiments/exp016_xxx

    # Scan + filter by experiment ID prefix
    python scripts/compare_experiments.py --scan --filter exp015 exp016

    # CSV only (no HTML)
    python scripts/compare_experiments.py --scan --no-html

    # Only houses that appear in ALL experiments
    python scripts/compare_experiments.py --scan --common-only

    # Specific houses
    python scripts/compare_experiments.py --scan --houses 221,305,344

Output:
    OUTPUT/comparisons/comparison_YYYYMMDD_HHMMSS/
        experiment_comparison_per_house.csv
        experiment_comparison_aggregate.csv
        experiment_comparison.html
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Project paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
SRC_DIR = PROJECT_DIR / 'src'
sys.path.insert(0, str(SRC_DIR))

DEFAULT_EXPERIMENTS_DIR = PROJECT_DIR / 'OUTPUT' / 'experiments'
DEFAULT_OUTPUT_DIR = PROJECT_DIR / 'OUTPUT' / 'comparisons'


def main():
    parser = argparse.ArgumentParser(
        description='Compare results across multiple experiment runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input mode
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--experiments', nargs='+', metavar='DIR',
        help='Experiment directory paths to compare',
    )
    input_group.add_argument(
        '--scan', action='store_true',
        help='Scan OUTPUT/experiments/ for all experiment directories',
    )

    # Filters
    parser.add_argument(
        '--filter', nargs='+', metavar='PREFIX',
        help='Only include experiments whose name starts with these prefixes (with --scan)',
    )
    parser.add_argument(
        '--houses', type=str,
        help='Comma-separated house IDs to include',
    )
    parser.add_argument(
        '--common-only', action='store_true',
        help='Only include houses present in ALL experiments',
    )

    # Output
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})',
    )
    parser.add_argument(
        '--no-html', action='store_true',
        help='Skip HTML report, only produce CSVs',
    )
    parser.add_argument(
        '--scan-dir', type=str, default=None,
        help=f'Directory to scan (default: {DEFAULT_EXPERIMENTS_DIR})',
    )

    args = parser.parse_args()

    # Discover experiment directories
    from comparison.experiment_comparison import (
        compare_experiments, discover_experiments, load_experiment_data,
    )

    if args.scan:
        scan_dir = Path(args.scan_dir) if args.scan_dir else DEFAULT_EXPERIMENTS_DIR
        experiment_dirs = discover_experiments(scan_dir, args.filter)
        if not experiment_dirs:
            print(f"No experiments found in {scan_dir}")
            if args.filter:
                print(f"  (filtered by: {args.filter})")
            sys.exit(1)
    else:
        experiment_dirs = [Path(d) for d in args.experiments]
        # Validate
        for d in experiment_dirs:
            if not d.exists():
                print(f"ERROR: Directory not found: {d}")
                sys.exit(1)
            if not (d / 'experiment_metadata.json').exists():
                print(f"ERROR: Not a valid experiment directory (no metadata): {d}")
                sys.exit(1)

    print(f"Comparing {len(experiment_dirs)} experiments:")
    for d in experiment_dirs:
        print(f"  - {d.name}")

    # Parse house filter
    house_filter = None
    if args.houses:
        house_filter = [h.strip() for h in args.houses.split(',')]
        print(f"House filter: {house_filter}")

    if args.common_only:
        print("Mode: common houses only")

    # Run comparison
    per_house_df, aggregate_df = compare_experiments(
        experiment_dirs,
        common_only=args.common_only,
        house_filter=house_filter,
    )

    if per_house_df.empty:
        print("ERROR: No data found across experiments")
        sys.exit(1)

    # Create output directory
    output_base = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_base / f'comparison_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Save CSVs
    per_house_path = output_dir / 'experiment_comparison_per_house.csv'
    aggregate_path = output_dir / 'experiment_comparison_aggregate.csv'

    per_house_df.to_csv(per_house_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    print(f"\nPer-house CSV: {per_house_path}")
    print(f"Aggregate CSV: {aggregate_path}")

    # Print aggregate summary
    print(f"\n{'='*70}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*70}")
    for _, row in aggregate_df.iterrows():
        pct = row.get('mean_explained_pct', 0)
        median = row.get('median_explained_pct', 0)
        cr = row.get('mean_classified_rate', 0)
        print(f"  {row['exp_id']:30s}  mean={pct:5.1f}%  median={median:5.1f}%  "
              f"classified={cr:.1%}  houses={row['n_houses']}")
    print(f"{'='*70}")

    # Generate HTML
    if not args.no_html:
        from comparison.comparison_html import generate_comparison_html

        # Load experiment data for config section
        experiments_data = []
        for d in experiment_dirs:
            data = load_experiment_data(d)
            if data:
                experiments_data.append(data)

        html_path = output_dir / 'experiment_comparison.html'
        generate_comparison_html(per_house_df, aggregate_df, str(html_path), experiments_data)
        print(f"\nHTML report: {html_path}")

    n_houses = per_house_df['house_id'].nunique()
    print(f"\nDone. {len(experiment_dirs)} experiments, {n_houses} houses compared.")


if __name__ == '__main__':
    main()
