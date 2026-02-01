"""
Generate plots for recurring patterns.

Usage:
    python generate_pattern_plots.py --house 1234
    python generate_pattern_plots.py --house 1234 --patterns 1,2,3
    python generate_pattern_plots.py --all

    # Direct date list mode (copy dates from HTML report):
    python generate_pattern_plots.py --house 1234 --dates "2021-05-05, 2021-05-06" --time 14:30
    python generate_pattern_plots.py --house 1234 --dates "All dates: 2021-05-05, 2021-05-06" --time 08:00

This script loads the analysis results and generates individual plots
for each occurrence of recurring patterns.
"""
import sys
import os
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization.pattern_plots import generate_pattern_plots


def find_latest_experiment() -> Path:
    """Find the most recent experiment directory."""
    project_root = Path(__file__).parent.parent.parent
    output_base = project_root / "experiment_pipeline" / "OUTPUT"

    # Search in OUTPUT/experiments first
    experiments_dir = output_base / "experiments"
    search_dirs = []

    if experiments_dir.exists():
        search_dirs.extend([d for d in experiments_dir.iterdir() if d.is_dir()])

    # Also check OUTPUT directly
    search_dirs.extend([d for d in output_base.iterdir() if d.is_dir() and d.name != "experiments"])

    # Filter to actual experiments (have house_* dirs)
    valid_experiments = []
    for d in search_dirs:
        has_houses = any(
            (d / h / "run_0").exists() or (d / "run_0" / h).exists()
            for h in d.iterdir()
            if h.is_dir() and h.name.startswith("house_")
        )
        if has_houses:
            valid_experiments.append(d)

    if not valid_experiments:
        raise ValueError("No experiment directories found")

    # Return most recent
    return max(valid_experiments, key=lambda x: x.stat().st_mtime)


def find_latest_analysis() -> Path:
    """Find the most recent analysis output directory."""
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "OUTPUT"

    if not output_dir.exists():
        raise ValueError("No analysis output directory found")

    analysis_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("analysis_")]

    if not analysis_dirs:
        raise ValueError("No analysis runs found")

    return max(analysis_dirs, key=lambda x: x.stat().st_mtime)


def load_analysis_data(analysis_dir: Path, house_id: str) -> dict:
    """Load analysis data for a specific house."""
    # Look for JSON file with analysis data - check house_reports folder first (new location)
    json_file = analysis_dir / "house_reports" / f"house_{house_id}_analysis.json"
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Fallback: check old location (analysis_dir directly)
    json_file = analysis_dir / f"house_{house_id}_analysis.json"
    if json_file.exists():
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Try to find in aggregate JSON
    aggregate_file = analysis_dir / "all_houses_analysis.json"
    if aggregate_file.exists():
        with open(aggregate_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for house_data in data.get('houses', []):
                if str(house_data.get('house_id')) == str(house_id):
                    return house_data

    return None


def generate_from_date_list(args):
    """
    Generate plots from a direct list of dates (copy from HTML report).

    This allows users to copy the date list from the HTML report and generate
    plots without needing the analysis JSON.
    """
    # Parse dates - handle various formats
    dates_str = args.dates.strip()

    # Remove "All dates:" prefix if present
    if dates_str.lower().startswith("all dates:"):
        dates_str = dates_str[len("all dates:"):].strip()

    # Split by comma and clean up
    dates = [d.strip() for d in dates_str.split(',') if d.strip()]

    if not dates:
        print("Error: No valid dates found in --dates argument")
        return

    print(f"Parsed {len(dates)} dates: {dates[0]} ... {dates[-1]}")

    # Find experiment directory
    try:
        experiment_dir = Path(args.experiment) if args.experiment else find_latest_experiment()
        print(f"Using experiment: {experiment_dir}")
    except Exception as e:
        print(f"Error finding experiment: {e}")
        return

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "OUTPUT" / "date_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Create a synthetic pattern from the arguments
    # Phase and magnitude are optional - used only for highlighting
    pattern = {
        'phase': args.phase if args.phase else None,  # None = no specific phase highlight
        'magnitude': args.magnitude if args.magnitude else 0,
        'avg_start_time': args.time,
        'duration_minutes': args.duration,
        'dates': dates,
        'interval_type': 'custom',
    }

    time_str = args.time.replace(':', '')
    if args.phase and args.magnitude:
        print(f"\nPattern: {args.phase.upper()} {args.magnitude}W @ {args.time} ({args.duration} min)")
    else:
        print(f"\nTime window: {args.time} ({args.duration} min)")
    print(f"Dates: {len(dates)}")

    # Generate plots
    result = generate_pattern_plots(
        experiment_dir,
        args.house,
        [pattern],
        output_dir,
        hours_before=args.hours_before,
        hours_after=args.hours_after
    )

    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nGenerated {result['plots_generated']} plots")
        if result.get('errors'):
            print(f"Errors ({len(result['errors'])}):")
            for err in result['errors'][:5]:
                print(f"  - {err}")
            if len(result['errors']) > 5:
                print(f"  ... and {len(result['errors']) - 5} more")

    print(f"\nOutput location: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate plots for recurring patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From analysis data:
  python generate_pattern_plots.py --house 1234
  python generate_pattern_plots.py --house 1234 --patterns 1,2,3

  # Direct date list (copy from HTML report) - only house, dates, and time required:
  python generate_pattern_plots.py --house 1234 --dates "2021-05-05, 2021-05-06" --time 14:30
  python generate_pattern_plots.py --house 1234 --dates "All dates: 2021-05-05, 2021-05-06" --time 08:00

  # With optional phase/magnitude for highlighting:
  python generate_pattern_plots.py --house 1234 --dates "2021-05-05, 2021-05-06" --time 14:30 --phase w1 --magnitude 2000
        """
    )
    parser.add_argument('--house', type=str, help='House ID to generate plots for')
    parser.add_argument('--all', action='store_true', help='Generate plots for all houses')
    parser.add_argument('--patterns', type=str, default=None,
                        help='Comma-separated pattern indices (1-based) to generate')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Path to experiment directory')
    parser.add_argument('--analysis', type=str, default=None,
                        help='Path to analysis output directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots')
    parser.add_argument('--hours-before', type=float, default=1.0,
                        help='Hours to show before pattern (default: 1.0)')
    parser.add_argument('--hours-after', type=float, default=1.0,
                        help='Hours to show after pattern (default: 1.0)')

    # Direct date list mode (copy from HTML report)
    parser.add_argument('--dates', type=str, default=None,
                        help='Comma-separated list of dates (YYYY-MM-DD). Can include "All dates:" prefix.')
    parser.add_argument('--phase', type=str, default=None,
                        help='Phase for highlighting (w1/w2/w3). Optional - all phases shown anyway.')
    parser.add_argument('--magnitude', type=int, default=None,
                        help='Magnitude in watts for annotation. Optional.')
    parser.add_argument('--time', type=str, default='12:00',
                        help='Center time (HH:MM) for plot window (default: 12:00)')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration in minutes for plot window (default: 60)')

    args = parser.parse_args()

    # Date list mode - generate plots directly from dates
    if args.dates:
        if not args.house:
            parser.error("--house is required when using --dates")
        return generate_from_date_list(args)

    if not args.house and not args.all:
        parser.error("Either --house, --all, or --dates must be specified")

    # Find directories
    try:
        experiment_dir = Path(args.experiment) if args.experiment else find_latest_experiment()
        print(f"Using experiment: {experiment_dir}")
    except Exception as e:
        print(f"Error finding experiment: {e}")
        return

    try:
        analysis_dir = Path(args.analysis) if args.analysis else find_latest_analysis()
        print(f"Using analysis: {analysis_dir}")
    except Exception as e:
        print(f"Error finding analysis: {e}")
        print("Note: Run full analysis first with: python run_analysis.py --full")
        return

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = analysis_dir / "pattern_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print()

    # Get house list
    if args.all:
        # Find all houses with analysis
        house_ids = []
        for f in analysis_dir.glob("house_*_analysis.json"):
            house_id = f.stem.replace("house_", "").replace("_analysis", "")
            house_ids.append(house_id)

        # Also check aggregate file
        aggregate_file = analysis_dir / "all_houses_analysis.json"
        if aggregate_file.exists():
            with open(aggregate_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for house_data in data.get('houses', []):
                    h_id = str(house_data.get('house_id'))
                    if h_id not in house_ids:
                        house_ids.append(h_id)

        if not house_ids:
            print("No houses found in analysis output")
            return
        print(f"Found {len(house_ids)} houses")
    else:
        house_ids = [args.house]

    # Parse pattern indices if specified
    pattern_indices = None
    if args.patterns:
        pattern_indices = [int(p) - 1 for p in args.patterns.split(',')]

    # Generate plots for each house
    total_plots = 0
    for house_id in house_ids:
        print(f"\nProcessing house {house_id}...")

        # Load analysis data
        analysis_data = load_analysis_data(analysis_dir, house_id)
        if not analysis_data:
            print(f"  No analysis data found for house {house_id}")
            continue

        # Get patterns from analysis
        first_iter = analysis_data.get('first_iteration', {})
        patterns_data = first_iter.get('patterns', {})
        recurring_matches = patterns_data.get('recurring_matches', {})
        patterns = recurring_matches.get('patterns', [])

        if not patterns:
            print(f"  No recurring patterns found for house {house_id}")
            continue

        # Filter patterns if indices specified
        if pattern_indices is not None:
            patterns = [patterns[i] for i in pattern_indices if i < len(patterns)]

        print(f"  Found {len(patterns)} patterns to plot")

        # Generate plots
        result = generate_pattern_plots(
            experiment_dir,
            house_id,
            patterns,
            output_dir,
            hours_before=args.hours_before,
            hours_after=args.hours_after
        )

        if 'error' in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Generated {result['plots_generated']} plots")
            total_plots += result['plots_generated']

            if result.get('errors'):
                print(f"  Errors: {len(result['errors'])}")

    print()
    print(f"Total plots generated: {total_plots}")
    print(f"Output location: {output_dir}")


if __name__ == "__main__":
    main()
