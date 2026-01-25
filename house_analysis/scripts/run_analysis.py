"""
Run house analysis on all houses or a specific house.

Usage:
    python run_analysis.py                    # Analyze all houses
    python run_analysis.py --house 140        # Analyze specific house
    python run_analysis.py --list             # List available houses
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path
SCRIPT_DIR = Path(__file__).parent.parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

# Default paths - can be overridden via arguments
DEFAULT_INPUT_DIR = PROJECT_ROOT / "INPUT" / "HouseholdData"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "OUTPUT"


def list_available_houses(input_dir: Path) -> list:
    """List all available house CSV files."""
    houses = []
    if input_dir.exists():
        for f in input_dir.glob("*.csv"):
            house_id = f.stem
            houses.append(house_id)
    return sorted(houses)


def run_single_house_analysis(house_id: str, input_dir: Path, output_dir: Path) -> dict:
    """Run analysis on a single house."""
    from reports import analyze_single_house, generate_house_report, load_house_data

    input_file = input_dir / f"{house_id}.csv"
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return None

    print(f"Analyzing house {house_id}...")

    # Load data
    data = load_house_data(str(input_file))
    print(f"  Loaded {len(data)} rows")

    # Run analysis
    analysis = analyze_single_house(data, house_id)

    # Generate report
    house_output_dir = output_dir / "per_house"
    report_path = generate_house_report(analysis, str(house_output_dir), format='json')
    print(f"  Report saved to: {report_path}")

    # Print summary
    coverage = analysis.get('coverage', {})
    quality = analysis.get('data_quality', {})
    flags = analysis.get('flags', {})

    print(f"  Coverage: {coverage.get('coverage_ratio', 0):.1%}")
    print(f"  Days: {coverage.get('days_span', 0)}")
    print(f"  Quality score: {quality.get('quality_score', 0):.0f}/100")

    active_flags = [k for k, v in flags.items() if v]
    if active_flags:
        print(f"  Flags: {', '.join(active_flags)}")

    return analysis


def run_all_houses_analysis(input_dir: Path, output_dir: Path) -> None:
    """Run analysis on all houses and generate aggregate report."""
    from reports import (
        analyze_single_house, generate_house_report, load_house_data,
        aggregate_all_houses, generate_summary_report, create_comparison_table
    )

    houses = list_available_houses(input_dir)
    if not houses:
        print(f"No house files found in {input_dir}")
        return

    print(f"Found {len(houses)} houses to analyze")
    print("=" * 60)

    all_analyses = []
    errors = []

    for i, house_id in enumerate(houses, 1):
        print(f"\n[{i}/{len(houses)}] ", end="")
        try:
            analysis = run_single_house_analysis(house_id, input_dir, output_dir)
            if analysis:
                all_analyses.append(analysis)
        except Exception as e:
            print(f"  Error: {e}")
            errors.append({'house_id': house_id, 'error': str(e)})

    print("\n" + "=" * 60)
    print(f"Completed: {len(all_analyses)} houses analyzed, {len(errors)} errors")

    if all_analyses:
        print("\nGenerating aggregate report...")

        # Aggregate
        aggregate = aggregate_all_houses(all_analyses)

        # Save reports
        aggregate_dir = output_dir / "aggregate"
        output_paths = generate_summary_report(aggregate, str(aggregate_dir))

        print(f"Aggregate report saved to: {output_paths.get('json', aggregate_dir)}")

        # Create comparison table
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        comparison_path = aggregate_dir / f'comparison_table_{timestamp}.csv'
        comparison_df = create_comparison_table(all_analyses, str(comparison_path))
        print(f"Comparison table saved to: {comparison_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total houses: {aggregate.get('total_houses', 0)}")

        if 'quality_tiers' in aggregate:
            print("\nQuality tiers:")
            for tier, info in aggregate['quality_tiers'].items():
                print(f"  {tier}: {info['count']} houses")

        if 'flag_percentages' in aggregate:
            print("\nCommon issues:")
            for flag, pct in sorted(aggregate['flag_percentages'].items(),
                                   key=lambda x: -x[1]):
                if pct > 0:
                    print(f"  {flag}: {pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Analyze household power data')
    parser.add_argument('--house', type=str, help='Specific house ID to analyze')
    parser.add_argument('--list', action='store_true', help='List available houses')
    parser.add_argument('--input-dir', type=str, help='Input directory with CSV files')
    parser.add_argument('--output-dir', type=str, help='Output directory for reports')

    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_INPUT_DIR
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    if args.list:
        houses = list_available_houses(input_dir)
        print(f"\nAvailable houses ({len(houses)}):")
        for h in houses:
            print(f"  {h}")
        return

    os.makedirs(output_dir, exist_ok=True)

    if args.house:
        run_single_house_analysis(args.house, input_dir, output_dir)
    else:
        run_all_houses_analysis(input_dir, output_dir)


if __name__ == "__main__":
    main()
