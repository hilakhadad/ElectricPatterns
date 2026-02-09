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
from tqdm import tqdm

# Add src directory to path
SCRIPT_DIR = Path(__file__).parent.parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

# Default paths - can be overridden via arguments
DEFAULT_INPUT_DIR = PROJECT_ROOT / "INPUT" / "HouseholdData"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "OUTPUT"


def list_available_houses(input_dir: Path) -> list:
    """List all available houses (looks for subfolders containing pkl files)."""
    houses = []
    if input_dir.exists():
        # Look for subfolders (each subfolder is a house)
        for folder in input_dir.iterdir():
            if folder.is_dir():
                # Check if folder contains pkl files
                pkl_files = list(folder.glob("*.pkl"))
                if pkl_files:
                    houses.append(folder.name)
    return sorted(houses)


def load_house_data_from_folder(house_folder: Path):
    """Load and concatenate all pkl files from a house folder."""
    import pandas as pd

    pkl_files = sorted(house_folder.glob("*.pkl"))
    if not pkl_files:
        return None

    dfs = []
    for f in pkl_files:
        df = pd.read_pickle(f)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values('timestamp').reset_index(drop=True)

    # Rename columns if needed (some files use '1','2','3' instead of 'w1','w2','w3')
    if '1' in data.columns and 'w1' not in data.columns:
        data = data.rename(columns={'1': 'w1', '2': 'w2', '3': 'w3'})

    return data


def run_single_house_analysis(house_id: str, input_dir: Path, output_dir: Path,
                               quiet: bool = False) -> dict:
    """Run analysis on a single house."""
    from reports import analyze_single_house, generate_house_report
    from visualization import generate_single_house_html_report

    def log(msg):
        if not quiet:
            print(msg)

    # Look for house folder
    house_folder = input_dir / house_id
    if not house_folder.exists() or not house_folder.is_dir():
        tqdm.write(f"Error: House folder not found: {house_folder}")
        return None

    log(f"Analyzing house {house_id}...")

    # Load data from all monthly files in the folder
    data = load_house_data_from_folder(house_folder)
    if data is None or len(data) == 0:
        tqdm.write(f"Error: No data files found in {house_folder}")
        return None
    log(f"  Loaded {len(data)} rows")

    # Run analysis
    analysis = analyze_single_house(data, house_id)

    # Generate JSON report
    house_output_dir = output_dir / "per_house"
    report_path = generate_house_report(analysis, str(house_output_dir), format='json')
    log(f"  JSON report saved to: {report_path}")

    # Generate HTML report for this house
    html_path = house_output_dir / f"house_{house_id}.html"
    generate_single_house_html_report(analysis, str(html_path))
    log(f"  HTML report saved to: {html_path}")

    # Print summary
    coverage = analysis.get('coverage', {})
    quality = analysis.get('data_quality', {})
    flags = analysis.get('flags', {})

    log(f"  Coverage: {coverage.get('coverage_ratio', 0):.1%}")
    log(f"  Days: {coverage.get('days_span', 0)}")
    log(f"  Quality score: {quality.get('quality_score', 0):.0f}/100")

    active_flags = [k for k, v in flags.items() if v]
    if active_flags:
        log(f"  Flags: {', '.join(active_flags)}")

    return analysis


def run_all_houses_analysis(input_dir: Path, output_dir: Path) -> None:
    """Run analysis on all houses and generate aggregate report."""
    from reports import (
        analyze_single_house, generate_house_report, load_house_data,
        aggregate_all_houses, generate_summary_report, create_comparison_table
    )
    from visualization import generate_html_report

    houses = list_available_houses(input_dir)
    if not houses:
        print(f"No house folders found in {input_dir}")
        return

    # Create timestamped run folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = output_dir / f"run_{timestamp}"
    os.makedirs(run_output_dir, exist_ok=True)

    print(f"Found {len(houses)} houses to analyze")
    print(f"Output directory: {run_output_dir}")
    print("=" * 60)

    all_analyses = []
    errors = []

    for house_id in tqdm(houses, desc="Analyzing houses", unit="house", mininterval=30):
        try:
            analysis = run_single_house_analysis(house_id, input_dir, run_output_dir, quiet=False)
            if analysis:
                all_analyses.append(analysis)
        except Exception as e:
            tqdm.write(f"  Error in house {house_id}: {e}")
            errors.append({'house_id': house_id, 'error': str(e)})

    print("\n" + "=" * 60)
    print(f"Completed: {len(all_analyses)} houses analyzed, {len(errors)} errors")

    if all_analyses:
        print("\nGenerating aggregate report...")

        # Aggregate
        aggregate = aggregate_all_houses(all_analyses)

        # Save reports - aggregate dir is inside run folder
        aggregate_dir = run_output_dir / "aggregate"
        output_paths = generate_summary_report(aggregate, str(aggregate_dir))

        print(f"Aggregate report saved to: {output_paths.get('json', aggregate_dir)}")

        # Create comparison table
        comparison_path = aggregate_dir / 'comparison_table.csv'
        comparison_df = create_comparison_table(all_analyses, str(comparison_path))
        print(f"Comparison table saved to: {comparison_path}")

        # Generate HTML report
        html_path = aggregate_dir / 'report.html'
        generate_html_report(all_analyses, str(html_path))
        print(f"HTML report saved to: {html_path}")

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
    parser.add_argument('--input-dir', type=str, help='Input directory with pkl files')
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
