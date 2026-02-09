"""
Analyze experiment logs to extract rejection statistics.

Parses log files to count rejection reasons across all houses,
identifying why events weren't matched.

Usage:
    python analyze_logs.py                           # Auto-select latest experiment
    python analyze_logs.py --experiment exp007_20260205_142750
    python analyze_logs.py --experiment exp007_20260205_142750 --house 140
    python analyze_logs.py --all --output results.json
"""
import re
import json
import random
from pathlib import Path
import argparse
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# Maximum number of values to keep per rejection reason for distribution analysis
# Using reservoir sampling to keep memory bounded while maintaining representative sample
MAX_SAMPLES_PER_REASON = 100000


def find_latest_experiment(base_paths: list) -> Path:
    """Find the most recent experiment folder based on timestamp in name."""
    all_experiments = []

    for base_path in base_paths:
        if base_path.exists():
            for exp_dir in base_path.iterdir():
                if exp_dir.is_dir() and exp_dir.name.startswith('exp'):
                    all_experiments.append(exp_dir)

    if not all_experiments:
        return None

    # Sort by name (which includes timestamp) - latest will be last
    all_experiments.sort(key=lambda x: x.name)
    return all_experiments[-1]


def extract_numeric_values(detail_str: str) -> dict:
    """
    Extract all numeric values from a detail string.

    Examples:
        'min=-45W' -> {'min': -45}
        'max_dev=500W, min_dev=-100W' -> {'max_dev': 500, 'min_dev': -100}
        'on=2400W, off=1600W, diff=800W' -> {'on': 2400, 'off': 1600, 'diff': 800}
    """
    values = {}
    # Match patterns like key=value where value can be negative and have decimals
    pattern = r'(\w+)=(-?\d+(?:\.\d+)?)'
    for match in re.finditer(pattern, detail_str):
        key = match.group(1)
        value = float(match.group(2))
        values[key] = value
    return values


def get_primary_value(reason: str, values: dict) -> float:
    """Get the primary value for a rejection reason (for distribution analysis)."""
    # Map rejection reasons to their primary value key
    primary_keys = {
        'negative_remaining_power': 'min',
        'negative_event_power': 'min',
        'partial_negative_remaining': 'min',
        'partial_negative_event': 'min',
        'magnitude_mismatch': 'diff',
        'unstable_power': 'max_dev',
        'power_drop': 'drop',
        'partial_unstable_power': 'dev',
        'ratio_too_small': None,  # Special case - use ratio
    }

    key = primary_keys.get(reason)
    if key and key in values:
        return values[key]

    # Fallback: try common keys
    for k in ['min', 'diff', 'dev', 'max_dev', 'drop']:
        if k in values:
            return values[k]

    return None


def parse_log_file(log_path: Path) -> dict:
    """Parse a single log file and extract rejection statistics."""
    stats = {
        'rejections': defaultdict(int),
        'matches': defaultdict(int),
        'rejection_values': defaultdict(list),  # Store sampled numeric values for distributions
        'rejection_counts': defaultdict(int),   # Track total count for sampling
    }

    rejection_pattern = re.compile(r'REJECTED (\S+)-(\S+): (\w+)')
    correctable_pattern = re.compile(r'CORRECTABLE (\S+)-(\S+): (\w+)')
    # Match pattern now handles compound tags like EXACT-SPIKE, NOISY-CLOSE-MEDIUM-CORRECTED, PARTIAL-EXTENDED
    match_pattern = re.compile(r'Matched ([\w-]+): (\S+) <-> (\S+)')

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Check for rejection
                rej_match = rejection_pattern.search(line)
                if rej_match:
                    on_id, off_id, reason = rej_match.groups()
                    stats['rejections'][reason] += 1

                    # Extract additional details if present
                    detail_match = re.search(r'\(([^)]+)\)', line)
                    if detail_match:
                        detail = detail_match.group(1)
                        values = extract_numeric_values(detail)
                        # Extract primary value for distribution analysis
                        primary_value = get_primary_value(reason, values)
                        if primary_value is not None:
                            # Reservoir sampling: keep up to MAX_SAMPLES_PER_REASON values
                            stats['rejection_counts'][reason] += 1
                            n = stats['rejection_counts'][reason]
                            if n <= MAX_SAMPLES_PER_REASON:
                                stats['rejection_values'][reason].append(primary_value)
                            else:
                                # Randomly replace with decreasing probability
                                j = random.randint(1, n)
                                if j <= MAX_SAMPLES_PER_REASON:
                                    stats['rejection_values'][reason][j-1] = primary_value
                    continue

                # Check for correctable (small negative that was fixed)
                corr_match = correctable_pattern.search(line)
                if corr_match:
                    on_id, off_id, reason = corr_match.groups()
                    stats['matches']['CORRECTED'] += 1
                    continue

                # Check for successful match
                match_result = match_pattern.search(line)
                if match_result:
                    match_type = match_result.group(1)
                    stats['matches'][match_type] += 1

    except Exception as e:
        print(f"Error reading {log_path}: {e}")

    return stats


def aggregate_stats(all_stats: list) -> dict:
    """Aggregate statistics from all houses."""
    agg = {
        'total_rejections': 0,
        'total_matches': 0,
        'rejection_reasons': defaultdict(int),
        'match_types': defaultdict(int),
        'houses_analyzed': len(all_stats),
        'rejection_values': defaultdict(list),  # Aggregated sampled values for distributions
        'rejection_value_counts': defaultdict(int),  # Total count before sampling
    }

    for stats in all_stats:
        for reason, count in stats['rejections'].items():
            agg['rejection_reasons'][reason] += count
            agg['total_rejections'] += count

        for match_type, count in stats['matches'].items():
            agg['match_types'][match_type] += count
            agg['total_matches'] += count

        # Aggregate rejection values with reservoir sampling across houses
        for reason, values in stats['rejection_values'].items():
            current_count = agg['rejection_value_counts'][reason]
            for val in values:
                current_count += 1
                if len(agg['rejection_values'][reason]) < MAX_SAMPLES_PER_REASON:
                    agg['rejection_values'][reason].append(val)
                else:
                    j = random.randint(1, current_count)
                    if j <= MAX_SAMPLES_PER_REASON:
                        agg['rejection_values'][reason][j-1] = val
            agg['rejection_value_counts'][reason] = current_count

    return agg


def print_distribution(values: list, reason: str, total_count: int = None):
    """Print distribution statistics for a rejection reason."""
    if not values:
        return

    arr = np.array(values)
    sample_note = ""
    if total_count and total_count > len(values):
        sample_note = f" (sampled from {total_count:,})"

    print(f"\n  Distribution for {reason} (n={len(values):,}{sample_note}):")
    print(f"    Min: {np.min(arr):.0f}W")
    print(f"    P5:  {np.percentile(arr, 5):.0f}W")
    print(f"    P25: {np.percentile(arr, 25):.0f}W")
    print(f"    P50 (median): {np.percentile(arr, 50):.0f}W")
    print(f"    P75: {np.percentile(arr, 75):.0f}W")
    print(f"    P95: {np.percentile(arr, 95):.0f}W")
    print(f"    Max: {np.max(arr):.0f}W")

    # Show how many would pass with different thresholds
    if reason in ['negative_remaining_power', 'negative_event_power',
                  'partial_negative_remaining', 'partial_negative_event']:
        # For negative power, values are negative - count how many are "just barely" negative
        thresholds = [-5, -10, -20, -50, -100]
        print(f"    --- Would pass if threshold changed (estimated) ---")
        for thresh in thresholds:
            count = np.sum(arr >= thresh)
            pct = count / len(arr) * 100
            print(f"    Threshold >= {thresh}W: ~{pct:.1f}%")


def analyze_match_tags(match_types: dict) -> dict:
    """
    Analyze compound match tags to extract statistics by component.

    Tags are in format: [NOISY-|PARTIAL-]{magnitude_quality}-{duration}[-CORRECTED]
    """
    stats = {
        'magnitude_quality': defaultdict(int),  # EXACT, CLOSE, APPROX, LOOSE
        'duration': defaultdict(int),           # SPIKE, QUICK, MEDIUM, EXTENDED
        'special': defaultdict(int),            # NOISY, PARTIAL, CORRECTED
    }

    magnitude_tags = {'EXACT', 'CLOSE', 'APPROX', 'LOOSE'}
    duration_tags = {'SPIKE', 'QUICK', 'MEDIUM', 'EXTENDED'}
    special_tags = {'NOISY', 'PARTIAL', 'CORRECTED'}

    for tag, count in match_types.items():
        parts = tag.split('-')
        for part in parts:
            if part in magnitude_tags:
                stats['magnitude_quality'][part] += count
            elif part in duration_tags:
                stats['duration'][part] += count
            elif part in special_tags:
                stats['special'][part] += count

    return stats


def print_analysis(agg: dict, all_stats: list):
    """Print the analysis results."""
    print(f"\n{'='*60}")
    print("LOG ANALYSIS RESULTS")
    print(f"{'='*60}")

    print(f"\nHouses analyzed: {agg['houses_analyzed']}")
    print(f"Total matches: {agg['total_matches']:,}")
    print(f"Total rejections logged: {agg['total_rejections']:,}")

    if agg['total_matches'] + agg['total_rejections'] > 0:
        match_rate = agg['total_matches'] / (agg['total_matches'] + agg['total_rejections']) * 100
        print(f"Match rate: {match_rate:.1f}%")

    print(f"\n--- Match Types ---")
    for match_type, count in sorted(agg['match_types'].items(), key=lambda x: -x[1]):
        print(f"  {match_type}: {count:,}")

    # Analyze tag components
    tag_stats = analyze_match_tags(agg['match_types'])

    if tag_stats['magnitude_quality']:
        print(f"\n--- By Magnitude Quality ---")
        for quality in ['EXACT', 'CLOSE', 'APPROX', 'LOOSE']:
            count = tag_stats['magnitude_quality'].get(quality, 0)
            if count > 0:
                pct = count / agg['total_matches'] * 100 if agg['total_matches'] > 0 else 0
                print(f"  {quality}: {count:,} ({pct:.1f}%)")

    if tag_stats['duration']:
        print(f"\n--- By Duration ---")
        for duration in ['SPIKE', 'QUICK', 'MEDIUM', 'EXTENDED']:
            count = tag_stats['duration'].get(duration, 0)
            if count > 0:
                pct = count / agg['total_matches'] * 100 if agg['total_matches'] > 0 else 0
                print(f"  {duration}: {count:,} ({pct:.1f}%)")

    if tag_stats['special']:
        print(f"\n--- Special Tags ---")
        for special in ['NOISY', 'PARTIAL', 'CORRECTED']:
            count = tag_stats['special'].get(special, 0)
            if count > 0:
                pct = count / agg['total_matches'] * 100 if agg['total_matches'] > 0 else 0
                print(f"  {special}: {count:,} ({pct:.1f}%)")

    print(f"\n--- Rejection Reasons ---")
    for reason, count in sorted(agg['rejection_reasons'].items(), key=lambda x: -x[1]):
        pct = count / agg['total_rejections'] * 100 if agg['total_rejections'] > 0 else 0
        print(f"  {reason}: {count:,} ({pct:.1f}%)")

    # Show distributions for each rejection reason
    print(f"\n{'='*60}")
    print("REJECTION VALUE DISTRIBUTIONS")
    print(f"{'='*60}")

    has_distributions = False
    for reason, count in sorted(agg['rejection_reasons'].items(), key=lambda x: -x[1]):
        values = agg['rejection_values'].get(reason, [])
        total_count = agg['rejection_value_counts'].get(reason, len(values))
        if values:
            print_distribution(values, reason, total_count)
            has_distributions = True

    if not has_distributions:
        print("\n  No numeric values found in logs.")
        print("  This likely means the logs were created before the logging update.")
        print("  Run a new experiment to get logs with detailed rejection values.")

    # Show houses with most rejections
    house_rejections = [(stats.get('house_id', 'unknown'), sum(stats['rejections'].values()))
                        for stats in all_stats]
    house_rejections.sort(key=lambda x: -x[1])

    print(f"\n{'='*60}")
    print("TOP 10 HOUSES BY REJECTION COUNT")
    print(f"{'='*60}")
    for house_id, count in house_rejections[:10]:
        print(f"  House {house_id}: {count:,} rejections")


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment logs for rejection reasons')
    parser.add_argument('--experiment', '-e', default=None, help='Experiment folder name (auto-selects latest if not specified)')
    parser.add_argument('--house', '-H', default=None, help='Specific house ID to analyze')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all houses (default behavior, kept for compatibility)')
    parser.add_argument('--output', '-o', default=None, help='Output JSON file path for results')

    args = parser.parse_args()

    # Find experiment path - check both local and pipeline OUTPUT locations
    base_paths = [
        Path(__file__).parent.parent.parent / "experiment_pipeline" / "OUTPUT" / "experiments",
        Path("/home/hilakese/role_based_segregation_dev/experiment_pipeline/OUTPUT/experiments"),
    ]

    experiment_path = None

    if args.experiment:
        # User specified an experiment
        for base_path in base_paths:
            potential_path = base_path / args.experiment
            if potential_path.exists():
                experiment_path = potential_path
                break

        if experiment_path is None:
            print(f"Experiment not found: {args.experiment}")
            print(f"Searched in: {base_paths}")
            return
    else:
        # Auto-select latest experiment
        experiment_path = find_latest_experiment(base_paths)
        if experiment_path is None:
            print("No experiments found!")
            print(f"Searched in: {base_paths}")
            return
        print(f"Auto-selected latest experiment: {experiment_path.name}")

    print(f"Analyzing logs in: {experiment_path}")

    # Find all log files - support multiple folder structures
    all_stats = []

    # Check for central logs folder first (new structure)
    central_logs_dir = experiment_path / "logs"

    if central_logs_dir.exists():
        # New structure: logs are in experiment_path/logs/
        if args.house:
            # Single house
            log_file = central_logs_dir / f"{args.house}.log"
            if log_file.exists():
                stats = parse_log_file(log_file)
                stats['house_id'] = args.house
                all_stats.append(stats)
        else:
            # All houses - find all non-test log files
            log_files = [f for f in central_logs_dir.glob("*.log") if not f.name.startswith('test_')]
            for log_file in tqdm(log_files, desc="Parsing logs"):
                house_id = log_file.stem  # filename without extension
                stats = parse_log_file(log_file)
                stats['house_id'] = house_id
                all_stats.append(stats)
    else:
        # Old structure: logs are in house_X/logs/
        if args.house:
            house_paths = [
                experiment_path / "run_0" / f"house_{args.house}" / "logs",
                experiment_path / f"house_{args.house}" / "logs",
            ]
            for house_path in house_paths:
                if house_path.exists():
                    log_files = list(house_path.glob("*.log"))
                    for log_file in log_files:
                        if 'test' not in log_file.name:
                            stats = parse_log_file(log_file)
                            stats['house_id'] = args.house
                            all_stats.append(stats)
                    break
        else:
            run_0_path = experiment_path / "run_0"
            if run_0_path.exists():
                house_dirs = [d for d in run_0_path.iterdir() if d.is_dir() and d.name.startswith('house_')]
            else:
                house_dirs = [d for d in experiment_path.iterdir() if d.is_dir() and d.name.startswith('house_')]

            for house_dir in tqdm(house_dirs, desc="Parsing logs"):
                house_id = house_dir.name.replace('house_', '')
                logs_dir = house_dir / "logs"

                if logs_dir.exists():
                    log_files = [f for f in logs_dir.glob("*.log") if 'test' not in f.name]
                    for log_file in log_files:
                        stats = parse_log_file(log_file)
                        stats['house_id'] = house_id
                        all_stats.append(stats)

    if not all_stats:
        print("No log files found!")
        return

    # Aggregate and print results
    agg = aggregate_stats(all_stats)
    print_analysis(agg, all_stats)

    # Save to JSON file if requested
    if args.output:
        output_data = {
            'experiment': experiment_path.name,
            'houses_analyzed': agg['houses_analyzed'],
            'total_matches': agg['total_matches'],
            'total_rejections': agg['total_rejections'],
            'match_rate': agg['total_matches'] / (agg['total_matches'] + agg['total_rejections']) * 100 if (agg['total_matches'] + agg['total_rejections']) > 0 else 0,
            'rejection_reasons': dict(agg['rejection_reasons']),
            'match_types': dict(agg['match_types']),
            'distributions': {}
        }

        # Add distribution summaries
        for reason, values in agg['rejection_values'].items():
            if values:
                arr = np.array(values)
                output_data['distributions'][reason] = {
                    'count': len(values),
                    'min': float(np.min(arr)),
                    'p5': float(np.percentile(arr, 5)),
                    'p25': float(np.percentile(arr, 25)),
                    'p50': float(np.percentile(arr, 50)),
                    'p75': float(np.percentile(arr, 75)),
                    'p95': float(np.percentile(arr, 95)),
                    'max': float(np.max(arr)),
                }

        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
