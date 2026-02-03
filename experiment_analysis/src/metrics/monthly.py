"""
Monthly breakdown metrics for experiment results.

Analyzes performance per month to identify problematic periods.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


def _get_house_dir(experiment_dir: Path, house_id: str, run_number: int) -> Path:
    """Get the house directory, supporting both old and new structures."""
    new_dir = experiment_dir / f"run_{run_number}" / f"house_{house_id}"
    if new_dir.exists():
        return new_dir
    return experiment_dir / f"house_{house_id}" / f"run_{run_number}" / f"house_{house_id}"


def calculate_monthly_metrics(experiment_dir: Path, house_id: str,
                               run_number: int = 0) -> Dict[str, Any]:
    """
    Calculate metrics broken down by month.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number to analyze

    Returns:
        Dictionary with monthly breakdown metrics
    """
    metrics = {
        'house_id': house_id,
        'run_number': run_number,
        'monthly_data': [],
    }

    house_dir = _get_house_dir(experiment_dir, house_id, run_number)

    # Load on_off events with month info
    on_off_dir = house_dir / "on_off"
    if on_off_dir.exists():
        on_off_files = sorted(on_off_dir.glob("on_off_*.csv"))
    else:
        on_off_files = list(house_dir.glob("on_off_*.csv"))

    if not on_off_files:
        return metrics

    # Load matches with month info
    matches_dir = house_dir / "matches"
    if matches_dir.exists():
        matches_files = sorted(matches_dir.glob(f"matches_{house_id}_*.csv"))
    else:
        matches_files = list(house_dir.glob("matches_*.csv"))

    # Process each month
    monthly_stats = {}

    for on_off_file in on_off_files:
        # Extract month from filename (e.g., on_off_1500_01_2019.csv)
        filename = on_off_file.stem
        parts = filename.split('_')

        # Try to find month/year in filename
        month_year = None
        if len(parts) >= 4:
            try:
                month = int(parts[-2])
                year = int(parts[-1])
                month_year = f"{year}-{month:02d}"
            except ValueError:
                pass

        if not month_year:
            # Fall back to reading timestamps from file
            try:
                df = pd.read_csv(on_off_file, nrows=1)
                if 'start' in df.columns:
                    ts = pd.to_datetime(df['start'].iloc[0], format='%d/%m/%Y %H:%M', errors='coerce')
                    if pd.notna(ts):
                        month_year = ts.strftime('%Y-%m')
            except Exception:
                continue

        if not month_year:
            continue

        # Load on_off data for this month
        try:
            on_off_df = pd.read_csv(on_off_file)
        except Exception:
            continue

        # Calculate metrics for this month
        month_metrics = {
            'month': month_year,
            'total_events': len(on_off_df),
            'on_events': len(on_off_df[on_off_df['event'] == 'on']),
            'off_events': len(on_off_df[on_off_df['event'] == 'off']),
        }

        # Matching stats
        if 'matched' in on_off_df.columns:
            matched = on_off_df['matched'].sum()
            month_metrics['matched_events'] = int(matched)
            month_metrics['unmatched_events'] = len(on_off_df) - int(matched)
            month_metrics['matching_rate'] = matched / len(on_off_df) if len(on_off_df) > 0 else 0

        # Phase distribution
        if 'phase' in on_off_df.columns:
            phase_counts = on_off_df['phase'].value_counts().to_dict()
            month_metrics['events_by_phase'] = phase_counts

        # Magnitude stats
        if 'magnitude' in on_off_df.columns:
            mags = on_off_df['magnitude'].abs()
            month_metrics['avg_magnitude'] = mags.mean()
            month_metrics['total_power'] = mags.sum()

        monthly_stats[month_year] = month_metrics

    # Load matches per month
    for matches_file in matches_files:
        filename = matches_file.stem
        parts = filename.split('_')

        month_year = None
        if len(parts) >= 4:
            try:
                month = int(parts[-2])
                year = int(parts[-1])
                month_year = f"{year}-{month:02d}"
            except ValueError:
                pass

        if not month_year:
            try:
                df = pd.read_csv(matches_file, nrows=1)
                if 'on_start' in df.columns:
                    ts = pd.to_datetime(df['on_start'].iloc[0], format='%d/%m/%Y %H:%M', errors='coerce')
                    if pd.notna(ts):
                        month_year = ts.strftime('%Y-%m')
            except Exception:
                continue

        if not month_year or month_year not in monthly_stats:
            continue

        try:
            matches_df = pd.read_csv(matches_file)
        except Exception:
            continue

        monthly_stats[month_year]['total_matches'] = len(matches_df)

        if 'tag' in matches_df.columns:
            tag_counts = matches_df['tag'].value_counts().to_dict()
            monthly_stats[month_year]['matches_by_tag'] = tag_counts

        if 'on_magnitude' in matches_df.columns:
            monthly_stats[month_year]['matched_power'] = matches_df['on_magnitude'].abs().sum()

        # Add matched minutes (sum of all match durations)
        # Fix negative durations (events crossing midnight) by adding 1440 min
        if 'duration' in matches_df.columns:
            durations = matches_df['duration'].copy()
            durations = durations.apply(lambda x: x + 1440 if x < 0 else x)
            monthly_stats[month_year]['matched_minutes'] = float(durations.sum())
        elif 'on_start' in matches_df.columns and 'off_end' in matches_df.columns:
            # Fallback: calculate duration from timestamps (for older files without duration column)
            on_starts = pd.to_datetime(matches_df['on_start'], dayfirst=True)
            off_ends = pd.to_datetime(matches_df['off_end'], dayfirst=True)
            durations = (off_ends - on_starts).dt.total_seconds() / 60
            # Fix negative durations (events crossing midnight)
            durations = durations.apply(lambda x: x + 1440 if x < 0 else x)
            monthly_stats[month_year]['matched_minutes'] = float(durations.sum())

    # Convert to sorted list
    metrics['monthly_data'] = [
        monthly_stats[k] for k in sorted(monthly_stats.keys())
    ]

    # Calculate summary statistics
    if metrics['monthly_data']:
        matching_rates = [m.get('matching_rate', 0) for m in metrics['monthly_data']]
        metrics['avg_monthly_matching_rate'] = np.mean(matching_rates)
        metrics['min_monthly_matching_rate'] = np.min(matching_rates)
        metrics['max_monthly_matching_rate'] = np.max(matching_rates)
        metrics['std_monthly_matching_rate'] = np.std(matching_rates)

        # Find problematic months (matching rate < 40%)
        problematic = [m for m in metrics['monthly_data'] if m.get('matching_rate', 0) < 0.4]
        metrics['problematic_months'] = [m['month'] for m in problematic]
        metrics['n_problematic_months'] = len(problematic)

        # Find best months
        best = [m for m in metrics['monthly_data'] if m.get('matching_rate', 0) >= 0.6]
        metrics['best_months'] = [m['month'] for m in best]

    return metrics


def get_monthly_summary(experiment_dir: Path, house_id: str,
                        run_number: int = 0) -> str:
    """
    Generate a text summary of monthly performance.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number

    Returns:
        Formatted summary string
    """
    metrics = calculate_monthly_metrics(experiment_dir, house_id, run_number)

    lines = []
    lines.append(f"Monthly breakdown for House {house_id}:")
    lines.append("-" * 50)

    if not metrics.get('monthly_data'):
        lines.append("No monthly data available.")
        return '\n'.join(lines)

    # Monthly table
    lines.append(f"{'Month':<10} {'Events':<8} {'Matched':<8} {'Rate':<8} {'Matches':<8}")
    lines.append("-" * 50)

    for m in metrics['monthly_data']:
        month = m.get('month', '?')
        events = m.get('total_events', 0)
        matched = m.get('matched_events', 0)
        rate = m.get('matching_rate', 0)
        matches = m.get('total_matches', 0)

        # Mark problematic months
        marker = " *" if rate < 0.4 else ""
        lines.append(f"{month:<10} {events:<8} {matched:<8} {rate:.1%}{marker:<4} {matches:<8}")

    lines.append("-" * 50)

    # Summary
    if metrics.get('problematic_months'):
        lines.append(f"\n* Problematic months (<40%): {', '.join(metrics['problematic_months'])}")

    if metrics.get('best_months'):
        lines.append(f"Best months (>=60%): {', '.join(metrics['best_months'][:5])}")

    lines.append(f"\nMatching rate range: {metrics.get('min_monthly_matching_rate', 0):.1%} - "
                f"{metrics.get('max_monthly_matching_rate', 0):.1%}")

    return '\n'.join(lines)


def create_monthly_comparison_table(analyses: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a comparison table of monthly performance across houses.

    Args:
        analyses: List of analysis results (must include monthly_data)

    Returns:
        DataFrame with monthly comparison
    """
    rows = []

    for a in analyses:
        house_id = a.get('house_id', 'unknown')
        monthly_data = a.get('monthly_data', [])

        for m in monthly_data:
            rows.append({
                'house_id': house_id,
                'month': m.get('month'),
                'total_events': m.get('total_events', 0),
                'matched_events': m.get('matched_events', 0),
                'matching_rate': m.get('matching_rate', 0),
                'total_matches': m.get('total_matches', 0),
                'matched_power': m.get('matched_power', 0),
                'matched_minutes': m.get('matched_minutes', 0),
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.sort_values(['house_id', 'month'])

    return df


def find_common_problematic_months(analyses: List[Dict[str, Any]],
                                    threshold: float = 0.4) -> Dict[str, List[str]]:
    """
    Find months that are problematic across multiple houses.

    Args:
        analyses: List of analysis results
        threshold: Matching rate threshold for "problematic"

    Returns:
        Dict of month -> list of house_ids with issues
    """
    month_issues = {}

    for a in analyses:
        house_id = a.get('house_id', 'unknown')
        monthly_data = a.get('monthly_data', [])

        for m in monthly_data:
            if m.get('matching_rate', 0) < threshold:
                month = m.get('month')
                if month not in month_issues:
                    month_issues[month] = []
                month_issues[month].append(house_id)

    # Sort by number of affected houses
    return dict(sorted(month_issues.items(), key=lambda x: len(x[1]), reverse=True))
