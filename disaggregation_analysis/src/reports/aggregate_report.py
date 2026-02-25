"""
Aggregate report generation for multiple houses.

Combines experiment results across all houses into summary reports.
"""
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from reports.experiment_report import analyze_experiment_house
from metrics.monthly import calculate_monthly_metrics, find_common_problematic_months
import json

logger = logging.getLogger(__name__)


def load_pre_analysis_scores(house_analysis_path: Path) -> Dict[str, Any]:
    """
    Load quality scores from house_analysis output.

    Supports:
    - Single JSON file with list of analyses
    - Single JSON file with 'analyses' key
    - Directory containing per-house JSON files (per_house/ or run_*/per_house/)

    Args:
        house_analysis_path: Path to house_analysis JSON file or directory

    Returns:
        Dictionary mapping house_id -> dict with keys:
            - quality_score: float or 'faulty'
            - nan_continuity: str (continuous/minor_gaps/discontinuous/fragmented/unknown)
            - max_nan_pct: float
    """
    scores = {}
    house_analysis_path = Path(house_analysis_path)

    if not house_analysis_path.exists():
        print(f"Warning: Pre-analysis path not found: {house_analysis_path}")
        return scores

    def _extract_house_info(analysis: dict) -> Optional[tuple]:
        """Extract house_id and info dict from a single analysis."""
        house_id = str(analysis.get('house_id', ''))
        if not house_id:
            return None
        quality = analysis.get('data_quality', {})
        quality_label = quality.get('quality_label', None)
        quality_score = quality.get('quality_score', None)

        if quality_label and quality_label.startswith('faulty'):
            qs = quality_label  # 'faulty_dead_phase', 'faulty_high_nan', or 'faulty_both'
        elif quality_score is not None:
            qs = quality_score
        else:
            return None

        return house_id, {
            'quality_score': qs,
            'nan_continuity': quality.get('nan_continuity_label', 'unknown'),
            'max_nan_pct': quality.get('max_phase_nan_pct', 0),
        }

    # If it's a directory, look for per_house JSON files
    if house_analysis_path.is_dir():
        # Check for per_house subdirectory
        per_house_dir = house_analysis_path / "per_house"
        if not per_house_dir.exists():
            per_house_dir = house_analysis_path  # Maybe it IS the per_house dir

        json_files = list(per_house_dir.glob("analysis_*.json"))
        if not json_files:
            print(f"Warning: No analysis_*.json files found in {per_house_dir}")
            return scores

        files_iter = tqdm(json_files, desc="Loading pre-analysis", unit="file") if HAS_TQDM else json_files
        for json_file in files_iter:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)
                result = _extract_house_info(analysis)
                if result:
                    scores[result[0]] = result[1]
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")

        print(f"Loaded pre-analysis quality scores for {len(scores)} houses from {per_house_dir}")
        return scores

    # It's a file - try to read as JSON
    try:
        with open(house_analysis_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle list of analyses or dict with 'analyses' key
        if isinstance(data, list):
            analyses = data
        elif isinstance(data, dict) and 'analyses' in data:
            analyses = data['analyses']
        else:
            print(f"Warning: Unexpected format in {house_analysis_path}")
            return scores

        for analysis in analyses:
            result = _extract_house_info(analysis)
            if result:
                scores[result[0]] = result[1]

        print(f"Loaded pre-analysis quality scores for {len(scores)} houses")
    except Exception as e:
        print(f"Warning: Failed to load pre-analysis scores: {e}")

    return scores


def _analyze_single_house(args):
    """Worker function for parallel processing."""
    experiment_dir, house_id, max_iterations, fast_mode = args
    start_time = time.time()
    result = analyze_experiment_house(Path(experiment_dir), house_id, max_iterations, fast_mode=fast_mode)
    elapsed = time.time() - start_time
    result['_analysis_time'] = elapsed
    return result


def _save_house_report_incremental(analysis: Dict[str, Any], output_dir: Path) -> bool:
    """
    Save house report immediately after analysis completes.

    Saves JSON data only. HTML reports are generated at the end (after all houses complete)
    to avoid None formatting issues during incremental saving.

    Returns:
        True if save succeeded, False if it failed
    """
    house_id = analysis.get('house_id', 'unknown')

    # Save JSON analysis data
    json_path = output_dir / f"house_{house_id}_analysis.json"
    try:
        # Remove non-serializable fields
        serializable = {k: v for k, v in analysis.items() if not k.startswith('_')}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"    !! Failed to save JSON for house {house_id}: {e}", flush=True)
        return False


def aggregate_experiment_results(experiment_dir: Path,
                                 house_ids: Optional[List[str]] = None,
                                 max_iterations: int = 10,
                                 max_workers: int = 4,
                                 fast_mode: bool = False,
                                 pre_analysis_scores: Optional[Dict[str, Any]] = None,
                                 incremental_output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Aggregate experiment results for multiple houses.

    Args:
        experiment_dir: Path to experiment output directory
        house_ids: List of house IDs to analyze (None = auto-detect)
        max_iterations: Maximum iterations to check per house
        max_workers: Number of parallel workers
        fast_mode: Skip expensive pattern analysis for faster results
        pre_analysis_scores: Optional dict mapping house_id -> quality_score from house_analysis
        incremental_output_dir: Optional dir to save reports incrementally as each house completes

    Returns:
        List of analysis results for each house
    """
    logger.info("Starting aggregate experiment results for %s", experiment_dir)

    # Auto-detect houses if not specified
    if house_ids is None:
        house_ids = _detect_houses(experiment_dir)

    logger.info("Aggregating results for %d houses", len(house_ids))
    mode_str = " (FAST MODE)" if fast_mode else ""
    print(f"Detected {len(house_ids)} houses to analyze{mode_str}", flush=True)
    print(f"Houses: {', '.join(house_ids[:5])}{'...' if len(house_ids) > 5 else ''}", flush=True)

    if len(house_ids) == 0:
        return []

    # Prepare tasks
    tasks = [(str(experiment_dir), house_id, max_iterations, fast_mode) for house_id in house_ids]

    analyses = []
    completed_count = 0
    total_time = 0
    start_time = time.time()

    print(f"\nStarting analysis with {max_workers} parallel workers...", flush=True)
    print("-" * 60, flush=True)

    # Prepare incremental output directory if specified
    if incremental_output_dir:
        incremental_output_dir = Path(incremental_output_dir)
        incremental_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Incremental reports will be saved to: {incremental_output_dir}", flush=True)

    # Use parallel processing for speed
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyze_single_house, task): task[1] for task in tasks}

        if HAS_TQDM:
            futures_iter = tqdm(as_completed(futures), total=len(house_ids),
                               desc="Analyzing houses", unit="house")
        else:
            futures_iter = as_completed(futures)

        for future in futures_iter:
            house_id = futures[future]
            try:
                result = future.result()
                analyses.append(result)
                completed_count += 1
                house_time = result.get('_analysis_time', 0)
                total_time += house_time

                # === INCREMENTAL SAVE: Write report immediately after analysis ===
                if incremental_output_dir:
                    if result.get('status') == 'error':
                        print(f"    !! Skipping save for house {house_id}: status=error", flush=True)
                    else:
                        save_success = _save_house_report_incremental(result, incremental_output_dir)
                        if save_success:
                            print(f"    -> Saved reports for house {house_id}", flush=True)
                        # Errors are already logged by _save_house_report_incremental

                # Calculate progress stats
                elapsed = time.time() - start_time
                avg_time = elapsed / completed_count
                remaining = len(house_ids) - completed_count
                eta_seconds = avg_time * remaining

                # Log progress - verbose output
                status = result.get('status', 'unknown')
                score = result.get('scores', {}).get('overall_score', 0)

                # Build verbose details string
                details = []
                iterations = result.get('iterations', {})
                if iterations:
                    details.append(f"iters:{len(iterations)}")

                events = result.get('event_stats', {})
                if events:
                    on_count = events.get('on_events', 0)
                    matches = events.get('matches', 0)
                    if on_count:
                        details.append(f"events:{on_count}")
                    if matches:
                        details.append(f"matches:{matches}")

                ac_det = result.get('ac_detection', {})
                if ac_det.get('has_central_ac'):
                    details.append("central_ac")
                if ac_det.get('has_regular_ac'):
                    details.append("regular_ac")

                boiler = result.get('boiler_detection', {})
                if boiler.get('has_boiler'):
                    details.append("boiler")

                multi_phase = boiler.get('suspicious_multi_phase', {})
                if multi_phase.get('total_count', 0) > 0:
                    details.append(f"multi_phase:{multi_phase['total_count']}")

                details_str = " | ".join(details) if details else "no patterns"

                # Always print verbose output (flush=True for SLURM compatibility)
                print(f"  House {house_id}: {status} | score:{score:.0f} | {details_str} | {house_time:.1f}s", flush=True)

                if HAS_TQDM:
                    futures_iter.set_postfix(
                        last=house_id,
                        score=f"{score:.0f}",
                        eta=f"{eta_seconds/60:.1f}m"
                    )
            except Exception as e:
                completed_count += 1
                print(f"[{completed_count}/{len(house_ids)}] House {house_id}: ERROR - {e}", flush=True)
                analyses.append({'house_id': house_id, 'status': 'error', 'error': str(e)})

    total_elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Analysis complete: {len(analyses)} houses in {total_elapsed:.1f}s "
          f"(avg: {total_elapsed/len(analyses):.1f}s/house)")

    # Inject pre-analysis quality scores if provided
    if pre_analysis_scores:
        matched = 0
        for analysis in analyses:
            house_id = str(analysis.get('house_id', ''))
            if house_id in pre_analysis_scores:
                house_pre = pre_analysis_scores[house_id]
                if isinstance(house_pre, dict):
                    analysis['pre_analysis_quality_score'] = house_pre.get('quality_score')
                else:
                    # Backward compatibility: old format was scalar
                    analysis['pre_analysis_quality_score'] = house_pre
                matched += 1
        if matched > 0:
            print(f"Injected pre-analysis quality scores for {matched} houses")

    logger.info("Completed aggregate results for %d houses", len(analyses))

    return analyses


def _detect_houses(experiment_dir: Path) -> List[str]:
    """Detect house IDs from experiment directory structure."""
    house_ids = set()

    # Try new structure: experiment_dir/run_0/house_X/
    # Also supports dynamic threshold: experiment_dir/run_0_th2000/house_X/
    for run_dir in experiment_dir.glob("run_0*"):
        if run_dir.is_dir():
            for item in run_dir.iterdir():
                if item.is_dir() and item.name.startswith("house_"):
                    house_id = item.name.replace("house_", "")
                    house_ids.add(house_id)

    # Also check old structure: experiment_dir/house_X/run_0/...
    for item in experiment_dir.iterdir():
        if item.is_dir() and item.name.startswith("house_"):
            # Verify it has a run_0 subdirectory
            if (item / "run_0").exists():
                house_id = item.name.replace("house_", "")
                house_ids.add(house_id)

    return sorted(list(house_ids), key=lambda x: int(x) if x.isdigit() else x)


def generate_summary_report(analyses: List[Dict[str, Any]]) -> str:
    """
    Generate text summary report for all houses.

    Args:
        analyses: List of analysis results from aggregate_experiment_results

    Returns:
        Formatted summary text
    """
    logger.info("Generating summary report for %d analyses", len(analyses))

    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT SUMMARY REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Filter to houses with data
    valid_analyses = [a for a in analyses if a.get('status') != 'no_data']
    n_houses = len(valid_analyses)

    if n_houses == 0:
        lines.append("\nNo experiment data found for any house.")
        return '\n'.join(lines)

    # Summary statistics
    lines.append(f"\n--- Overview ---")
    lines.append(f"Houses analyzed: {n_houses}")

    # Average scores
    overall_scores = [a.get('scores', {}).get('overall_score', 0) for a in valid_analyses]
    matching_scores = [a.get('scores', {}).get('matching_score', 0) for a in valid_analyses]
    seg_scores = [a.get('scores', {}).get('segmentation_score', 0) for a in valid_analyses]

    lines.append(f"Average overall score: {sum(overall_scores)/n_houses:.0f}/100")
    lines.append(f"Average matching score: {sum(matching_scores)/n_houses:.0f}/100")
    lines.append(f"Average segmentation score: {sum(seg_scores)/n_houses:.0f}/100")

    # Iteration statistics
    iterations_completed = [a.get('iterations', {}).get('iterations_completed', 0)
                           for a in valid_analyses]
    lines.append(f"Average iterations: {sum(iterations_completed)/n_houses:.1f}")

    # Flag counts
    lines.append(f"\n--- Issues Summary ---")
    flag_counts = {}
    for a in valid_analyses:
        for flag, value in a.get('flags', {}).items():
            if value:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {flag.replace('_', ' ').title()}: {count} houses")

    # Houses with damaged phases
    damaged_houses = [a for a in valid_analyses if a.get('flags', {}).get('has_damaged_phases')]
    non_damaged_houses = [a for a in valid_analyses if not a.get('flags', {}).get('has_damaged_phases')]

    if damaged_houses:
        lines.append(f"\n--- Damaged Phases ---")
        lines.append(f"Houses with damaged phases: {len(damaged_houses)}")
        for a in damaged_houses[:5]:
            house_id = a.get('house_id', 'unknown')
            damaged = a.get('scores', {}).get('damaged_phases', [])
            score = a.get('scores', {}).get('overall_score', 0)
            lines.append(f"  House {house_id}: {', '.join(damaged)} damaged (score: {score:.0f})")
        if len(damaged_houses) > 5:
            lines.append(f"  ... and {len(damaged_houses) - 5} more")

    # Score tiers (from non-damaged houses only for fair comparison)
    lines.append(f"\n--- Score Distribution ---")
    non_damaged_scores = [a.get('scores', {}).get('overall_score', 0) for a in non_damaged_houses]

    if non_damaged_scores:
        excellent = sum(1 for s in non_damaged_scores if s >= 80)
        good = sum(1 for s in non_damaged_scores if 60 <= s < 80)
        fair = sum(1 for s in non_damaged_scores if 40 <= s < 60)
        poor = sum(1 for s in non_damaged_scores if s < 40)
    else:
        excellent = good = fair = poor = 0

    lines.append(f"  Damaged phases: {len(damaged_houses)} houses")
    lines.append(f"  Excellent (80+): {excellent} houses")
    lines.append(f"  Good (60-79): {good} houses")
    lines.append(f"  Fair (40-59): {fair} houses")
    lines.append(f"  Poor (<40): {poor} houses")

    # Best and worst performers
    sorted_by_score = sorted(valid_analyses,
                             key=lambda x: x.get('scores', {}).get('overall_score', 0),
                             reverse=True)

    lines.append(f"\n--- Top Performers ---")
    for a in sorted_by_score[:5]:
        house_id = a.get('house_id', 'unknown')
        score = a.get('scores', {}).get('overall_score', 0)
        damaged = a.get('scores', {}).get('damaged_phases', [])
        damaged_str = f" [damaged: {','.join(damaged)}]" if damaged else ""
        lines.append(f"  House {house_id}: {score:.0f}/100{damaged_str}")

    # Houses requiring focus - comprehensive list
    lines.append(f"\n--- Houses Requiring Focus ---")

    # 1. All houses with damaged phases
    if damaged_houses:
        lines.append(f"\n  DAMAGED PHASES ({len(damaged_houses)} houses):")
        for a in sorted(damaged_houses, key=lambda x: x.get('house_id', '0')):
            house_id = a.get('house_id', 'unknown')
            damaged = a.get('scores', {}).get('damaged_phases', [])
            healthy = a.get('scores', {}).get('healthy_phases', [])
            score = a.get('scores', {}).get('overall_score', 0)
            lines.append(f"    House {house_id}: {','.join(damaged)} damaged, "
                        f"healthy: {','.join(healthy)}, score: {score:.0f}")

    # 2. Poor performers (score < 40) without damaged phases
    poor_houses = [a for a in non_damaged_houses
                   if a.get('scores', {}).get('overall_score', 0) < 40]
    if poor_houses:
        lines.append(f"\n  POOR SCORE <40 ({len(poor_houses)} houses):")
        for a in sorted(poor_houses, key=lambda x: x.get('scores', {}).get('overall_score', 0)):
            house_id = a.get('house_id', 'unknown')
            score = a.get('scores', {}).get('overall_score', 0)
            matching = a.get('scores', {}).get('matching_score', 0)
            seg = a.get('scores', {}).get('segmentation_score', 0)
            flags = [k.replace('_', ' ') for k, v in a.get('flags', {}).items() if v]
            lines.append(f"    House {house_id}: {score:.0f} (match:{matching:.0f}, seg:{seg:.0f}) "
                        f"- {', '.join(flags[:3])}")

    # 3. Fair performers (40-60) that might benefit from algorithm improvements
    fair_houses = [a for a in non_damaged_houses
                   if 40 <= a.get('scores', {}).get('overall_score', 0) < 60]
    if fair_houses:
        lines.append(f"\n  FAIR SCORE 40-59 ({len(fair_houses)} houses):")
        for a in sorted(fair_houses, key=lambda x: x.get('scores', {}).get('overall_score', 0))[:10]:
            house_id = a.get('house_id', 'unknown')
            score = a.get('scores', {}).get('overall_score', 0)
            matching = a.get('scores', {}).get('matching_score', 0)
            seg = a.get('scores', {}).get('segmentation_score', 0)
            lines.append(f"    House {house_id}: {score:.0f} (match:{matching:.0f}, seg:{seg:.0f})")
        if len(fair_houses) > 10:
            lines.append(f"    ... and {len(fair_houses) - 10} more")

    # Summary of focus areas
    lines.append(f"\n  SUMMARY:")
    lines.append(f"    Total houses requiring focus: {len(damaged_houses) + len(poor_houses)}")
    lines.append(f"    - Damaged phases: {len(damaged_houses)}")
    lines.append(f"    - Poor score: {len(poor_houses)}")
    lines.append(f"    - Fair score (improvement potential): {len(fair_houses)}")

    logger.info("Completed summary report for %d houses", n_houses)

    return '\n'.join(lines)


def generate_monthly_analysis(experiment_dir: Path,
                               analyses: List[Dict[str, Any]],
                               max_houses: int = 20) -> str:
    """
    Generate monthly breakdown analysis for all houses.

    Args:
        experiment_dir: Path to experiment directory
        analyses: List of analysis results
        max_houses: Maximum houses to include in detailed monthly analysis

    Returns:
        Formatted monthly analysis report
    """
    logger.info("Generating monthly analysis for %d analyses (max_houses=%d)", len(analyses), max_houses)

    lines = []
    lines.append("=" * 70)
    lines.append("MONTHLY ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Calculate monthly metrics for each house
    monthly_analyses = []
    house_ids = [a.get('house_id') for a in analyses if a.get('status') != 'no_data']

    for house_id in house_ids[:max_houses]:
        monthly_data = calculate_monthly_metrics(experiment_dir, house_id, 0)
        if monthly_data.get('monthly_data'):
            monthly_analyses.append(monthly_data)

    if not monthly_analyses:
        lines.append("\nNo monthly data available.")
        return '\n'.join(lines)

    # Find common problematic months
    problematic_months = find_common_problematic_months(monthly_analyses, threshold=0.4)

    if problematic_months:
        lines.append(f"\n--- Problematic Months (matching rate < 40%) ---")
        for month, house_ids in list(problematic_months.items())[:10]:
            n_houses = len(house_ids)
            pct = n_houses / len(monthly_analyses) * 100
            lines.append(f"  {month}: {n_houses} houses ({pct:.0f}%) - {', '.join(house_ids[:5])}"
                        f"{'...' if len(house_ids) > 5 else ''}")

    # Monthly averages across all houses
    all_months = {}
    for ma in monthly_analyses:
        for m in ma.get('monthly_data', []):
            month = m.get('month')
            if month not in all_months:
                all_months[month] = {'rates': [], 'events': [], 'matches': []}
            all_months[month]['rates'].append(m.get('matching_rate', 0))
            all_months[month]['events'].append(m.get('total_events', 0))
            all_months[month]['matches'].append(m.get('total_matches', 0))

    if all_months:
        lines.append(f"\n--- Monthly Averages Across Houses ---")
        lines.append(f"{'Month':<10} {'Avg Rate':<10} {'Avg Events':<12} {'Avg Matches':<12} {'Houses':<8}")
        lines.append("-" * 55)

        for month in sorted(all_months.keys()):
            data = all_months[month]
            avg_rate = np.mean(data['rates'])
            avg_events = np.mean(data['events'])
            avg_matches = np.mean(data['matches'])
            n_houses = len(data['rates'])

            marker = " *" if avg_rate < 0.4 else ""
            lines.append(f"{month:<10} {avg_rate:.1%}{marker:<5} {avg_events:<12.0f} {avg_matches:<12.0f} {n_houses:<8}")

    # Houses with high variation across months
    lines.append(f"\n--- Houses with High Monthly Variation ---")
    high_var_houses = []
    for ma in monthly_analyses:
        if ma.get('std_monthly_matching_rate', 0) > 0.15:  # > 15% std
            high_var_houses.append({
                'house_id': ma.get('house_id'),
                'std': ma.get('std_monthly_matching_rate', 0),
                'min': ma.get('min_monthly_matching_rate', 0),
                'max': ma.get('max_monthly_matching_rate', 0),
                'problematic': ma.get('problematic_months', []),
            })

    if high_var_houses:
        high_var_houses.sort(key=lambda x: x['std'], reverse=True)
        for h in high_var_houses[:10]:
            lines.append(f"  House {h['house_id']}: {h['min']:.0%} - {h['max']:.0%} "
                        f"(std: {h['std']:.0%})")
            if h['problematic']:
                lines.append(f"    Problematic: {', '.join(h['problematic'][:3])}")
    else:
        lines.append("  No houses with high variation (std > 15%)")

    logger.info("Completed monthly analysis report")

    return '\n'.join(lines)


def create_comparison_table(analyses: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create comparison DataFrame for all houses.

    Args:
        analyses: List of analysis results

    Returns:
        DataFrame with comparison metrics
    """
    logger.info("Creating comparison table for %d analyses", len(analyses))

    rows = []

    for a in analyses:
        if a.get('status') == 'no_data':
            continue

        iterations = a.get('iterations', {})
        scores = a.get('scores', {})
        first = a.get('first_iteration', {})
        matching = first.get('matching', {})
        seg = first.get('segmentation', {})
        flags = a.get('flags', {})

        # Get damaged phases info
        damaged_phases = scores.get('damaged_phases', [])
        healthy_phases = scores.get('healthy_phases', [])

        # Determine category
        overall_score = scores.get('overall_score', 0)
        has_damaged = flags.get('has_damaged_phases', False)
        if has_damaged:
            category = 'DAMAGED'
        elif overall_score >= 80:
            category = 'EXCELLENT'
        elif overall_score >= 60:
            category = 'GOOD'
        elif overall_score >= 40:
            category = 'FAIR'
        else:
            category = 'POOR'

        row = {
            'house_id': a.get('house_id'),
            'category': category,
            'overall_score': overall_score,
            'matching_score': scores.get('matching_score', 0),
            'segmentation_score': scores.get('segmentation_score', 0),
            'iterations': iterations.get('iterations_completed', 0),
            'first_events': iterations.get('first_iter_events', 0),
            'last_events': iterations.get('last_iter_events', 0),
            'events_reduction': iterations.get('events_reduction_ratio', 0),
            'first_matching_rate': iterations.get('first_iter_matching_rate', 0),
            'last_matching_rate': iterations.get('last_iter_matching_rate', 0),
            'total_matches': matching.get('total_matches', 0),
            'non_m_matches': matching.get('tag_breakdown', {}).get('NON-M', 0),
            'noisy_matches': matching.get('tag_breakdown', {}).get('NOISY', 0),
            'partial_matches': matching.get('tag_breakdown', {}).get('PARTIAL', 0),
            'segmentation_ratio': seg.get('segmentation_ratio', 0),
            'healthy_seg_ratio': scores.get('avg_3phase_segmentation_ratio', seg.get('segmentation_ratio', 0)),
            'total_power_kw': seg.get('total_power', 0) / 1000,
            'segmented_power_kw': seg.get('total_segmented_power', 0) / 1000,
            'negative_values': seg.get('negative_value_count', 0),
            'n_flags': sum(1 for v in flags.values() if v),
            'has_damaged_phases': has_damaged,
            'damaged_phases': ','.join(damaged_phases) if damaged_phases else '',
            'n_healthy_phases': len(healthy_phases),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if not df.empty:
        # Sort by category priority (DAMAGED first, then POOR, FAIR, GOOD, EXCELLENT)
        # then by score within each category
        category_order = {'DAMAGED': 0, 'POOR': 1, 'FAIR': 2, 'GOOD': 3, 'EXCELLENT': 4}
        df['_category_order'] = df['category'].map(category_order)
        df = df.sort_values(['_category_order', 'overall_score'], ascending=[True, True])
        df = df.drop('_category_order', axis=1)

    logger.info("Completed comparison table with %d rows", len(df))

    return df


def get_focus_houses(analyses: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Get a focused list of houses that need attention.

    Returns houses with:
    - Damaged phases
    - Poor score (<40)
    - Fair score (40-60) for improvement potential

    Args:
        analyses: List of analysis results

    Returns:
        DataFrame with focus houses sorted by priority
    """
    logger.info("Identifying focus houses from %d analyses", len(analyses))

    focus_rows = []

    for a in analyses:
        if a.get('status') == 'no_data':
            continue

        scores = a.get('scores', {})
        flags = a.get('flags', {})
        iterations = a.get('iterations', {})

        overall_score = scores.get('overall_score', 0)
        has_damaged = flags.get('has_damaged_phases', False)
        damaged_phases = scores.get('damaged_phases', [])
        healthy_phases = scores.get('healthy_phases', [])

        # Determine priority and reason
        if has_damaged:
            priority = 1
            reason = f"Damaged phases: {','.join(damaged_phases)}"
            category = 'DAMAGED'
        elif overall_score < 40:
            priority = 2
            reason = "Low overall score"
            category = 'POOR'
        elif overall_score < 60:
            priority = 3
            reason = "Improvement potential"
            category = 'FAIR'
        else:
            continue  # Skip good/excellent houses

        # Collect issues
        issues = []
        if flags.get('low_matching_rate'):
            issues.append('low_match')
        if flags.get('low_segmentation'):
            issues.append('low_seg')
        if flags.get('has_negative_values'):
            issues.append('negatives')
        if flags.get('few_matches'):
            issues.append('few_matches')
        if flags.get('no_improvement'):
            issues.append('no_improve')

        focus_rows.append({
            'house_id': a.get('house_id'),
            'priority': priority,
            'category': category,
            'reason': reason,
            'overall_score': overall_score,
            'matching_score': scores.get('matching_score', 0),
            'segmentation_score': scores.get('segmentation_score', 0),
            'matching_rate': iterations.get('first_iter_matching_rate', 0),
            'damaged_phases': ','.join(damaged_phases) if damaged_phases else '',
            'healthy_phases': ','.join(healthy_phases) if healthy_phases else 'w1,w2,w3',
            'issues': ','.join(issues) if issues else '',
        })

    df = pd.DataFrame(focus_rows)

    if not df.empty:
        df = df.sort_values(['priority', 'overall_score'], ascending=[True, True])

    logger.info("Completed focus houses: %d houses need attention", len(df))

    return df
