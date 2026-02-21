"""
Single house experiment report generation.

Analyzes experiment results for a single house across all iterations.
"""
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from metrics.matching import calculate_matching_metrics, _get_house_dir, _load_monthly_files
from metrics.segmentation import (
    calculate_segmentation_metrics,
    calculate_threshold_explanation_metrics,
    calculate_threshold_explanation_all_iterations
)
from metrics.events import calculate_event_metrics
from metrics.iterations import calculate_iteration_metrics
from metrics.patterns import calculate_pattern_metrics, detect_ac_patterns, detect_boiler_patterns, analyze_device_usage_patterns
from metrics.monthly import calculate_monthly_metrics

# Enable verbose logging for debugging
_VERBOSE = False


def _load_house_run_data(experiment_dir: Path, house_id: str, run_number: int = 0) -> Dict[str, Any]:
    """
    Load all data files for a house run once.

    Returns a dict with pre-loaded DataFrames that can be passed to metric functions
    via the 'preloaded' parameter, avoiding redundant file I/O.

    Returns:
        Dict with keys: 'on_off', 'matches', 'unmatched_on', 'unmatched_off', 'house_dir'
    """
    house_dir = _get_house_dir(experiment_dir, house_id, run_number)

    data = {'house_dir': house_dir}

    # Load on_off events (used by: matching, events, patterns)
    data['on_off'] = _load_monthly_files(house_dir, "on_off", "on_off_*.pkl")

    # Load matches (used by: matching, patterns, AC detection, boiler detection)
    data['matches'] = _load_monthly_files(house_dir, "matches", f"matches_{house_id}_*.pkl")

    # Load unmatched (used by: matching)
    data['unmatched_on'] = _load_monthly_files(house_dir, "unmatched_on", f"unmatched_on_{house_id}_*.pkl")
    data['unmatched_off'] = _load_monthly_files(house_dir, "unmatched_off", f"unmatched_off_{house_id}_*.pkl")

    # Parse timestamps once for all consumers
    if data['on_off'] is not None:
        for col in ['start', 'end']:
            if col in data['on_off'].columns:
                data['on_off'][col] = pd.to_datetime(data['on_off'][col], format='mixed', dayfirst=True, errors='coerce')

    if data['matches'] is not None:
        for col in ['on_start', 'on_end', 'off_start', 'off_end']:
            if col in data['matches'].columns:
                data['matches'][col] = pd.to_datetime(data['matches'][col], format='mixed', dayfirst=True, errors='coerce')

    return data


def load_experiment_data(experiment_dir: Path, house_id: str,
                         run_number: int = 0) -> Dict[str, pd.DataFrame]:
    """
    Load all experiment output files for a house.

    Supports both old structure (files directly in house_dir) and new monthly
    folder structure (files in subfolders like on_off/, matches/, summarized/).

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Iteration number (default 0)

    Returns:
        Dictionary with DataFrames for each file type
    """
    # Structure: experiment_dir/run_N/house_X/ (new), run_N_thXXXX/house_X/ (dynamic), or old
    from metrics.iterations import _find_run_dir
    run_dir = _find_run_dir(experiment_dir, run_number)
    if run_dir is not None:
        house_dir = run_dir / f"house_{house_id}"
    else:
        house_dir = experiment_dir / f"run_{run_number}" / f"house_{house_id}"
    if not house_dir.exists():
        # Fall back to old structure
        house_dir = experiment_dir / f"house_{house_id}" / f"run_{run_number}" / f"house_{house_id}"

    data = {}

    if not house_dir.exists():
        return data

    # Load on_off events - check both new subfolder and old direct location
    on_off_dir = house_dir / "on_off"
    if on_off_dir.exists():
        on_off_files = sorted(on_off_dir.glob("on_off_*.csv"))
        if on_off_files:
            data['on_off'] = pd.concat([pd.read_csv(f) for f in on_off_files], ignore_index=True)
    else:
        on_off_files = list(house_dir.glob("on_off_*.csv"))
        if on_off_files:
            data['on_off'] = pd.read_csv(on_off_files[0])

    # Load matches - check both new subfolder and old direct location
    matches_dir = house_dir / "matches"
    if matches_dir.exists():
        matches_files = sorted(matches_dir.glob(f"matches_{house_id}_*.csv"))
        if matches_files:
            data['matches'] = pd.concat([pd.read_csv(f) for f in matches_files], ignore_index=True)
    else:
        matches_files = list(house_dir.glob("matches_*.csv"))
        if matches_files:
            data['matches'] = pd.read_csv(matches_files[0])

    # Load summarized data - check both new subfolder and old direct location
    summarized_dir = house_dir / "summarized"
    if summarized_dir.exists():
        summarized_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.csv"))
        if summarized_files:
            data['summarized'] = pd.concat([pd.read_csv(f) for f in summarized_files], ignore_index=True)
    else:
        # Try old segmented file
        segmented_files = list(house_dir.glob("segmented_*.csv"))
        if segmented_files:
            data['segmented'] = pd.read_csv(segmented_files[0])

    # Load remainder events if exists
    remainder_files = list(house_dir.glob("remainder_*.csv"))
    if remainder_files:
        data['remainder'] = pd.read_csv(remainder_files[0])

    return data


def analyze_experiment_house(experiment_dir: Path, house_id: str,
                             max_iterations: int = 10,
                             verbose: bool = False,
                             fast_mode: bool = False) -> Dict[str, Any]:
    """
    Perform complete analysis of experiment results for a house.

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        max_iterations: Maximum iterations to analyze
        verbose: Print progress logs
        fast_mode: Skip expensive pattern analysis for faster results

    Returns:
        Dictionary with all analysis results
    """
    def log(msg):
        if verbose or _VERBOSE:
            print(f"  [House {house_id}] {msg}")

    analysis = {
        'house_id': house_id,
        'experiment_dir': str(experiment_dir),
    }

    # Get iteration overview first
    log("Calculating iteration metrics...")
    t0 = time.time()
    iteration_metrics = calculate_iteration_metrics(
        experiment_dir, house_id, max_iterations, verbose=False
    )
    analysis['iterations'] = iteration_metrics
    log(f"Iteration metrics done ({time.time()-t0:.2f}s)")

    iterations_completed = iteration_metrics.get('iterations_completed', 0)

    if iterations_completed == 0:
        analysis['status'] = 'no_data'
        return analysis

    analysis['status'] = 'complete'

    # Load all data for run_0 once (avoids redundant file I/O)
    log("Loading run_0 data...")
    t0 = time.time()
    preloaded = _load_house_run_data(experiment_dir, house_id, 0)
    log(f"Data loaded ({time.time()-t0:.2f}s)")

    # Analyze first iteration (run_0)
    log("Analyzing first iteration...")
    t0 = time.time()

    log("  - matching metrics...")
    matching = calculate_matching_metrics(experiment_dir, house_id, 0, preloaded=preloaded)

    log("  - segmentation metrics...")
    segmentation = calculate_segmentation_metrics(experiment_dir, house_id, 0)

    log("  - event metrics...")
    events = calculate_event_metrics(experiment_dir, house_id, 0, preloaded=preloaded)

    # Compute damaged phases ONCE (used by AC detection, flags, and scores)
    damaged_info = _detect_damaged_phases_from_segmentation(segmentation)

    # Pattern metrics are expensive - skip in fast mode
    if not fast_mode:
        log("  - pattern metrics...")
        patterns = calculate_pattern_metrics(experiment_dir, house_id, 0, preloaded=preloaded)

        # Detect AC patterns (central and regular)
        log("  - AC detection...")
        ac_patterns = detect_ac_patterns(
            experiment_dir, house_id, 0,
            damaged_phases=damaged_info.get('damaged_phases', []),
            preloaded=preloaded
        )
        patterns['ac_detection'] = ac_patterns

        # Detect boiler patterns (long isolated high-power events)
        log("  - boiler detection...")
        boiler_patterns = detect_boiler_patterns(experiment_dir, house_id, 0, preloaded=preloaded)
        patterns['boiler_detection'] = boiler_patterns

        # Merge reclassified AC events (boiler candidates that turned out to be AC)
        reclassified = boiler_patterns.get('reclassified_as_ac', {}).get('activations', [])
        if reclassified:
            log(f"  - {len(reclassified)} boiler candidates reclassified as AC")
            existing_regular = ac_patterns.get('regular_ac', {}).get('activations', [])
            existing_regular.extend(reclassified)
            existing_regular.sort(key=lambda x: (x['date'], x['on_time']))
            ac_patterns['regular_ac']['activations'] = existing_regular
            ac_patterns['regular_ac']['total_count'] = len(existing_regular)
            if not ac_patterns.get('has_regular_ac') and len(existing_regular) >= 3:
                ac_patterns['has_regular_ac'] = True
            patterns['ac_detection'] = ac_patterns

        # Analyze device usage patterns (seasonal and time of day)
        log("  - device usage patterns...")
        device_usage = analyze_device_usage_patterns(ac_patterns, boiler_patterns)
        patterns['device_usage'] = device_usage
    else:
        patterns = {}

    analysis['first_iteration'] = {
        'matching': matching,
        'segmentation': segmentation,
        'events': events,
        'patterns': patterns,
    }
    log(f"First iteration done ({time.time()-t0:.2f}s)")

    # Classification metrics (dynamic threshold experiments only)
    _add_classification_if_available(analysis, experiment_dir, house_id, log)

    # NOTE: last_iteration and progression calculations removed - not displayed in HTML
    # This saves ~50% of analysis time per house

    # Calculate monthly breakdown
    if not fast_mode:
        log("  - monthly metrics...")
        monthly_metrics = calculate_monthly_metrics(experiment_dir, house_id, 0)
        analysis['monthly'] = monthly_metrics

    # Calculate threshold explanation metrics (using 1300W threshold from pipeline config)
    log("  - threshold explanation metrics...")
    threshold_metrics = calculate_threshold_explanation_metrics(
        experiment_dir, house_id, 0, threshold=1300
    )
    analysis['threshold_explanation'] = threshold_metrics

    # Calculate threshold explanation per iteration
    log("  - threshold explanation per iteration...")
    threshold_per_iter = calculate_threshold_explanation_all_iterations(
        experiment_dir, house_id, max_iterations=10, threshold=1300
    )
    analysis['threshold_explanation_per_iteration'] = threshold_per_iter

    # Generate flags for easy filtering (pass pre-computed damaged_info)
    analysis['flags'] = _generate_experiment_flags(analysis, damaged_info=damaged_info)

    # Calculate overall scores (pass pre-computed damaged_info)
    analysis['scores'] = _calculate_experiment_scores(analysis, damaged_info=damaged_info)

    return analysis


def _add_classification_if_available(analysis: Dict[str, Any], experiment_dir: Path,
                                      house_id: str, log) -> None:
    """Add device classification metrics if classification data exists (dynamic threshold experiments)."""
    import json

    # Check if any run directory has classification data
    has_classification = False
    for d in experiment_dir.glob("run_*_th*/house_*/classification"):
        if d.is_dir() and any(d.glob("classification_*.pkl")):
            has_classification = True
            break

    if not has_classification:
        return

    try:
        log("  - classification metrics (dynamic threshold)...")
        from metrics.classification import calculate_classification_metrics
        classification = calculate_classification_metrics(experiment_dir, house_id)
        analysis['classification'] = classification

        # Load activation list if it exists
        activation_path = experiment_dir / "activation_lists" / f"activation_list_{house_id}.json"
        # Fallback: check legacy location (experiment root)
        if not activation_path.exists():
            activation_path = experiment_dir / f"activation_list_{house_id}.json"
        if activation_path.exists():
            with open(activation_path, 'r') as f:
                analysis['activation_list'] = json.load(f)
    except Exception as e:
        log(f"  - classification metrics failed: {e}")


def _detect_damaged_phases_from_segmentation(seg: Dict[str, Any], threshold_ratio: float = 0.01) -> Dict[str, Any]:
    """
    Detect damaged phases directly from segmentation data.

    Args:
        seg: Segmentation metrics dictionary
        threshold_ratio: Minimum ratio compared to max phase (default 0.01 = 1%)

    Returns:
        Dictionary with damaged_phases and healthy_phases lists
    """
    result = {
        'damaged_phases': [],
        'healthy_phases': [],
    }

    phases = ['w1', 'w2', 'w3']
    phase_powers = {}

    for phase in phases:
        power = seg.get(f'{phase}_total_power', 0)
        phase_powers[phase] = power

    if not phase_powers or all(p == 0 for p in phase_powers.values()):
        result['healthy_phases'] = phases
        return result

    max_power = max(phase_powers.values())

    if max_power == 0:
        result['healthy_phases'] = phases
        return result

    for phase, power in phase_powers.items():
        ratio = power / max_power if max_power > 0 else 0
        if ratio < threshold_ratio:
            result['damaged_phases'].append(phase)
        else:
            result['healthy_phases'].append(phase)

    return result


def _detect_damaged_phases(analysis: Dict[str, Any], threshold_ratio: float = 0.01) -> Dict[str, Any]:
    """
    Detect damaged phases based on power distribution.

    A phase is considered damaged if its total power is less than threshold_ratio
    (default 1%) of the maximum phase's power.

    Args:
        analysis: Analysis results containing segmentation data
        threshold_ratio: Minimum ratio compared to max phase (default 0.01 = 1%)

    Returns:
        Dictionary with:
        - damaged_phases: list of damaged phase names
        - healthy_phases: list of healthy phase names
        - phase_powers: dict of phase -> total_power
        - has_damaged_phases: boolean
    """
    result = {
        'damaged_phases': [],
        'healthy_phases': [],
        'phase_powers': {},
        'has_damaged_phases': False
    }

    first = analysis.get('first_iteration', {})
    seg = first.get('segmentation', {})

    # Get per-phase total power
    phases = ['w1', 'w2', 'w3']
    phase_powers = {}

    for phase in phases:
        power = seg.get(f'{phase}_total_power', 0)
        phase_powers[phase] = power

    result['phase_powers'] = phase_powers

    # Find max power among phases
    if not phase_powers or all(p == 0 for p in phase_powers.values()):
        # No power data - cannot determine damaged phases
        result['healthy_phases'] = phases
        return result

    max_power = max(phase_powers.values())

    if max_power == 0:
        result['healthy_phases'] = phases
        return result

    # Classify phases
    for phase, power in phase_powers.items():
        ratio = power / max_power if max_power > 0 else 0
        if ratio < threshold_ratio:
            result['damaged_phases'].append(phase)
        else:
            result['healthy_phases'].append(phase)

    result['has_damaged_phases'] = len(result['damaged_phases']) > 0

    return result


def _generate_experiment_flags(analysis: Dict[str, Any],
                               damaged_info: Dict[str, Any] = None) -> Dict[str, bool]:
    """Generate boolean flags for experiment issues."""
    flags = {}

    iterations = analysis.get('iterations', {})
    first = analysis.get('first_iteration', {})

    # Low matching rate
    first_rate = iterations.get('first_iter_matching_rate', 0)
    flags['low_matching_rate'] = first_rate < 0.5

    # Low improvement across iterations
    # Calculate as: additional power segmented by iterations 1+ / total power
    # Flag appears if additional iterations contributed less than 5% of total data
    iterations_completed = iterations.get('iterations_completed', 0)
    if iterations_completed > 1:
        iterations_data = iterations.get('iterations_data', [])
        seg = first.get('segmentation', {})
        total_power = seg.get('total_power', 0)

        if iterations_data and total_power > 0:
            # Matched power from first iteration only
            first_iter_matched = iterations_data[0].get('matched_power', 0) if iterations_data else 0
            # Total matched power across ALL iterations
            total_matched = iterations.get('total_matched_power', 0)
            # Additional contribution from iterations 1+
            additional_matched = total_matched - first_iter_matched

            additional_contribution = additional_matched / total_power
            flags['low_improvement'] = additional_contribution < 0.05
        else:
            flags['low_improvement'] = False
    else:
        flags['low_improvement'] = False

    # Has negative values (segmentation error)
    flags['has_negative_values'] = iterations.get('has_negative_values', False)

    # Low segmentation ratio
    seg = first.get('segmentation', {})
    seg_ratio = seg.get('segmentation_ratio', 0)
    flags['low_segmentation'] = seg_ratio < 0.3

    # Few events matched
    matching = first.get('matching', {})
    flags['few_matches'] = matching.get('total_matches', 0) < 10

    # High partial match ratio (might indicate calibration issues)
    tag_breakdown = matching.get('tag_breakdown', {})
    total_matches = matching.get('total_matches', 1)
    partial_count = tag_breakdown.get('PARTIAL', 0)
    flags['high_partial_ratio'] = (partial_count / total_matches) > 0.3 if total_matches > 0 else False

    # Many remainder events
    flags['many_remainders'] = matching.get('remainder_events', 0) > 20

    # Damaged phases detection (use pre-computed if available)
    if damaged_info is None:
        damaged_info = _detect_damaged_phases(analysis)
    flags['has_damaged_phases'] = len(damaged_info.get('damaged_phases', [])) > 0

    # Recurring patterns detection (positive flag)
    patterns = first.get('patterns', {})
    recurring_matches = patterns.get('recurring_matches', {})
    all_patterns = recurring_matches.get('patterns', [])
    # Count frequent patterns (every 1-10 days) with duration > 20 min
    frequent_patterns = [
        p for p in all_patterns
        if (p.get('duration_minutes', 0) > 20) and (
            p.get('interval_type') in ('daily', 'weekly')
            or (p.get('avg_interval_days') is not None and p.get('avg_interval_days') <= 10)
        )
    ]
    flags['has_recurring_patterns'] = len(frequent_patterns) >= 3

    return flags


def _calculate_experiment_scores(analysis: Dict[str, Any],
                                 damaged_info: Dict[str, Any] = None) -> Dict[str, float]:
    """
    Calculate experiment quality scores.

    Phase averaging: all metrics are averaged across ALL 3 phases.
    Damaged phases count as 0 (not excluded), so a house with 1 healthy phase
    scoring 75% gets (0 + 0 + 75) / 3 = 25%, reflecting that only 1/3 of
    the electrical system is being analyzed.
    """
    scores = {}

    iterations = analysis.get('iterations', {})
    first = analysis.get('first_iteration', {})
    flags = analysis.get('flags', {})

    # Detect damaged phases (use pre-computed if available)
    if damaged_info is None:
        damaged_info = _detect_damaged_phases(analysis)
    scores['damaged_phases'] = damaged_info.get('damaged_phases', [])
    scores['healthy_phases'] = damaged_info.get('healthy_phases', ['w1', 'w2', 'w3'])

    # Matching score (0-100)
    first_rate = iterations.get('first_iter_matching_rate', 0)
    matching_score = first_rate * 100

    # Bonus for improvement
    rate_change = iterations.get('matching_rate_change', 0)
    if rate_change > 0:
        matching_score = min(100, matching_score + rate_change * 50)

    scores['matching_score'] = matching_score

    # Segmentation score (0-100)
    # Average across ALL 3 phases (damaged phases = 0)
    seg = first.get('segmentation', {})
    all_phases = ['w1', 'w2', 'w3']
    damaged_phases = set(damaged_info.get('damaged_phases', []))

    phase_seg_ratios = []
    for phase in all_phases:
        if phase in damaged_phases:
            phase_seg_ratios.append(0.0)  # Damaged phase counts as 0
        else:
            phase_seg_ratios.append(seg.get(f'{phase}_segmentation_ratio', 0) or 0)

    # 3-phase average (damaged = 0, not excluded)
    avg_seg_ratio = sum(phase_seg_ratios) / 3
    scores['avg_3phase_segmentation_ratio'] = avg_seg_ratio
    segmentation_score = avg_seg_ratio * 100

    # Penalty for negative values
    if flags.get('has_negative_values'):
        neg_count = iterations.get('total_negative_values', 0)
        segmentation_score -= min(20, neg_count * 0.5)

    scores['segmentation_score'] = max(0, segmentation_score)

    # Overall score - matching weighted 70%, segmentation 30%
    scores['overall_score'] = (
        scores['matching_score'] * 0.7 +
        scores['segmentation_score'] * 0.3
    )

    return scores


def generate_experiment_report(analysis: Dict[str, Any]) -> str:
    """
    Generate text report for a single house experiment.

    Args:
        analysis: Analysis results from analyze_experiment_house

    Returns:
        Formatted text report
    """
    lines = []
    house_id = analysis.get('house_id', 'unknown')

    lines.append(f"=" * 60)
    lines.append(f"EXPERIMENT REPORT: House {house_id}")
    lines.append(f"=" * 60)

    if analysis.get('status') == 'no_data':
        lines.append("\nNo experiment data found for this house.")
        return '\n'.join(lines)

    # Iterations overview
    iterations = analysis.get('iterations', {})
    lines.append(f"\n--- Iterations Overview ---")
    lines.append(f"Iterations completed: {iterations.get('iterations_completed', 0)}")
    lines.append(f"First iteration: {iterations.get('first_iter_events', 0)} events, "
                f"{iterations.get('first_iter_matching_rate', 0):.1%} matched")
    lines.append(f"Last iteration: {iterations.get('last_iter_events', 0)} events, "
                f"{iterations.get('last_iter_matching_rate', 0):.1%} matched")

    if 'events_reduction_ratio' in iterations:
        lines.append(f"Events reduction: {iterations['events_reduction_ratio']:.1%}")
    if 'total_matched_power' in iterations:
        lines.append(f"Total matched power: {iterations['total_matched_power']/1000:.1f} kW")

    # First iteration details
    first = analysis.get('first_iteration', {})
    if first:
        matching = first.get('matching', {})
        lines.append(f"\n--- First Iteration Matching ---")
        lines.append(f"Total events: {matching.get('total_on_events', 0)} ON, "
                    f"{matching.get('total_off_events', 0)} OFF")
        matched_on = matching.get('matched_on_events', 0) or 0
        on_rate = matching.get('on_matching_rate', 0) or 0
        lines.append(f"Matched: {matched_on} ({on_rate:.1%})")

        tag_breakdown = matching.get('tag_breakdown', {})
        if tag_breakdown:
            lines.append(f"By tag: " + ", ".join(f"{k}: {v}" for k, v in tag_breakdown.items()))

        seg = first.get('segmentation', {})
        lines.append(f"\n--- First Iteration Segmentation ---")
        lines.append(f"Total power: {seg.get('total_power', 0)/1000:.1f} kW")
        lines.append(f"Segmented: {seg.get('total_segmented_power', 0)/1000:.1f} kW "
                    f"({seg.get('segmentation_ratio', 0):.1%})")
        lines.append(f"Remaining: {seg.get('total_remaining_power', 0)/1000:.1f} kW")

        neg_count = seg.get('negative_value_count', 0)
        if neg_count > 0:
            lines.append(f"WARNING: {neg_count} negative values detected!")

    # High-Power Energy Segregated
    th_expl = analysis.get('threshold_explanation', {})
    if th_expl and 'total_minutes_above_th' in th_expl:
        lines.append(f"\n--- High-Power Energy Segregated (>{th_expl.get('threshold', 1300)}W) ---")
        lines.append(f"Total minutes above threshold: {th_expl.get('total_minutes_above_th', 0):,}")
        lines.append(f"Minutes segregated: {th_expl.get('total_minutes_explained', 0):,} "
                    f"({th_expl.get('total_explanation_rate', 0):.1%})")
        for phase in ['w1', 'w2', 'w3']:
            above = th_expl.get(f'{phase}_minutes_above_th', 0)
            explained = th_expl.get(f'{phase}_minutes_explained', 0)
            rate = th_expl.get(f'{phase}_explanation_rate', 0)
            lines.append(f"  {phase}: {above:,} above TH, {explained:,} segregated ({rate:.1%})")

    # Scores
    scores = analysis.get('scores', {})
    lines.append(f"\n--- Scores ---")

    # Show damaged phases if any
    damaged_phases = scores.get('damaged_phases', [])
    healthy_phases = scores.get('healthy_phases', [])
    if damaged_phases:
        lines.append(f"DAMAGED PHASES: {', '.join(damaged_phases)}")
        lines.append(f"Healthy phases: {', '.join(healthy_phases)}")
        lines.append(f"(Scores calculated from healthy phases only)")

    lines.append(f"Matching score: {scores.get('matching_score', 0):.0f}/100")
    lines.append(f"Segmentation score: {scores.get('segmentation_score', 0):.0f}/100")
    lines.append(f"Overall score: {scores.get('overall_score', 0):.0f}/100")

    # Flags
    flags = analysis.get('flags', {})
    active_flags = [k for k, v in flags.items() if v]
    if active_flags:
        lines.append(f"\n--- Issues ---")
        for flag in active_flags:
            lines.append(f"  - {flag.replace('_', ' ').title()}")

    return '\n'.join(lines)
