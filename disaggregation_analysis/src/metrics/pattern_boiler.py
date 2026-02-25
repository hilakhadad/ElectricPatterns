"""
Boiler pattern detection and validation.

Extracted from patterns.py -- contains boiler-specific
detection, filtering, and pattern validation.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


def _filter_ac_from_boiler_candidates(
    boiler_activations: List[Dict],
    matches_df: pd.DataFrame,
    compressor_window_minutes: int = 60,
    min_nearby_cycles: int = 2,
    min_cycle_duration: int = 3,
    max_cycle_duration: int = 24,
    min_cycle_magnitude: int = 800,
) -> tuple:
    """
    Filter out boiler candidates that have compressor cycling patterns nearby.

    AC compressors create patterns that look like boilers (long, high power) but
    are followed or preceded by short ON/OFF cycles (3-24 min). If such cycles
    exist near a boiler candidate on the same phase, it's likely AC, not boiler.

    Args:
        boiler_activations: List of boiler candidate dicts
        matches_df: All matches DataFrame with datetime columns
        compressor_window_minutes: Window after off_end to search for cycling (default 60 min)
        min_nearby_cycles: Minimum compressor cycles to reclassify (default 2)
        min_cycle_duration: Min cycle duration in minutes (default 3)
        max_cycle_duration: Max cycle duration in minutes (default 24)
        min_cycle_magnitude: Min cycle magnitude in watts (default 800)

    Returns:
        Tuple of (remaining_boiler_activations, reclassified_as_ac)
    """
    if not boiler_activations or matches_df is None or matches_df.empty:
        return boiler_activations, []

    compressor_window = pd.Timedelta(minutes=compressor_window_minutes)

    remaining_boiler = []
    reclassified_ac = []

    for activation in boiler_activations:
        phase = activation['phase']
        boiler_magnitude = activation['magnitude']

        # Reconstruct timestamps
        on_start = pd.to_datetime(f"{activation['date']} {activation['on_time']}")
        if activation.get('off_time'):
            off_end = pd.to_datetime(f"{activation['date']} {activation['off_time']}")
            if off_end < on_start:
                off_end += pd.Timedelta(days=1)
        else:
            off_end = on_start + pd.Timedelta(minutes=activation['duration_minutes'])

        # Search window: from before on_start to after off_end
        # The key fix: extend beyond off_end, not just from on_start
        search_start = on_start - compressor_window
        search_end = off_end + compressor_window

        # Find compressor-like cycles on same phase in window
        nearby = matches_df[
            (matches_df['phase'] == phase) &
            (matches_df['duration'] >= min_cycle_duration) &
            (matches_df['duration'] <= max_cycle_duration) &
            (matches_df['on_magnitude'].abs() >= min_cycle_magnitude) &
            (matches_df['on_start'] >= search_start) &
            (matches_df['on_start'] <= search_end)
        ]

        if len(nearby) >= min_nearby_cycles:
            # Check magnitude similarity: cycles should be at least 50% of boiler magnitude
            similar_cycles = nearby[
                nearby['on_magnitude'].abs() >= boiler_magnitude * 0.5
            ]

            if len(similar_cycles) >= min_nearby_cycles:
                activation['reclassified_reason'] = (
                    f"Found {len(similar_cycles)} compressor cycles "
                    f"({min_cycle_duration}-{max_cycle_duration} min) nearby on {phase}"
                )
                reclassified_ac.append(activation)
                continue

        remaining_boiler.append(activation)

    return remaining_boiler, reclassified_ac


def detect_boiler_patterns(experiment_dir: Path, house_id: str,
                           run_number: int = 0,
                           min_duration_minutes: int = 25,
                           min_magnitude: int = 1500,
                           isolation_window_minutes: int = 30,
                           preloaded: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Detect water heater (boiler) patterns.

    Boiler characteristics:
    - Long duration (typically 25-60+ minutes)
    - High power (typically 1500-3000W)
    - Isolated: no medium-duration events before or after within a time window

    Args:
        experiment_dir: Path to experiment output directory
        house_id: House identifier
        run_number: Run number to analyze
        min_duration_minutes: Minimum match duration to consider (default 25 min)
        min_magnitude: Minimum magnitude in watts (default 1500W)
        isolation_window_minutes: Window to check for nearby events (default 30 min)
        preloaded: Optional dict with pre-loaded DataFrames ('matches')

    Returns:
        Dictionary with boiler detection results
    """
    logger.debug("detect_boiler_patterns: house_id=%s, run_number=%d, preloaded=%s",
                 house_id, run_number, bool(preloaded))
    result = {
        'boiler': {
            'activations': [],
            'total_count': 0,
            'avg_duration': 0,
            'avg_magnitude': 0,
        },
        'has_boiler': False,
    }

    if preloaded:
        matches_df = preloaded.get('matches')
        if matches_df is not None:
            matches_df = matches_df.copy()
    else:
        from metrics.patterns import _get_house_dir, _load_monthly_files
        house_dir = _get_house_dir(experiment_dir, house_id, run_number)
        if not house_dir.exists():
            return result
        matches_df = _load_monthly_files(house_dir, "matches", f"matches_{house_id}_*.pkl")

    if matches_df is None or len(matches_df) == 0:
        logger.warning("detect_boiler_patterns: No matches data found for house %s run %d",
                       house_id, run_number)
        return result

    # Parse timestamps (skip if already datetime)
    for col in ['on_start', 'on_end', 'off_start', 'off_end']:
        if col in matches_df.columns and not pd.api.types.is_datetime64_any_dtype(matches_df[col]):
            matches_df[col] = pd.to_datetime(matches_df[col], format='mixed', dayfirst=True, errors='coerce')

    # Step 1: Find all long-duration, high-power matches (potential boilers)
    if 'duration' not in matches_df.columns or 'on_magnitude' not in matches_df.columns:
        return result

    long_matches = matches_df[
        (matches_df['duration'] >= min_duration_minutes) &
        (matches_df['on_magnitude'].abs() >= min_magnitude)
    ].copy()

    if long_matches.empty:
        return result

    # Step 2: Find medium-duration matches (these should NOT be near boiler events)
    # Medium = 3-24 minutes (as defined in segmentation)
    medium_matches = matches_df[
        (matches_df['duration'] > 2) &
        (matches_df['duration'] <= 24)
    ].copy()

    isolation_window = pd.Timedelta(minutes=isolation_window_minutes)

    # Optimized: use merge_asof to find nearby medium events
    # Process each phase separately
    boiler_activations = []

    for phase in long_matches['phase'].unique():
        phase_long = long_matches[long_matches['phase'] == phase].copy()
        phase_long = phase_long.sort_values('on_start').reset_index(drop=True)

        if medium_matches.empty:
            # No medium events - all long events are isolated
            isolated = phase_long
        else:
            phase_medium = medium_matches[medium_matches['phase'] == phase].copy()

            if phase_medium.empty:
                # No medium events on this phase - all isolated
                isolated = phase_long
            else:
                phase_medium = phase_medium.sort_values('on_start').reset_index(drop=True)

                # Use merge_asof to find nearest medium event
                # Ensure both DataFrames are sorted and have reset index (merge_asof requires this)
                left_df = phase_long[['on_start', 'off_end', 'off_start', 'on_magnitude', 'duration', 'phase']].copy()
                left_df = left_df.sort_values('on_start').reset_index(drop=True)
                right_df = phase_medium[['on_start']].rename(columns={'on_start': 'nearby_medium_start'}).copy()
                right_df = right_df.sort_values('nearby_medium_start').reset_index(drop=True)

                merged = pd.merge_asof(
                    left_df,
                    right_df,
                    left_on='on_start',
                    right_on='nearby_medium_start',
                    tolerance=isolation_window,
                    direction='nearest'
                )

                # Isolated = no nearby medium event found
                isolated = merged[merged['nearby_medium_start'].isna()].copy()

        # Build activation dicts
        for _, row in isolated.iterrows():
            on_start = row['on_start']
            if pd.isna(on_start):
                continue
            off_end = row.get('off_end') if pd.notna(row.get('off_end')) else row.get('off_start')

            activation = {
                'date': on_start.strftime('%Y-%m-%d'),
                'on_time': on_start.strftime('%H:%M'),
                'off_time': off_end.strftime('%H:%M') if pd.notna(off_end) else '',
                'duration_minutes': int(round(row['duration'])),
                'magnitude': int(abs(row['on_magnitude'])),
                'phase': phase,
            }
            boiler_activations.append(activation)

    # Sort by date and time
    boiler_activations.sort(key=lambda x: (x['date'], x['on_time']))

    # ===== Filter out boiler candidates with compressor cycling nearby =====
    # AC compressors create long activations followed by short ON/OFF cycles.
    # The original isolation check misses these because it only checks near on_start,
    # but compressor cycles after off_end can be >30 min from on_start for long events.
    boiler_activations, reclassified_ac = _filter_ac_from_boiler_candidates(
        boiler_activations, matches_df
    )

    # ===== NEW: Detect multi-phase simultaneous events (might be EV charging, not boiler) =====
    # Boiler should only use ONE phase. If multiple phases are active at the same time,
    # it's likely something else (EV charging, heavy industrial equipment, etc.)

    sync_tolerance = pd.Timedelta(minutes=5)  # Events within 5 min are "simultaneous"

    # Convert to DataFrame for easier processing
    if boiler_activations:
        boiler_df = pd.DataFrame(boiler_activations)
        boiler_df['on_datetime'] = pd.to_datetime(boiler_df['date'] + ' ' + boiler_df['on_time'])

        # Find events that have other-phase events at the same time
        multi_phase_events = []
        single_phase_events = []

        for i, row in boiler_df.iterrows():
            on_time = row['on_datetime']
            phase = row['phase']

            # Find other-phase events at similar time
            other_phase_events = boiler_df[
                (boiler_df['phase'] != phase) &
                (abs(boiler_df['on_datetime'] - on_time) <= sync_tolerance)
            ]

            if len(other_phase_events) > 0:
                # This event has simultaneous activity on other phases
                event = row.to_dict()
                event['other_phases_active'] = other_phase_events['phase'].unique().tolist()
                event['num_phases_active'] = len(other_phase_events['phase'].unique()) + 1
                event.pop('on_datetime', None)  # Remove helper column
                multi_phase_events.append(event)
            else:
                # Single-phase event - likely real boiler
                event = row.to_dict()
                event['other_phases_active'] = []
                event['num_phases_active'] = 1
                event.pop('on_datetime', None)
                single_phase_events.append(event)

        # Count events per phase for single-phase events
        phase_counts = {}
        for evt in single_phase_events:
            p = evt['phase']
            phase_counts[p] = phase_counts.get(p, 0) + 1

        # Find dominant phase (most boiler events)
        dominant_phase = max(phase_counts, key=phase_counts.get) if phase_counts else None
        result['boiler']['dominant_phase'] = dominant_phase
        result['boiler']['phase_distribution'] = phase_counts

        # Final boiler activations = single-phase events only
        result['boiler']['activations'] = single_phase_events
        result['boiler']['total_count'] = len(single_phase_events)

        if single_phase_events:
            result['boiler']['avg_duration'] = sum(a['duration_minutes'] for a in single_phase_events) / len(single_phase_events)
            result['boiler']['avg_magnitude'] = sum(a['magnitude'] for a in single_phase_events) / len(single_phase_events)

        # Multi-phase events table (might be EV charging or other heavy load)
        result['suspicious_multi_phase'] = {
            'activations': multi_phase_events,
            'total_count': len(multi_phase_events),
            'description': 'Events with boiler-like characteristics but multiple phases active simultaneously - may be EV charging or other high-power device',
        }

        if multi_phase_events:
            # Check if these are truly synchronized (3 phases = likely EV or central device)
            three_phase_count = sum(1 for e in multi_phase_events if e['num_phases_active'] == 3)
            two_phase_count = sum(1 for e in multi_phase_events if e['num_phases_active'] == 2)

            result['suspicious_multi_phase']['three_phase_count'] = three_phase_count
            result['suspicious_multi_phase']['two_phase_count'] = two_phase_count

            if three_phase_count > len(multi_phase_events) * 0.5:
                result['suspicious_multi_phase']['likely_device'] = 'EV_charging_or_central_device'
            else:
                result['suspicious_multi_phase']['likely_device'] = 'unknown'

    else:
        result['boiler']['activations'] = []
        result['boiler']['total_count'] = 0
        result['boiler']['dominant_phase'] = None
        result['boiler']['phase_distribution'] = {}
        result['suspicious_multi_phase'] = {'activations': [], 'total_count': 0}
        reclassified_ac = []

    result['has_boiler'] = result['boiler']['total_count'] >= 3  # Need at least 3 single-phase activations

    # Include reclassified events (boiler candidates that are actually AC)
    result['reclassified_as_ac'] = {
        'activations': reclassified_ac,
        'total_count': len(reclassified_ac),
    }

    return result
