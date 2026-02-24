"""
Spike statistics computation for transient event filtering.

Computes detailed statistics about filtered (spike) vs kept events,
broken down by iteration and phase.
"""
import pandas as pd

from .config import MIN_EVENT_DURATION_MINUTES, PHASES


def _empty_spike_stats() -> dict:
    """Return empty spike stats structure."""
    return {
        'min_duration_threshold': MIN_EVENT_DURATION_MINUTES,
        'spike_count': 0,
        'spike_total_minutes': 0.0,
        'kept_count': 0,
        'kept_total_minutes': 0.0,
        'short_count': 0,
        'short_minutes': 0.0,
        'long_count': 0,
        'long_minutes': 0.0,
        'long_duration_threshold': 25,
        'by_iteration': {},
        'by_phase': {},
    }


def _compute_spike_stats(
    spikes: pd.DataFrame,
    kept: pd.DataFrame,
    min_duration: float,
) -> dict:
    """Compute statistics about filtered transient events."""
    spike_minutes = float(spikes['duration'].sum()) if len(spikes) > 0 else 0.0
    kept_minutes = float(kept['duration'].sum()) if len(kept) > 0 else 0.0

    # Duration breakdown of kept events: short (2-15 min) vs long (>=15 min)
    LONG_DURATION_THRESHOLD = 15  # minutes
    if len(kept) > 0:
        short_mask = kept['duration'] < LONG_DURATION_THRESHOLD
        short_count = int(short_mask.sum())
        long_count = int((~short_mask).sum())
        short_minutes = round(float(kept.loc[short_mask, 'duration'].sum()), 1)
        long_minutes = round(float(kept.loc[~short_mask, 'duration'].sum()), 1)
    else:
        short_count = long_count = 0
        short_minutes = long_minutes = 0.0

    # Breakdown by iteration
    by_iteration = {}
    for iter_num in sorted(set(
        list(spikes['iteration'].unique()) + list(kept['iteration'].unique())
    )):
        iter_spikes = spikes[spikes['iteration'] == iter_num]
        iter_kept = kept[kept['iteration'] == iter_num]
        by_iteration[int(iter_num)] = {
            'spike_count': len(iter_spikes),
            'spike_minutes': round(float(iter_spikes['duration'].sum()), 1) if len(iter_spikes) > 0 else 0.0,
            'kept_count': len(iter_kept),
            'kept_minutes': round(float(iter_kept['duration'].sum()), 1) if len(iter_kept) > 0 else 0.0,
        }

    # Breakdown by phase
    by_phase = {}
    for phase in PHASES:
        phase_spikes = spikes[spikes['phase'] == phase] if 'phase' in spikes.columns else pd.DataFrame()
        phase_kept = kept[kept['phase'] == phase] if 'phase' in kept.columns else pd.DataFrame()
        if len(phase_spikes) > 0 or len(phase_kept) > 0:
            by_phase[phase] = {
                'spike_count': len(phase_spikes),
                'spike_minutes': round(float(phase_spikes['duration'].sum()), 1) if len(phase_spikes) > 0 else 0.0,
                'kept_count': len(phase_kept),
                'kept_minutes': round(float(phase_kept['duration'].sum()), 1) if len(phase_kept) > 0 else 0.0,
            }

    return {
        'min_duration_threshold': min_duration,
        'spike_count': len(spikes),
        'spike_total_minutes': round(spike_minutes, 1),
        'kept_count': len(kept),
        'kept_total_minutes': round(kept_minutes, 1),
        'short_count': short_count,
        'short_minutes': short_minutes,
        'long_count': long_count,
        'long_minutes': long_minutes,
        'long_duration_threshold': LONG_DURATION_THRESHOLD,
        'by_iteration': by_iteration,
        'by_phase': by_phase,
    }
