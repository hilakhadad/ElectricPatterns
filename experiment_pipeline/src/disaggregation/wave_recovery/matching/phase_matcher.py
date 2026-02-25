"""
Cross-phase wave pattern matching.

When a wave is detected on one phase, search the other phases in the same
time window for similar wave patterns — possibly at lower magnitudes.
This is critical for detecting 3-phase central AC where W1 might show a
strong wave but W2/W3 have weaker versions that M1 completely missed.

Algorithm:
    1. For each detected wave on phase X, define a search window on phases Y, Z.
    2. In that window, scan remaining_Y and remaining_Z for wave patterns using
       relaxed thresholds (wave_min_rise_watts * 0.5).
    3. Mark cross-phase matches with their source phase.
    4. Return additional waves found on other phases.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..detection.wave_detector import WavePattern, detect_wave_patterns


# Window padding (minutes) around a detected wave when searching other phases
CROSS_PHASE_WINDOW_PAD = 5

# Relaxation factor for cross-phase detection (lower magnitude OK if time-aligned)
CROSS_PHASE_RISE_FACTOR = 0.5


def find_cross_phase_waves(
    detected_waves: Dict[str, List[WavePattern]],
    remaining_by_phase: Dict[str, pd.Series],
    config,
    logger=None,
) -> Dict[str, List[WavePattern]]:
    """
    Search other phases for wave patterns aligned with already-detected waves.

    Parameters
    ----------
    detected_waves : dict
        {phase: [WavePattern, ...]} — waves already found per phase.
    remaining_by_phase : dict
        {phase: pd.Series} — remaining power indexed by timestamp for each phase.
    config : ExperimentConfig
        Wave recovery parameters; wave_min_rise_watts used with relaxation.
    logger : optional
        Logger for debug output.

    Returns
    -------
    dict
        {phase: [WavePattern, ...]} — *additional* waves found via cross-phase
        search (does NOT include the original detected_waves).
    """
    all_phases = sorted(remaining_by_phase.keys())
    extra_waves: Dict[str, List[WavePattern]] = {p: [] for p in all_phases}

    # Build a set of (phase, start) tuples already known — avoid duplicates
    known = set()
    for phase, waves in detected_waves.items():
        for w in waves:
            known.add((phase, w.start))

    # Build a relaxed config copy for cross-phase search
    relaxed_config = _relaxed_config(config)

    for source_phase, waves in detected_waves.items():
        other_phases = [p for p in all_phases if p != source_phase]
        for wave in waves:
            for target_phase in other_phases:
                if target_phase not in remaining_by_phase:
                    continue

                target_remaining = remaining_by_phase[target_phase]
                if target_remaining.empty:
                    continue

                # Define search window
                window_start = wave.start - pd.Timedelta(minutes=CROSS_PHASE_WINDOW_PAD)
                window_end = wave.end + pd.Timedelta(minutes=CROSS_PHASE_WINDOW_PAD)

                # Slice the target remaining to this window
                mask = (target_remaining.index >= window_start) & (target_remaining.index <= window_end)
                window_series = target_remaining.loc[mask]
                if len(window_series) < config.wave_min_duration_minutes:
                    continue

                # Detect with relaxed thresholds
                cross_waves = detect_wave_patterns(
                    window_series, target_phase, relaxed_config, logger=None
                )

                for cw in cross_waves:
                    if (cw.phase, cw.start) not in known:
                        known.add((cw.phase, cw.start))
                        extra_waves[target_phase].append(cw)
                        if logger:
                            logger.info(
                                f"Cross-phase wave on {target_phase} "
                                f"({cw.start} -> {cw.end}, {cw.duration_minutes} min, "
                                f"peak={cw.peak_power:.0f}W) "
                                f"found via {source_phase} template"
                            )

    if logger:
        total_extra = sum(len(v) for v in extra_waves.values())
        if total_extra:
            logger.debug(f"Cross-phase: {total_extra} additional waves found")

    return extra_waves


class _relaxed_config:
    """Thin wrapper that halves wave_min_rise_watts for cross-phase search."""

    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        if name == 'wave_min_rise_watts':
            return int(getattr(self._original, 'wave_min_rise_watts') * CROSS_PHASE_RISE_FACTOR)
        if name == '_original':
            raise AttributeError
        return getattr(self._original, name)
