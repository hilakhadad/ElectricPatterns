"""
Detection post-processing steps.

Applies near-threshold detection, tail extension, split-OFF merging,
and settling extension to detected ON/OFF events.
"""
import pandas as pd

from disaggregation.rectangle.detection import merge_overlapping_events
from disaggregation.rectangle.detection.near_threshold import detect_near_threshold_events


def apply_near_threshold(results_on, results_off, data, data_indexed, diff_col,
                         threshold, off_threshold, phase, logger,
                         min_factor=0.85, max_extend_minutes=3):
    """Detect and merge near-threshold events into existing results."""
    near_on, near_off = detect_near_threshold_events(
        data, data_indexed, diff_col, threshold, off_threshold,
        results_on, results_off, phase,
        min_factor=min_factor,
        max_extend_minutes=max_extend_minutes
    )
    if len(near_on) > 0:
        logger.info(f"    Found {len(near_on)} near-threshold ON events for {phase}")
        results_on = pd.concat([results_on, near_on], ignore_index=True)
    if len(near_off) > 0:
        logger.info(f"    Found {len(near_off)} near-threshold OFF events for {phase}")
        results_off = pd.concat([results_off, near_off], ignore_index=True)
    # Merge overlapping in case near-threshold events overlap with existing
    results_on = merge_overlapping_events(results_on, max_gap_minutes=0, data=data_indexed, phase=phase)
    results_off = merge_overlapping_events(results_off, max_gap_minutes=0, data=data_indexed, phase=phase)
    return results_on, results_off


def apply_tail_extension(results_off, data_indexed, phase, logger,
                         max_minutes=10, min_residual=100,
                         noise_tolerance=30, min_gain=100,
                         min_residual_fraction=0.05):
    """Extend OFF event tails for devices with soft landing."""
    from disaggregation.rectangle.detection.tail_extension import extend_off_event_tails
    results_off = extend_off_event_tails(
        results_off, data_indexed, phase,
        max_minutes=max_minutes, min_residual=min_residual,
        noise_tolerance=noise_tolerance, min_gain=min_gain,
        min_residual_fraction=min_residual_fraction
    )
    extended = results_off['tail_extended'].sum() if 'tail_extended' in results_off.columns else 0
    if extended > 0:
        logger.info(f"    Tail extended: {extended} OFF events for {phase}")
    return results_off


def apply_split_off_merger(results_off, results_on, data_indexed, phase, logger,
                           max_gap_minutes=2):
    """Merge split device shutdowns before settling extension."""
    from disaggregation.rectangle.detection.merger import merge_split_off_events
    before_merge = len(results_off)
    results_off = merge_split_off_events(
        results_off, results_on,
        max_gap_minutes=max_gap_minutes,
        data=data_indexed, phase=phase
    )
    if len(results_off) < before_merge:
        logger.info(f"    Split-OFF merged: {before_merge - len(results_off)} events for {phase}")
    return results_off


def apply_settling_extension(results_on, results_off, data_indexed, phase, logger,
                             settling_factor=0.7, max_settling_minutes=5):
    """Extend ON/OFF boundaries through transient settling periods."""
    from disaggregation.rectangle.detection.settling import (
        extend_settling_on_events, extend_settling_off_events
    )
    if len(results_on) > 0:
        before_mags = results_on['magnitude'].copy()
        results_on = extend_settling_on_events(
            results_on, data_indexed, phase,
            off_events=results_off,
            settling_factor=settling_factor,
            max_settling_minutes=max_settling_minutes,
        )
        extended_count = (results_on['magnitude'] != before_mags.values).sum() if len(results_on) == len(before_mags) else 0
        if extended_count > 0:
            logger.info(f"    Settling extended: {extended_count} ON events for {phase}")

    if len(results_off) > 0:
        before_mags = results_off['magnitude'].copy()
        results_off = extend_settling_off_events(
            results_off, data_indexed, phase,
            on_events=results_on,
            settling_factor=settling_factor,
            max_settling_minutes=max_settling_minutes,
        )
        extended_count = (results_off['magnitude'] != before_mags.values).sum() if len(results_off) == len(before_mags) else 0
        if extended_count > 0:
            logger.info(f"    Settling extended: {extended_count} OFF events for {phase}")

    return results_on, results_off
