"""
Disaggregation module — signal-level processing.

Detects power events, matches ON/OFF pairs, and extracts device power
profiles from the aggregate signal. Pure signal processing with zero
device knowledge.

Sub-packages:
    rectangle/      — Flat (rectangular) power extraction (original M1 pipeline)
    wave_recovery/  — Wave-shaped power extraction (Post-M1 recovery step)

Legacy shim packages (detection/, matching/, segmentation/, pipeline/) re-export
from rectangle/ for backward compatibility.
"""

# Re-export from rectangle sub-package for backward compatibility
from .rectangle.detection import (
    detect_sharp_events,
    detect_gradual_events,
    detect_near_threshold_events,
    extend_off_event_tails,
    merge_overlapping_events,
    merge_consecutive_on_events,
    merge_consecutive_off_events,
    expand_event,
)

from .rectangle.matching.stage1 import find_match as find_clean_match
from .rectangle.matching.stage2 import find_noisy_match
from .rectangle.matching.stage3 import find_partial_match
from .rectangle.matching.validator import is_valid_event_removal, build_match_tag

from .rectangle.segmentation.processor import process_phase_segmentation
from .rectangle.segmentation.summarizer import summarize_segmentation
from .rectangle.segmentation.evaluation import calculate_phase_metrics
from .rectangle.segmentation.restore import restore_skipped_to_unmatched

# Note: Pipeline orchestration steps (process_detection, process_matching, etc.)
# live at disaggregation/pipeline/*_step.py (shims) → disaggregation/rectangle/pipeline/*_step.py

__all__ = [
    # Detection
    'detect_sharp_events', 'detect_gradual_events',
    'detect_near_threshold_events', 'extend_off_event_tails',
    'merge_overlapping_events', 'merge_consecutive_on_events',
    'merge_consecutive_off_events', 'expand_event',
    # Matching
    'find_clean_match', 'find_noisy_match', 'find_partial_match',
    'is_valid_event_removal', 'build_match_tag',
    # Segmentation
    'process_phase_segmentation', 'summarize_segmentation',
    'calculate_phase_metrics', 'restore_skipped_to_unmatched',
]
