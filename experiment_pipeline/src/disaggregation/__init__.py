"""
Disaggregation module — signal-level processing.

Detects power events, matches ON/OFF pairs, and extracts device power
profiles from the aggregate signal. Pure signal processing with zero
device knowledge.

Sub-packages (currently re-exported from legacy locations):
    detection/    — Sharp + gradual event detection
    matching/     — Stage 1-3 ON/OFF pair matching
    segmentation/ — Power extraction and remaining signal computation

This is Phase 1 of the architecture split. Files still live at their
original locations; this package provides the new import paths.
"""

# Re-export from existing locations
from detection import (
    detect_sharp_events,
    detect_gradual_events,
    detect_near_threshold_events,
    extend_off_event_tails,
    merge_overlapping_events,
    merge_consecutive_on_events,
    merge_consecutive_off_events,
    expand_event,
)

from matching.stage1 import find_match as find_clean_match
from matching.stage2 import find_noisy_match
from matching.stage3 import find_partial_match
from matching.validator import is_valid_event_removal, build_match_tag

from segmentation.processor import process_phase_segmentation
from segmentation.summarizer import summarize_segmentation
from segmentation.evaluation import calculate_phase_metrics
from segmentation.restore import restore_skipped_to_unmatched

# Pipeline orchestration steps
from pipeline.detection import process_detection
from pipeline.matching import process_matching
from pipeline.segmentation import process_segmentation
from pipeline.evaluation import process_evaluation
from pipeline.visualization import process_visualization
from pipeline.evaluation_summary import generate_dynamic_evaluation_summary

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
    # Pipeline steps
    'process_detection', 'process_matching', 'process_segmentation',
    'process_evaluation', 'process_visualization',
    'generate_dynamic_evaluation_summary',
]
