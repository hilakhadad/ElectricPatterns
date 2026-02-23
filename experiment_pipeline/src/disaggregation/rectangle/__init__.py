"""
Rectangle disaggregation module — signal-level processing with flat (rectangular) power extraction.

Detects power events, matches ON/OFF pairs, and extracts device power
profiles from the aggregate signal as flat/rectangular shapes.

Sub-packages:
    detection/    — Sharp + gradual event detection
    matching/     — Stage 1-3 ON/OFF pair matching
    segmentation/ — Power extraction and remaining signal computation
    pipeline/     — Orchestration steps
"""

from .detection import (
    detect_sharp_events,
    detect_gradual_events,
    detect_near_threshold_events,
    extend_off_event_tails,
    merge_overlapping_events,
    merge_consecutive_on_events,
    merge_consecutive_off_events,
    expand_event,
)

from .matching.stage1 import find_match as find_clean_match
from .matching.stage2 import find_noisy_match
from .matching.stage3 import find_partial_match
from .matching.validator import is_valid_event_removal, build_match_tag

from .segmentation.processor import process_phase_segmentation
from .segmentation.summarizer import summarize_segmentation
from .segmentation.evaluation import calculate_phase_metrics
from .segmentation.restore import restore_skipped_to_unmatched
