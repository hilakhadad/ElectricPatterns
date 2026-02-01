"""
Event detection module.

Detects ON/OFF power events from household electricity data.
"""
from .sharp import detect_sharp_events
from .gradual import detect_gradual_events
from .merger import merge_overlapping_events, merge_consecutive_on_events, merge_consecutive_off_events
from .expander import expand_event

__all__ = [
    'detect_sharp_events',
    'detect_gradual_events',
    'merge_overlapping_events',
    'merge_consecutive_on_events',
    'merge_consecutive_off_events',
    'expand_event',
]
