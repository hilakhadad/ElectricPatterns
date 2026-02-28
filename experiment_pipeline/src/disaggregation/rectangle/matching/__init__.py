"""
Event matching module.

Contains algorithms for matching ON events to OFF events.
"""
from .validator import is_valid_event_removal, is_valid_partial_removal
from .stage1 import find_match
from .stage2 import find_noisy_match
from .stage3 import find_partial_match
from .complementary import find_complementary_off_matches
from .stack_matcher import find_matches_stack_based
from .io import save_events, save_remainder_events

__all__ = [
    'is_valid_event_removal',
    'is_valid_partial_removal',
    'find_match',
    'find_noisy_match',
    'find_partial_match',
    'find_complementary_off_matches',
    'find_matches_stack_based',
    'save_events',
    'save_remainder_events',
]
