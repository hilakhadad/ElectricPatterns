"""
Identification module — session-level device classification.

Takes disaggregation output (matched ON→OFF pairs from all iterations),
groups them into sessions, and classifies sessions as device types
(boiler, central AC, regular AC, unknown).

Key modules:
    config.py              — Constants and configuration
    session_grouper.py     — Load, deduplicate, group into sessions
    session_classifier.py  — Classify sessions
    session_output.py      — Session-level JSON builder
    cleanup.py             — Intermediate file cleanup
"""

from .session_grouper import (
    load_all_matches,
    deduplicate_cross_iteration,
    group_into_sessions,
    Session,
    MultiPhaseSession,
)

from .session_classifier import classify_sessions, ClassifiedSession

from .session_output import build_session_json

from .cleanup import cleanup_intermediate_files

from .config import IdentificationConfig

__all__ = [
    'load_all_matches',
    'deduplicate_cross_iteration',
    'group_into_sessions',
    'classify_sessions',
    'build_session_json',
    'cleanup_intermediate_files',
    'Session',
    'MultiPhaseSession',
    'ClassifiedSession',
    'IdentificationConfig',
]
