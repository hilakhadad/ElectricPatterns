"""
Identification module — device-level analysis.

Classifies matched activations into device types (boiler, central AC,
regular AC) and builds the unified JSON output. This module reads
disaggregation output files from disk.

Sub-packages:
    classifier/  — Device type classification rules
    output/      — Unified JSON activation builder
    cleanup.py   — Intermediate file cleanup
"""

# Re-export from sub-packages (moved here in Phase 4)
from .classifier.device_classifier import (
    classify_iteration_matches,
    generate_activation_list,
)

from .output.activation_builder import build_device_activations_json

from .cleanup import cleanup_intermediate_files

__all__ = [
    'classify_iteration_matches',
    'generate_activation_list',
    'build_device_activations_json',
    'cleanup_intermediate_files',
]
