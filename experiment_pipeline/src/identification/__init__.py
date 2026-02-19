"""
Identification module — device-level analysis.

Classifies matched activations into device types (boiler, central AC,
regular AC) and builds the unified JSON output. This module reads
disaggregation output files from disk.

Sub-packages (currently re-exported from legacy locations):
    classifier/  — Device type classification rules
    output/      — Unified JSON activation builder

This is Phase 1 of the architecture split. Files still live at their
original locations; this package provides the new import paths.
"""

# Re-export from existing locations
from classification.device_classifier import (
    classify_iteration_matches,
    generate_activation_list,
)

from output.activation_builder import build_device_activations_json

from pipeline.cleanup import cleanup_intermediate_files

__all__ = [
    'classify_iteration_matches',
    'generate_activation_list',
    'build_device_activations_json',
    'cleanup_intermediate_files',
]
