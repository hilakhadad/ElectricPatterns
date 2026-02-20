"""
experiment_pipeline/src/ — Root package.

Sub-packages:
    core/           — Configuration, paths, data loading
    disaggregation/ — Signal-level processing (detection, matching, segmentation)
    identification/ — Session-level device classification
    pipeline/       — Unified runner orchestrating disaggregation + identification
    visualization/  — Interactive and static plots
"""

from . import core
from . import disaggregation
from . import identification
from . import pipeline
from . import visualization

__all__ = [
    'core',
    'disaggregation',
    'identification',
    'pipeline',
    'visualization',
]
