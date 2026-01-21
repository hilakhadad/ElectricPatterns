# Source modules for experiment pipeline
"""
Modular structure:
- core/: Configuration, paths, logging
- detection/: Event detection (sharp, gradual)
- matching/: Event matching (stage1, stage2, noisy)
- segmentation/: Power segmentation and evaluation
- visualization/: Interactive and static plots
- pipeline/: Orchestration functions (process_*)
- legacy/: Original code backup
"""

from . import core
from . import detection
from . import matching
from . import segmentation
from . import visualization
from . import pipeline

__all__ = [
    'core',
    'detection',
    'matching',
    'segmentation',
    'visualization',
    'pipeline',
]
