"""
Pipeline orchestration module.

Contains the main process functions that coordinate the pipeline steps.
"""
from .detection import process_detection
from .matching import process_matching
from .segmentation import process_segmentation
from .evaluation import process_evaluation
from .visualization import process_visualization

__all__ = [
    'process_detection',
    'process_matching',
    'process_segmentation',
    'process_evaluation',
    'process_visualization',
]
