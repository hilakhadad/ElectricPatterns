"""
Pipeline orchestration module.

Contains the main process functions that coordinate the pipeline steps,
and the unified pipeline runner for both static and dynamic experiments.
"""
from .detection import process_detection
from .matching import process_matching
from .segmentation import process_segmentation
from .evaluation import process_evaluation
from .visualization import process_visualization
from .runner import run_pipeline
from .evaluation_summary import generate_dynamic_evaluation_summary
from .cleanup import cleanup_intermediate_files

__all__ = [
    'process_detection',
    'process_matching',
    'process_segmentation',
    'process_evaluation',
    'process_visualization',
    'run_pipeline',
    'generate_dynamic_evaluation_summary',
    'cleanup_intermediate_files',
]
