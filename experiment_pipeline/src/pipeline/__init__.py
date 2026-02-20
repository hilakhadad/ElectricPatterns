"""
Pipeline orchestration module.

Contains the unified pipeline runner. Process steps now live under
disaggregation.pipeline and identification.
"""
from .runner import run_pipeline

from disaggregation.pipeline.detection_step import process_detection
from disaggregation.pipeline.matching_step import process_matching
from disaggregation.pipeline.segmentation_step import process_segmentation
from disaggregation.pipeline.evaluation_step import process_evaluation
from disaggregation.pipeline.visualization_step import process_visualization
from disaggregation.pipeline.evaluation_summary import generate_dynamic_evaluation_summary
from identification.cleanup import cleanup_intermediate_files

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
