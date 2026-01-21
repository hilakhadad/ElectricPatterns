"""
Segmentation module.

Separates power consumption into device-specific time series.
"""
from .processor import process_phase_segmentation
from .summarizer import summarize_segmentation
from .errors import log_negative_values

__all__ = [
    'process_phase_segmentation',
    'summarize_segmentation',
    'log_negative_values',
]
