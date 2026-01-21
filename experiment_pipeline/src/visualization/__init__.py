"""
Visualization module.

Creates interactive and static plots of segmentation results.
"""
from .interactive import plot_interactive
from .static import plot_static
from .utils import simplify_event_id, create_title, split_by_day_night

__all__ = [
    'plot_interactive',
    'plot_static',
    'simplify_event_id',
    'create_title',
    'split_by_day_night',
]
