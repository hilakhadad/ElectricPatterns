"""
Chart generation for device identification (Module 2) reports.

This module is a facade -- actual implementations are split across:
  - charts_session.py     -- session overview, boiler/AC/temporal/unclassified charts
  - charts_device.py      -- device activations detail, summary table, chart rendering
  - charts_confidence.py  -- confidence distribution overview
  - charts_spike.py       -- spike analysis charts
"""
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Color constants (consistent with classification_charts.py)
GREEN = '#28a745'
GRAY = '#6c757d'
ORANGE = '#e67e22'
YELLOW = '#eab308'
PURPLE = '#6f42c1'
BLUE = '#007bff'
RED = '#dc3545'
LIGHT_GREEN = '#d4edda'

DEVICE_COLORS = {
    'boiler': '#007bff',
    'three_phase_device': '#6f42c1',
    'central_ac': '#dc3545',
    'regular_ac': '#e67e22',
    'recurring_pattern': '#17a2b8',
    'unknown': '#6c757d',
    'unclassified': '#6c757d',
}

DEVICE_DISPLAY_NAMES = {
    'boiler': 'Water Heater (Boiler)',
    'three_phase_device': '3-Phase Device (Charger?)',
    'central_ac': 'Central AC (Multi-phase)',
    'regular_ac': 'Regular AC (Single-phase)',
    'recurring_pattern': 'Recurring Pattern (Discovered)',
    'unknown': 'Unclassified',
    'unclassified': 'Unclassified',
}

# Re-export from sub-modules for backward compatibility
from visualization.charts_session import (
    create_session_overview,
    create_boiler_analysis,
    create_ac_analysis,
    _count_phase_combos,
    create_temporal_heatmap,
    create_unclassified_analysis,
)

from visualization.charts_device import (
    _parse_iso,
    _dur_str,
    create_device_summary_table,
    _parse_session_row,
    _group_central_ac_for_display,
    create_device_activations_detail,
    _hex_to_rgba,
    _extract_chart_window,
    _build_chart_row_html,
    _build_activation_charts_script,
)

from visualization.charts_confidence import (
    create_confidence_overview,
)

from visualization.charts_spike import (
    create_spike_analysis,
)

__all__ = [
    # Session charts
    'create_session_overview',
    'create_boiler_analysis',
    'create_ac_analysis',
    'create_temporal_heatmap',
    'create_unclassified_analysis',
    # Device charts
    'create_device_summary_table',
    'create_device_activations_detail',
    # Confidence charts
    'create_confidence_overview',
    # Spike charts
    'create_spike_analysis',
    # Constants
    'DEVICE_COLORS',
    'DEVICE_DISPLAY_NAMES',
]
