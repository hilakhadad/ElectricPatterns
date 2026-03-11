"""
Configuration constants for device identification.

All thresholds and tolerances centralised in one place for easy tuning.
Adapted from device_classifier.py and patterns.py.
"""
from dataclasses import dataclass, field
from typing import List

# ============================================================================
# Transient event filtering (pre-identification)
# ============================================================================
MIN_EVENT_DURATION_MINUTES = 2      # events shorter than this are transient noise
                                     # (filters only 1-min spikes; keeps 2-min events)

# ============================================================================
# Session grouping
# ============================================================================
DEFAULT_SESSION_GAP_MINUTES = 30    # user requested ~30 (was 60 in legacy code)

# ============================================================================
# Boiler classification
# ============================================================================
BOILER_MIN_DURATION = 15            # minutes
BOILER_MIN_MAGNITUDE = 1500         # watts
BOILER_ISOLATION_WINDOW = 30        # minutes — no medium events nearby
BOILER_MIN_COUNT = 3                # sessions to confirm pattern (informational)

# Recurrence: require multiple consistent events to confirm boiler pattern.
# Non-recurring candidates return to pool for recurring pattern discovery.
BOILER_MIN_RECURRENCE = 3           # minimum similar events on same phase to confirm as boiler
BOILER_RECURRENCE_MAX_MAG_CV = 0.25 # max magnitude CV across recurring events (boilers have same heating element)
BOILER_RECURRENCE_MAX_DUR_CV = 0.40 # max duration CV (thermostat-controlled = consistent)

# Surrounding activity check: reject boiler if events are part of a larger appliance cycle
# (e.g. washing machine heating element, dishwasher heating phase).
# Other boiler candidates on the same phase are EXCLUDED from the activity count.
BOILER_CONTEXT_WINDOW_MINUTES = 60       # check ±60 minutes around each event
BOILER_CONTEXT_ACTIVITY_THRESHOLD = 150  # watts above phase baseline = "active"
BOILER_CONTEXT_MIN_ACTIVE_MINUTES = 10   # active minutes to flag one event as "surrounded"
BOILER_CONTEXT_REJECT_RATIO = 0.50       # if ≥50% of group events are surrounded → reject as boiler

# ============================================================================
# Central AC classification
# ============================================================================
CENTRAL_AC_SYNC_TOLERANCE = 10      # minutes — phase sync tolerance

# Central AC candidate thresholds (per-phase, more lenient than regular AC)
CENTRAL_AC_MIN_MAGNITUDE = 800      # watts — same as regular AC
CENTRAL_AC_MIN_CYCLES = 4           # cycles — same as regular AC
CENTRAL_AC_MIN_CYCLE_DURATION = 3   # minutes — same as AC_MIN_CYCLE_DURATION
CENTRAL_AC_MAX_CYCLE_DURATION = 30  # minutes — same as AC_MAX_CYCLE_DURATION
CENTRAL_AC_MAX_MAGNITUDE_CV = 0.30  # 30% (vs regular AC's 20%)
CENTRAL_AC_MAX_DURATION_CV = 0.40   # 40% — cycle durations should be consistent
CENTRAL_AC_MAX_GAP_CV = 0.50        # 50% — gaps between cycles should be regular

# ============================================================================
# Regular AC classification
# ============================================================================
AC_MIN_MAGNITUDE = 800              # watts
AC_MIN_CYCLE_DURATION = 3           # minutes
AC_MAX_CYCLE_DURATION = 30          # minutes
AC_MIN_INITIAL_DURATION = 10        # minutes — first activation in session
AC_MIN_FOLLOWING_CYCLES = 3         # cycles after initial (total ≥ 4)
AC_MAX_MAGNITUDE_CV = 0.20          # 20% coefficient of variation

# ============================================================================
# AC filter for boiler candidates
# ============================================================================
AC_FILTER_WINDOW = 60               # minutes — search window for compressor cycles
AC_FILTER_MIN_CYCLES = 2            # minimum nearby cycles to disqualify boiler
AC_FILTER_MIN_CYCLE_MAG = 800       # watts
AC_FILTER_MAG_RATIO = 0.50          # cycle ≥ 50% of boiler magnitude

# ============================================================================
# Three-phase device detection (charger / industrial equipment)
# ============================================================================
THREE_PHASE_OVERLAP_TOLERANCE = 0.10  # 10% of event duration as overlap margin
THREE_PHASE_MAX_DURATION_RATIO = 2.0  # max ratio between durations (e.g., 20min vs 40min OK, 20min vs 120min NOT)
THREE_PHASE_MIN_PHASES = 1            # events on 1+ other phases → three_phase_device
THREE_PHASE_MIN_OVERLAP_RATIO = 0.50  # at least 50% of shorter event must actually overlap

# ============================================================================
# Recurring pattern discovery (DBSCAN clustering on unknown sessions)
# ============================================================================
RECURRING_PATTERN_EPS = 0.35             # DBSCAN neighbourhood radius (normalised feature space)
RECURRING_PATTERN_MIN_SAMPLES = 3        # minimum sessions to form a core point
RECURRING_PATTERN_MAX_INTERNAL_CV = 0.20 # max magnitude/duration CV within a cluster to accept it
RECURRING_PATTERN_MIN_SESSIONS = 3       # minimum cluster size to keep as a pattern

# ============================================================================
# Multi-phase simultaneity
# ============================================================================
MULTI_PHASE_WINDOW = 5              # minutes

PHASES = ['w1', 'w2', 'w3']


@dataclass
class IdentificationConfig:
    """Runtime configuration for identification module."""
    session_gap_minutes: int = DEFAULT_SESSION_GAP_MINUTES
