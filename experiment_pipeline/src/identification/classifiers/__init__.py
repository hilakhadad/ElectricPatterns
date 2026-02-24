"""
Classifiers sub-package for device identification.

Contains per-device-type classifiers split from session_classifier.py:
  - boiler_classifier        -- boiler + three-phase identification
  - ac_classifier             -- AC cycling pattern detection
  - central_ac_classifier     -- cross-phase AC overlap
  - recurring_pattern_classifier -- per-house DBSCAN pattern discovery
  - unknown_classifier        -- unknown confidence/reason
  - scoring_utils             -- shared scoring functions
"""
