"""
Legacy module - original monolithic code.

This folder contains copies of the original files before refactoring.
The code here is preserved for reference and backward compatibility.

Original files:
- on_off_log.py: ON/OFF event detection (detection + merging + expansion)
- new_matcher.py: Event matching (stage1, stage2, stack-based, validation)
- detection_config.py: Experiment configuration
- data_util.py: Paths and logging utilities

New modular structure (in parent directory):
- core/: Configuration, paths, logging
- detection/: Sharp, gradual, merger, expander
- matching/: Validator, stage1, stage2, stack_matcher, io
"""
