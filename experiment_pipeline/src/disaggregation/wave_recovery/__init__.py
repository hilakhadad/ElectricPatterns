"""
Wave recovery module — extracts wave-shaped power patterns from remaining signal.

Post-M1 step that finds and extracts gradual-decay power patterns (e.g., AC compressor
cycles where power starts high and decreases monotonically). Also searches for cross-phase
patterns when a wave is detected on one phase.

Sub-packages (mirror rectangle/ structure):
    detection/    — Wave pattern detection in remaining power
    matching/     — Cross-phase pattern matching
    segmentation/ — Wave-shaped power extraction + validation
    pipeline/     — Orchestration step for runner.py
"""
