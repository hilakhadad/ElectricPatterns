"""Test configuration for experiment_pipeline regression tests."""
import sys
from pathlib import Path

# Add src/ to path so we can import pipeline modules
src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
