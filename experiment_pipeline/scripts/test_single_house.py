"""
Run the pipeline on a single house.

Supports both static and dynamic threshold experiments automatically.
The mode is determined by the experiment config:
- If the experiment has a threshold_schedule → dynamic mode
- Otherwise → static mode (legacy)

Usage:
    # Default (dynamic threshold, exp010):
    python scripts/test_single_house.py --house_id 305

    # Static experiment (legacy):
    python scripts/test_single_house.py --house_id 305 --experiment_name exp007_symmetric_threshold

    # With options:
    python scripts/test_single_house.py --house_id 305 --skip_visualization --minimal_output
"""
import sys
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows console (safer approach that won't close stdout)
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
    except Exception:
        pass

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

_SCRIPT_DIR = Path(__file__).parent.parent.absolute()

DEFAULT_HOUSE_ID = "221"


def main():
    """Run pipeline with command-line arguments or defaults."""
    import argparse
    from core.config import get_experiment, DEFAULT_EXPERIMENT

    parser = argparse.ArgumentParser(description="Run pipeline for a single house")
    parser.add_argument("--house_id", type=str, default=DEFAULT_HOUSE_ID,
                        help=f"House ID to process (default: {DEFAULT_HOUSE_ID})")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT,
                        help=f"Experiment name (default: {DEFAULT_EXPERIMENT})")
    parser.add_argument("--max_iterations", type=int, default=2,
                        help="Max iterations (static experiments only, default: 2)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--input_path", type=str, default=None,
                        help="Input data directory (default: INPUT/HouseholdData)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress console output")
    parser.add_argument("--skip_visualization", action="store_true",
                        help="Skip visualization step")
    parser.add_argument("--minimal_output", action="store_true",
                        help="Delete intermediate pkl files (dynamic experiments only)")

    args = parser.parse_args()

    exp_config = get_experiment(args.experiment_name)
    is_dynamic = exp_config.threshold_schedule is not None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_path or str(
        _SCRIPT_DIR / "OUTPUT" / "experiments" / f"{exp_config.exp_id}_{timestamp}"
    )

    print(f"\n{'='*60}")
    if is_dynamic:
        print(f"Dynamic Threshold Experiment: {exp_config.exp_id}")
        print(f"House: {args.house_id}")
        print(f"Threshold schedule: {exp_config.threshold_schedule}")
    else:
        print(f"Static Experiment: {exp_config.exp_id}")
        print(f"House: {args.house_id}")
        print(f"Threshold: {exp_config.threshold}W, Max iterations: {args.max_iterations}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    from pipeline.runner import run_pipeline

    result = run_pipeline(
        house_id=args.house_id,
        experiment_name=args.experiment_name,
        output_path=output_path,
        max_iterations=args.max_iterations,
        input_path=args.input_path,
        quiet=args.quiet,
        skip_visualization=args.skip_visualization,
        minimal_output=args.minimal_output,
    )

    if result['success']:
        print(f"\nSuccess! Completed {result['iterations']} iterations")
        print(f"Output saved to: {output_path}")
    else:
        print(f"\nFailed: {result['error']}")


if __name__ == "__main__":
    main()
