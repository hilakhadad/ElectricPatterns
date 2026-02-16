"""
Run the pipeline on multiple houses in parallel.

Supports both static and dynamic threshold experiments automatically.
The mode is determined by the experiment config (same as test_single_house.py).

Performance optimized:
- Auto-detects CPU cores for optimal parallelization
- Tracks time per house for ETA estimation
- Sorts houses by file size (largest first) for better load balancing

Usage:
    # Default (dynamic threshold, exp010):
    python scripts/test_array_of_houses.py

    # Static experiment (legacy):
    python scripts/test_array_of_houses.py --experiment_name exp007_symmetric_threshold

    # With options:
    python scripts/test_array_of_houses.py --skip_visualization --minimal_output
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Fix encoding for Windows console (safer approach)
if sys.platform == 'win32':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8
    except Exception:
        pass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# ============================================================================
# CONFIGURATION
# ============================================================================

_SCRIPT_DIR = Path(__file__).parent.parent.absolute()
_PROJECT_ROOT = _SCRIPT_DIR.parent
_INPUT_PATH = _PROJECT_ROOT / "INPUT" / "HouseholdData"


def get_houses_sorted_by_size():
    """Get house IDs sorted by file size (largest first) for better load balancing.

    Supports both old (single CSV) and new (folder with monthly files) structures.
    """
    houses = []

    # Check for folders (new structure: HouseholdData/1001/)
    for folder in _INPUT_PATH.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            total_size = sum(f.stat().st_size for f in folder.glob("*.csv"))
            houses.append((folder.name, total_size))

    # Check for CSV files (old structure: HouseholdData/1001.csv)
    for f in _INPUT_PATH.glob("*.csv"):
        if f.stem.isdigit():
            houses.append((f.stem, f.stat().st_size))

    # Sort by size descending so large files start first
    houses.sort(key=lambda x: -x[1])
    return [h[0] for h in houses]


# Auto-detect all houses from INPUT directory, sorted by size
# HOUSE_IDS = get_houses_sorted_by_size()
HOUSE_IDS = ["305", "1"]

# Auto-detect CPU cores
# NOTE: More workers != faster! Often 2-4 workers is optimal due to:
# - Disk I/O bottleneck (all processes read/write same disk)
# - Memory pressure (each process needs ~2-4GB RAM)
# - CPU thermal throttling
import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = min(4, CPU_COUNT)  # Try 2, 3, or 4 and see which is fastest

# ============================================================================


def process_single_house(args):
    """
    Worker function for parallel processing.
    Must be at module level for multiprocessing to work.
    """
    house_id, experiment_name, output_path, max_iterations, input_path, skip_visualization, minimal_output = args

    # Import here to avoid issues with multiprocessing
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from pipeline.runner import run_pipeline

    result = run_pipeline(
        house_id=house_id,
        experiment_name=experiment_name,
        output_path=output_path,
        max_iterations=max_iterations,
        input_path=input_path,
        quiet=True,  # Suppress console output in parallel mode
        skip_visualization=skip_visualization,
        minimal_output=minimal_output,
    )

    return {
        'house_id': house_id,
        'success': result['success'],
        'iterations': result['iterations'],
        'error': result.get('error'),
    }


def main():
    import argparse

    # Add src to path for config import
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from core.config import DEFAULT_EXPERIMENT, get_experiment

    parser = argparse.ArgumentParser(description="Run pipeline on multiple houses in parallel")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT,
                        help=f"Experiment name (default: {DEFAULT_EXPERIMENT})")
    parser.add_argument("--max_iterations", type=int, default=2,
                        help="Max iterations per house (static experiments only, default: 2)")
    parser.add_argument("--skip_visualization", action="store_true",
                        help="Skip visualization step (recommended for batch)")
    parser.add_argument("--minimal_output", action="store_true",
                        help="Delete intermediate pkl files (dynamic experiments only)")
    args = parser.parse_args()

    exp_config = get_experiment(args.experiment_name)
    is_dynamic = exp_config.threshold_schedule is not None

    start_time = time.time()

    print("=" * 60)
    if is_dynamic:
        print("DYNAMIC THRESHOLD PIPELINE — MULTIPLE HOUSES")
        print("=" * 60)
        print(f"Houses: {len(HOUSE_IDS)} total")
        print(f"Experiment: {args.experiment_name}")
        print(f"Threshold schedule: {exp_config.threshold_schedule}")
    else:
        print("STATIC PIPELINE — MULTIPLE HOUSES")
        print("=" * 60)
        print(f"Houses: {len(HOUSE_IDS)} total")
        print(f"Experiment: {args.experiment_name}")
        print(f"Threshold: {exp_config.threshold}W, Max iterations: {args.max_iterations}")
    print(f"Skip visualization: {args.skip_visualization}")
    print(f"Parallel workers: {min(MAX_WORKERS, len(HOUSE_IDS))} (CPU cores: {CPU_COUNT})")
    print("=" * 60 + "\n")

    if not HOUSE_IDS:
        print(f"ERROR: No house CSV files found in {_INPUT_PATH}")
        return

    # Create shared output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shared_output_dir = str(
        _SCRIPT_DIR / "OUTPUT" / "experiments" / f"{exp_config.exp_id}_{timestamp}"
    )
    os.makedirs(shared_output_dir, exist_ok=True)

    print(f"Output directory: {shared_output_dir}\n")

    # Prepare tasks — all houses share the same output root
    input_path = str(_INPUT_PATH)
    tasks = [
        (house_id, args.experiment_name, shared_output_dir,
         args.max_iterations, input_path, args.skip_visualization, args.minimal_output)
        for house_id in HOUSE_IDS
    ]

    # Run in parallel
    results = []
    num_workers = min(MAX_WORKERS, len(HOUSE_IDS))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_house, task): task[0] for task in tasks}

        # Progress bar
        futures_iter = as_completed(futures)
        if HAS_TQDM:
            futures_iter = tqdm(futures_iter, total=len(HOUSE_IDS), desc="Processing", unit="house")

        for future in futures_iter:
            house_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "OK" if result['success'] else "FAIL"
                if HAS_TQDM:
                    futures_iter.set_postfix(last=house_id, status=status)
                else:
                    print(f"[{house_id}] {status} ({result.get('iterations', 0)} iterations)")
            except Exception as e:
                print(f"[{house_id}] Exception: {e}")
                results.append({'house_id': house_id, 'success': False, 'error': str(e)})

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    print(f"Successful: {len(successful)}/{len(HOUSE_IDS)}")

    if failed:
        print(f"Failed: {len(failed)}/{len(HOUSE_IDS)}")
        for r in failed:
            print(f"  - House {r['house_id']}: {r.get('error', 'unknown error')}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    if HOUSE_IDS:
        print(f"Average per house: {total_time/len(HOUSE_IDS):.1f}s")
    print(f"\nResults saved to: {shared_output_dir}")


if __name__ == "__main__":
    main()
