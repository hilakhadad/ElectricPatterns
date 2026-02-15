"""
Run the dynamic threshold pipeline on multiple houses in parallel.

Calls run_dynamic_pipeline_for_house directly (no subprocess overhead).
Each house runs through the full threshold schedule [2000, 1500, 1100, 800].

Usage:
    python scripts/test_dynamic_threshold_array.py
    python scripts/test_dynamic_threshold_array.py --skip_visualization
"""
import sys
import os
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Fix encoding for Windows console
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

HOUSE_IDS = ["305", "1"]

EXPERIMENT_NAME = "exp010_dynamic_threshold"

import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = min(4, CPU_COUNT)

SKIP_VISUALIZATION = False

# ============================================================================


def process_single_house(args):
    """Worker function for parallel processing."""
    house_id, experiment_name, house_output, input_path, skip_visualization, minimal_output = args

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from test_dynamic_threshold import run_dynamic_pipeline_for_house

    result = run_dynamic_pipeline_for_house(
        house_id=house_id,
        experiment_name=experiment_name,
        output_path=house_output,
        input_path=input_path,
        quiet=True,
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

    parser = argparse.ArgumentParser(
        description="Run dynamic threshold pipeline on multiple houses"
    )
    parser.add_argument("--skip_visualization", action="store_true",
                        help="Skip visualization step (recommended for batch)")
    parser.add_argument("--minimal_output", action="store_true",
                        help="Delete intermediate pkl files after building unified JSON")
    args = parser.parse_args()

    skip_viz = args.skip_visualization or SKIP_VISUALIZATION
    minimal_output = args.minimal_output

    start_time = time.time()

    print("=" * 60)
    print("DYNAMIC THRESHOLD PIPELINE — MULTIPLE HOUSES")
    print("=" * 60)
    print(f"Houses: {len(HOUSE_IDS)} total")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Skip visualization: {skip_viz}")
    print(f"Parallel workers: {min(MAX_WORKERS, len(HOUSE_IDS))} (CPU cores: {CPU_COUNT})")
    print("=" * 60 + "\n")

    if not HOUSE_IDS:
        print(f"ERROR: No houses configured")
        return

    # Create shared output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shared_output_dir = str(
        _SCRIPT_DIR / "OUTPUT" / "experiments" / f"exp010_{timestamp}"
    )
    os.makedirs(shared_output_dir, exist_ok=True)

    print(f"Output directory: {shared_output_dir}\n")

    # Prepare tasks
    input_path = str(_INPUT_PATH)
    tasks = [
        (house_id, EXPERIMENT_NAME, shared_output_dir, input_path, skip_viz, minimal_output)
        for house_id in HOUSE_IDS
    ]

    # Run in parallel
    results = []
    num_workers = min(MAX_WORKERS, len(HOUSE_IDS))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_house, task): task[0] for task in tasks}

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
