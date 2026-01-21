"""
Test script to run the pipeline on multiple houses in parallel.
Calls run_pipeline_for_house directly (no subprocess overhead).
"""
import sys
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

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
_INPUT_PATH = _SCRIPT_DIR / "INPUT" / "HouseholdData"

# Auto-detect all houses from INPUT directory
# HOUSE_IDS = sorted([f.stem for f in _INPUT_PATH.glob("*.csv") if f.stem.isdigit()])
HOUSE_IDS = ["1", "1001", "2049", "2035"]

# Experiment name (must match one in detection_config.py)
EXPERIMENT_NAME = "exp005_asymmetric_windows"

# Number of iterations per house
MAX_ITERATIONS = 2

# Maximum parallel workers
MAX_WORKERS = 8

# ============================================================================


def process_single_house(args):
    """
    Worker function for parallel processing.
    Must be at module level for multiprocessing to work.
    """
    house_id, experiment_name, house_output, max_iterations, input_path = args

    # Import here to avoid issues with multiprocessing
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from test_single_house import run_pipeline_for_house

    result = run_pipeline_for_house(
        house_id=house_id,
        experiment_name=experiment_name,
        output_path=house_output,
        max_iterations=max_iterations,
        input_path=input_path,
        quiet=True  # Suppress console output in parallel mode
    )

    return {
        'house_id': house_id,
        'success': result['success'],
        'iterations': result['iterations'],
        'error': result.get('error')
    }


def main():
    print("=" * 60)
    print("RUNNING PIPELINE ON MULTIPLE HOUSES")
    print("=" * 60)
    print(f"Houses: {len(HOUSE_IDS)} total")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Max iterations per house: {MAX_ITERATIONS}")
    print(f"Max parallel workers: {min(MAX_WORKERS, len(HOUSE_IDS))}")
    print("=" * 60 + "\n")

    if not HOUSE_IDS:
        print(f"ERROR: No house CSV files found in {_INPUT_PATH}")
        return

    # Create shared output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shared_output_dir = str(_SCRIPT_DIR / "OUTPUT" / "experiments" / f"{EXPERIMENT_NAME}_{timestamp}")
    os.makedirs(shared_output_dir, exist_ok=True)

    print(f"Output directory: {shared_output_dir}\n")

    # Prepare tasks
    input_path = str(_INPUT_PATH)
    tasks = [
        (house_id, EXPERIMENT_NAME, f"{shared_output_dir}/house_{house_id}", MAX_ITERATIONS, input_path)
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
                    print(f"[{house_id}] {status}")
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

    print(f"\nResults saved to: {shared_output_dir}")


if __name__ == "__main__":
    main()
