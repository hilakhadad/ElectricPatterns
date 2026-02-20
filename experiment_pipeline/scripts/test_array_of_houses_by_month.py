"""
Test script to run the pipeline on multiple houses BY FILE ORDER.
Instead of completing all files for one house before moving to the next,
this script processes file 1 of ALL houses, then file 2 of ALL houses, etc.

This way you see results faster - houses with fewer files finish first,
and you quickly know if improvements exist across all houses.

Order:
  File 0: House A file 1, House B file 1, House C file 1...
  File 1: House A file 2, House B file 2, House C file 2...
  ...

Houses with fewer files finish earlier and you see partial results faster.

Usage:
    python test_array_of_houses_by_month.py                    # Run all houses
    python test_array_of_houses_by_month.py --houses 2,140    # Run specific houses only
    python test_array_of_houses_by_month.py --workers 2       # Limit parallel workers
"""
import sys
import os
import time
import importlib
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

# Experiment name (must match one in core/config.py)
EXPERIMENT_NAME = "exp010_dynamic_threshold"

# Maximum iterations (pipeline runs) per house
MAX_ITERATIONS = 2

# Auto-detect CPU cores
import multiprocessing
CPU_COUNT = multiprocessing.cpu_count()
MAX_WORKERS = min(4, CPU_COUNT)

# Skip visualization step (faster processing)
SKIP_VISUALIZATION = True


def get_houses_with_files():
    """Get house IDs with their input files, sorted by total size."""
    houses = {}

    for folder in _INPUT_PATH.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            files = sorted(folder.glob("*.csv"))
            if files:
                total_size = sum(f.stat().st_size for f in files)
                houses[folder.name] = {
                    'files': files,
                    'size': total_size
                }

    # Sort by size descending
    sorted_houses = sorted(houses.items(), key=lambda x: -x[1]['size'])
    return {h[0]: h[1] for h in sorted_houses}


def process_single_file(args):
    """
    Worker function for parallel processing of a single file through the pipeline.
    Must be at module level for multiprocessing to work.
    """
    house_id, input_file, file_index, run_number, experiment_name, output_path, _skip_visualization = args

    # Import here to avoid issues with multiprocessing
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    # Reload core.paths to update global paths for this run
    import core.paths
    importlib.reload(core.paths)

    # Override paths for this experiment
    core.paths.OUTPUT_BASE_PATH = output_path
    core.paths.OUTPUT_ROOT = output_path
    core.paths.INPUT_DIRECTORY = output_path
    core.paths.LOGS_DIRECTORY = f"{output_path}/logs/"
    core.paths.RAW_INPUT_DIRECTORY = str(_INPUT_PATH)

    # Reload core modules
    import core.logging_setup
    importlib.reload(core.logging_setup)
    import core
    importlib.reload(core)

    # Reload pipeline modules
    import pipeline
    importlib.reload(pipeline)
    import pipeline.detection
    import pipeline.matching
    import pipeline.segmentation
    import pipeline.evaluation
    importlib.reload(pipeline.detection)
    importlib.reload(pipeline.matching)
    importlib.reload(pipeline.segmentation)
    importlib.reload(pipeline.evaluation)

    process_detection = pipeline.detection.process_detection
    process_matching = pipeline.matching.process_matching
    process_segmentation = pipeline.segmentation.process_segmentation
    process_evaluation = pipeline.evaluation.process_evaluation

    from core import get_experiment

    try:
        exp_config = get_experiment(experiment_name)
        threshold = exp_config.threshold
    except KeyError as e:
        return {
            'house_id': house_id,
            'file_index': file_index,
            'success': False,
            'error': f"Unknown experiment: {e}"
        }

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/logs", exist_ok=True)

    try:
        # Run detection for this specific file
        process_detection(
            house_id=house_id,
            run_number=run_number,
            threshold=threshold,
            config=exp_config,
            input_file=str(input_file)
        )

        # Run matching - will process only new on_off files (skip logic built-in)
        process_matching(
            house_id=house_id,
            run_number=run_number,
            threshold=threshold
        )

        # Run segmentation - will process only new matches files (skip logic built-in)
        process_segmentation(
            house_id=house_id,
            run_number=run_number,
            skip_large_file=True
        )

        # Run evaluation
        process_evaluation(
            house_id=house_id,
            run_number=run_number,
            threshold=threshold
        )

        return {
            'house_id': house_id,
            'file_index': file_index,
            'file_name': input_file.name,
            'success': True,
            'error': None
        }

    except Exception as e:
        import traceback
        return {
            'house_id': house_id,
            'file_index': file_index,
            'file_name': input_file.name,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run pipeline by file order across all houses')
    parser.add_argument('--houses', type=str, default=None,
                        help='Comma-separated list of specific house IDs to run')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                        help=f'Number of parallel workers (default: {MAX_WORKERS})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: auto-generated with timestamp)')
    parser.add_argument('--run', type=int, default=0,
                        help='Run number (iteration) to process (default: 0)')
    args = parser.parse_args()

    # Get all houses with their files
    all_houses = get_houses_with_files()

    if args.houses:
        house_ids = [h.strip() for h in args.houses.split(',')]
        houses = {h: all_houses[h] for h in house_ids if h in all_houses}
    else:
        houses = all_houses

    if not houses:
        print("ERROR: No houses to process")
        return

    # Find maximum number of files across all houses
    max_files = max(len(h['files']) for h in houses.values())

    print("=" * 60)
    print("RUNNING PIPELINE BY FILE ORDER (across all houses)")
    print("=" * 60)
    print(f"Total houses: {len(houses)}")
    print(f"Max files per house: {max_files}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Run number: {args.run}")
    print(f"Parallel workers: {args.workers}")
    print("=" * 60 + "\n")

    # Create output directory
    if args.output:
        shared_output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shared_output_dir = str(_SCRIPT_DIR / "OUTPUT" / "experiments" / f"{EXPERIMENT_NAME}_{timestamp}")

    os.makedirs(shared_output_dir, exist_ok=True)
    print(f"Output directory: {shared_output_dir}\n")

    # Save run info
    with open(Path(shared_output_dir) / "_run_info.txt", 'w') as f:
        f.write(f"experiment={EXPERIMENT_NAME}\n")
        f.write(f"mode=by_file_order\n")
        f.write(f"started={datetime.now().isoformat()}\n")
        f.write(f"total_houses={len(houses)}\n")
        f.write(f"max_files={max_files}\n")
        f.write(f"run_number={args.run}\n")

    start_time = time.time()
    all_results = []
    completed_houses = set()

    # Process file by file (file 0 of all houses, then file 1, etc.)
    for file_index in range(max_files):
        # Get houses that have this file index
        active_tasks = []
        for house_id, house_data in houses.items():
            if house_id in completed_houses:
                continue
            if file_index < len(house_data['files']):
                input_file = house_data['files'][file_index]
                active_tasks.append((
                    house_id,
                    input_file,
                    file_index,
                    args.run,
                    EXPERIMENT_NAME,
                    f"{shared_output_dir}/house_{house_id}",
                    SKIP_VISUALIZATION
                ))
            else:
                # House has no more files
                completed_houses.add(house_id)

        if not active_tasks:
            print(f"\nAll houses completed at file index {file_index}")
            break

        print(f"\n{'='*50}")
        print(f"FILE INDEX {file_index} ({len(active_tasks)} houses)")
        print(f"{'='*50}")

        # Run in parallel
        results = []
        num_workers = min(args.workers, len(active_tasks))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_file, task): task[0] for task in active_tasks}

            futures_iter = as_completed(futures)
            if HAS_TQDM:
                futures_iter = tqdm(futures_iter, total=len(active_tasks),
                                    desc=f"File {file_index}", unit="house", leave=True)

            for future in futures_iter:
                house_id = futures[future]
                try:
                    result = future.result()
                    results.append(result)

                    status = "OK" if result['success'] else "FAIL"
                    if HAS_TQDM:
                        futures_iter.set_postfix(house=house_id, status=status)

                    if not result['success']:
                        print(f"  [{house_id}] FAILED: {result.get('error', 'unknown')}")

                except Exception as e:
                    print(f"  [{house_id}] EXCEPTION: {e}")
                    results.append({
                        'house_id': house_id,
                        'file_index': file_index,
                        'success': False,
                        'error': str(e)
                    })

        all_results.extend(results)

        # Summary for this file index
        successful = sum(1 for r in results if r['success'])
        print(f"  File {file_index}: {successful}/{len(active_tasks)} successful")

        # Save progress
        with open(Path(shared_output_dir) / "_progress.txt", 'w') as f:
            f.write(f"completed_file_indices={file_index + 1}/{max_files}\n")
            f.write(f"completed_houses={len(completed_houses)}/{len(houses)}\n")
            f.write(f"timestamp={datetime.now().isoformat()}\n")

    # Final summary
    total_time = time.time() - start_time
    successful_total = sum(1 for r in all_results if r['success'])
    failed_total = sum(1 for r in all_results if not r['success'])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {len(all_results)}")
    print(f"Successful: {successful_total}")
    print(f"Failed: {failed_total}")
    print(f"Houses completed: {len(completed_houses)}/{len(houses)}")

    if failed_total > 0:
        failed_results = [r for r in all_results if not r['success']]
        print(f"\nFailed files:")
        for r in failed_results[:10]:
            print(f"  - House {r['house_id']} file {r['file_index']}: {r.get('error', 'unknown')}")
        if len(failed_results) > 10:
            print(f"  ... and {len(failed_results)-10} more")

    print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"\nResults saved to: {shared_output_dir}")


if __name__ == "__main__":
    main()
