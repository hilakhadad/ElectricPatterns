"""
Test script to run the pipeline on multiple houses in parallel.
Uses test_single_house.py logic but runs multiple houses concurrently.
"""
import sys
import os
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars: pip install tqdm")

# ============================================================================
# CONFIGURATION - CHANGE THESE
# ============================================================================

# List of house IDs to process
# Auto-detect all houses from INPUT directory
_SCRIPT_DIR = Path(__file__).parent.parent.absolute()
_INPUT_PATH = _SCRIPT_DIR / "INPUT" / "HouseholdData"
HOUSE_IDS = [f.stem for f in _INPUT_PATH.glob("*.csv") if f.stem.isdigit() or f.stem in ['example', 'smaller_example']]

# Experiment name (must match one in detection_config.py)
EXPERIMENT_NAME = "exp004_noisy_matching"  # <- CHANGE THIS

# Number of iterations per house
MAX_ITERATIONS = 2

# Maximum parallel workers (set to len(HOUSE_IDS) or less based on your machine)
MAX_WORKERS = 4

# ============================================================================

# Get paths
LOCAL_INPUT_PATH = str(_SCRIPT_DIR / "INPUT" / "HouseholdData")
LOCAL_OUTPUT_BASE = str(_SCRIPT_DIR / "OUTPUT" / "experiments")


def run_single_house(house_id: str, experiment_name: str, output_path: str, max_iterations: int) -> dict:
    """
    Run the pipeline for a single house by calling test_single_house.py as subprocess.

    Returns dict with results.
    """
    print(f"[{house_id}] Starting...")

    # Normalize path to use forward slashes (works on Windows too)
    output_path_normalized = output_path.replace('\\', '/')

    # Create a temporary script that imports and runs test_single_house with our config
    script_content = f'''
import sys
import os
from pathlib import Path

# Override configuration before importing
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set config via environment or direct override
import test_single_house as tsh

# Override the module-level variables
tsh.HOUSE_ID = "{house_id}"
tsh.EXPERIMENT_NAME = "{experiment_name}"
tsh.MAX_ITERATIONS = {max_iterations}
tsh.LOCAL_OUTPUT_PATH = "{output_path_normalized}"

# Re-setup data_util paths
import data_util
data_util.OUTPUT_BASE_PATH = "{output_path_normalized}"
data_util.OUTPUT_ROOT = "{output_path_normalized}"
data_util.INPUT_DIRECTORY = "{output_path_normalized}"
data_util.LOGS_DIRECTORY = "{output_path_normalized}/logs/"

os.makedirs("{output_path_normalized}", exist_ok=True)
os.makedirs("{output_path_normalized}/logs", exist_ok=True)

# Run
tsh.main()
'''

    # Write temp script
    temp_script = Path(__file__).parent / f"_temp_run_{house_id}.py"
    try:
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(script_content)

        # Run it
        result = subprocess.run(
            [sys.executable, str(temp_script)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        success = result.returncode == 0

        if success:
            print(f"[{house_id}] Completed successfully")
        else:
            print(f"[{house_id}] Failed with return code {result.returncode}")
            if result.stderr:
                print(f"[{house_id}] Error: {result.stderr[:500]}")

        return {
            'house_id': house_id,
            'success': success,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    finally:
        # Clean up temp script
        if temp_script.exists():
            temp_script.unlink()


def main():
    print("="*60)
    print("RUNNING PIPELINE ON MULTIPLE HOUSES")
    print("="*60)
    print(f"Houses: {HOUSE_IDS}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Max iterations per house: {MAX_ITERATIONS}")
    print(f"Max parallel workers: {min(MAX_WORKERS, len(HOUSE_IDS))}")
    print("="*60 + "\n")

    # Validate input files exist
    missing = []
    for house_id in HOUSE_IDS:
        input_file = f"{LOCAL_INPUT_PATH}/{house_id}.csv"
        if not os.path.exists(input_file):
            missing.append(house_id)

    if missing:
        print(f"ERROR: Missing input files for houses: {missing}")
        print(f"Expected location: {LOCAL_INPUT_PATH}/")
        return

    # Create shared output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shared_output_dir = f"{LOCAL_OUTPUT_BASE}/{EXPERIMENT_NAME}_{timestamp}"
    os.makedirs(shared_output_dir, exist_ok=True)

    print(f"Output directory: {shared_output_dir}\n")

    # Prepare tasks - each house gets its own subdirectory
    tasks = []
    for house_id in HOUSE_IDS:
        house_output = f"{shared_output_dir}/house_{house_id}"
        tasks.append((house_id, EXPERIMENT_NAME, house_output, MAX_ITERATIONS))

    # Run in parallel with progress bar
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(HOUSE_IDS))) as executor:
        future_to_house = {
            executor.submit(run_single_house, *task): task[0]
            for task in tasks
        }

        # Use tqdm if available for progress bar
        futures_iter = concurrent.futures.as_completed(future_to_house)
        if HAS_TQDM:
            futures_iter = tqdm(
                futures_iter,
                total=len(HOUSE_IDS),
                desc="Processing houses",
                unit="house"
            )

        for future in futures_iter:
            house_id = future_to_house[future]
            try:
                result = future.result()
                results.append(result)
                if HAS_TQDM:
                    futures_iter.set_postfix(last=f"House {house_id}", status="OK" if result['success'] else "FAIL")
            except Exception as e:
                print(f"[{house_id}] Exception: {e}")
                results.append({
                    'house_id': house_id,
                    'success': False,
                    'error': str(e)
                })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    print(f"Successful: {len(successful)}/{len(HOUSE_IDS)}")
    for r in successful:
        print(f"  - House {r['house_id']}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(HOUSE_IDS)}")
        for r in failed:
            print(f"  - House {r['house_id']}: {r.get('error', 'see logs')}")

    print(f"\nResults saved to: {shared_output_dir}")


if __name__ == "__main__":
    main()
