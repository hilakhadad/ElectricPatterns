"""
Robust test script for the pipeline with proper assertions
This will FAIL if anything breaks during refactoring
"""
import sys
import os
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
HOUSE_ID = "example"
RUN_NUMBER = 0
THRESHOLD = 1600

# Expected outputs
OUTPUT_DIR = Path(__file__).parent.parent / "OUTPUT"
RESULTS_DIR = OUTPUT_DIR / f"run_{RUN_NUMBER}" / f"house_{HOUSE_ID}"
LOGS_DIR = OUTPUT_DIR / "logs"

class TestFailure(Exception):
    """Custom exception for test failures"""
    pass

def assert_file_exists(filepath, description):
    """Assert that a file exists and is not empty"""
    if not filepath.exists():
        raise TestFailure(f"FAIL: {description} file does not exist: {filepath}")

    if filepath.stat().st_size == 0:
        raise TestFailure(f"FAIL: {description} file is empty: {filepath}")

    print(f"  [OK] {description}: {filepath.name} ({filepath.stat().st_size:,} bytes)")
    return True

def assert_csv_valid(filepath, description, min_rows=1):
    """Assert that a CSV file is valid and has minimum rows"""
    try:
        df = pd.read_csv(filepath)
        if len(df) < min_rows:
            raise TestFailure(f"FAIL: {description} has only {len(df)} rows, expected at least {min_rows}")
        print(f"  [OK] {description}: {len(df)} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        raise TestFailure(f"FAIL: {description} CSV is invalid: {e}")

def test_step_1_on_off_detection():
    """Test on/off event detection"""
    print("\n" + "="*60)
    print("TEST 1: ON/OFF EVENT DETECTION")
    print("="*60)

    from on_off_log import process_house

    # Run the function
    print(f"Running: process_house('{HOUSE_ID}', {RUN_NUMBER}, {THRESHOLD})")
    process_house(HOUSE_ID, RUN_NUMBER, THRESHOLD)

    # Verify output file exists
    output_file = RESULTS_DIR / f"on_off_{THRESHOLD}.csv"
    assert_file_exists(output_file, "On/Off events file")

    # Verify CSV is valid
    df = assert_csv_valid(output_file, "On/Off events", min_rows=10)

    # Verify expected columns
    expected_cols = ['start', 'end', 'magnitude', 'duration', 'phase', 'event', 'event_id']
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise TestFailure(f"FAIL: Missing columns in on/off CSV: {missing_cols}")

    print(f"  [OK] All expected columns present")
    print(f"  [OK] Test 1 PASSED")
    return df

def test_step_2_event_matching():
    """Test event matching"""
    print("\n" + "="*60)
    print("TEST 2: EVENT MATCHING")
    print("="*60)

    from new_matcher import process_matches

    # Run the function
    print(f"Running: process_matches('{HOUSE_ID}', {RUN_NUMBER}, {THRESHOLD})")
    process_matches(HOUSE_ID, RUN_NUMBER, THRESHOLD)

    # Verify output files
    matches_file = RESULTS_DIR / f"matches_{HOUSE_ID}.csv"
    unmatched_on_file = RESULTS_DIR / f"unmatched_on_{HOUSE_ID}.csv"
    unmatched_off_file = RESULTS_DIR / f"unmatched_off_{HOUSE_ID}.csv"

    assert_file_exists(matches_file, "Matches file")
    assert_file_exists(unmatched_on_file, "Unmatched ON file")
    assert_file_exists(unmatched_off_file, "Unmatched OFF file")

    # Verify matches CSV
    df_matches = assert_csv_valid(matches_file, "Matches", min_rows=1)

    # Verify expected columns
    expected_cols = ['on_event_id', 'off_event_id', 'on_start', 'on_end',
                     'off_start', 'off_end', 'duration', 'on_magnitude',
                     'off_magnitude', 'tag', 'phase']
    missing_cols = set(expected_cols) - set(df_matches.columns)
    if missing_cols:
        raise TestFailure(f"FAIL: Missing columns in matches CSV: {missing_cols}")

    print(f"  [OK] All expected columns present")
    print(f"  [OK] Test 2 PASSED")
    return df_matches

def test_step_3_segmentation():
    """Test data segmentation"""
    print("\n" + "="*60)
    print("TEST 3: DATA SEGMENTATION")
    print("="*60)

    from segmentation import process_segmentation

    # Run the function
    print(f"Running: process_segmentation('{HOUSE_ID}', {RUN_NUMBER})")
    process_segmentation(HOUSE_ID, RUN_NUMBER)

    # Verify output files
    segmented_file = RESULTS_DIR / f"segmented_{HOUSE_ID}.csv"
    summarized_file = RESULTS_DIR / f"summarized_{HOUSE_ID}.csv"

    assert_file_exists(segmented_file, "Segmented file")
    assert_file_exists(summarized_file, "Summarized file")

    # Verify segmented CSV
    df_segmented = assert_csv_valid(segmented_file, "Segmented data", min_rows=100)

    # Verify core columns exist
    required_cols = ['timestamp', 'w1', 'w2', 'w3']
    missing_cols = set(required_cols) - set(df_segmented.columns)
    if missing_cols:
        raise TestFailure(f"FAIL: Missing core columns in segmented CSV: {missing_cols}")

    # Verify at least some event columns were created
    event_cols = [col for col in df_segmented.columns if 'event_power' in col]
    if len(event_cols) == 0:
        raise TestFailure(f"FAIL: No event_power columns found in segmented data")

    print(f"  [OK] Core columns present")
    print(f"  [OK] Found {len(event_cols)} event power columns")
    print(f"  [OK] Test 3 PASSED")
    return df_segmented

def test_log_file():
    """Test that log file was created"""
    print("\n" + "="*60)
    print("TEST 4: LOG FILE")
    print("="*60)

    log_file = LOGS_DIR / f"{HOUSE_ID}.log"
    assert_file_exists(log_file, "Log file")

    # Verify log contains expected messages
    with open(log_file, 'r', encoding='utf-8') as f:
        log_content = f.read()

    expected_messages = [
        "Detection On and Off events process",
        "Match process",
        "Segmentation process"
    ]

    for msg in expected_messages:
        if msg not in log_content:
            raise TestFailure(f"FAIL: Expected log message not found: '{msg}'")

    print(f"  [OK] All expected log messages present")
    print(f"  [OK] Test 4 PASSED")

def run_all_tests():
    """Run all tests and report results"""
    print("="*60)
    print("PIPELINE REGRESSION TEST SUITE")
    print("="*60)
    print(f"Testing house: {HOUSE_ID}")
    print(f"Run number: {RUN_NUMBER}")
    print(f"Threshold: {THRESHOLD}W")
    print(f"Output directory: {OUTPUT_DIR}")

    results = {
        'passed': [],
        'failed': []
    }

    tests = [
        ("On/Off Detection", test_step_1_on_off_detection),
        ("Event Matching", test_step_2_event_matching),
        ("Segmentation", test_step_3_segmentation),
        ("Log File", test_log_file),
    ]

    for test_name, test_func in tests:
        try:
            test_func()
            results['passed'].append(test_name)
        except TestFailure as e:
            print(f"\n[FAIL] {e}")
            results['failed'].append(test_name)
        except Exception as e:
            print(f"\n[FAIL] UNEXPECTED ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results['failed'].append(test_name)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total = len(results['passed']) + len(results['failed'])
    passed = len(results['passed'])
    failed = len(results['failed'])

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if results['passed']:
        print("\n[OK] PASSED:")
        for test in results['passed']:
            print(f"  - {test}")

    if results['failed']:
        print("\n[FAIL] FAILED:")
        for test in results['failed']:
            print(f"  - {test}")

    print("\n" + "="*60)

    # Return exit code (0 = success, 1 = failure)
    if failed > 0:
        print("[FAIL] TESTS FAILED - Pipeline is broken!")
        print("="*60)
        return 1
    else:
        print("[OK] ALL TESTS PASSED - Pipeline is working!")
        print("="*60)
        return 0

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
