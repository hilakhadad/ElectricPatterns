"""
Unit tests for individual functions
These test specific functions in isolation
"""
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_expand_event():
    """Test the expand_event function from on_off_log"""
    print("\n" + "="*60)
    print("UNIT TEST: expand_event")
    print("="*60)

    from on_off_log import expand_event

    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2022-01-01 10:00', periods=10, freq='1min'),
        'w1_diff': [0, 0, 100, 200, 300, 200, 100, 0, 0, 0]
    })

    event = {
        'start': pd.Timestamp('2022-01-01 10:02'),
        'end': pd.Timestamp('2022-01-01 10:06'),
        'magnitude': 1000
    }

    # Test expand_event
    try:
        result_start, result_end, result_mag = expand_event(event, data, 'on', 'w1_diff')
        print(f"  [OK] expand_event executed successfully")
        print(f"    Original: start={event['start']}, end={event['end']}, mag={event['magnitude']}")
        print(f"    Expanded: start={result_start}, end={result_end}, mag={result_mag}")
        return True
    except Exception as e:
        print(f"  [FAIL] expand_event failed: {e}")
        return False

def test_data_util_paths():
    """Test that data_util paths are correctly configured"""
    print("\n" + "="*60)
    print("UNIT TEST: data_util paths")
    print("="*60)

    from data_util import (RAW_INPUT_DIRECTORY, OUTPUT_BASE_PATH,
                           LOGS_DIRECTORY, ERRORS_DIRECTORY, OUTPUT_ROOT)

    base_dir = Path(__file__).parent.parent

    tests_passed = True

    # Test INPUT path
    expected_input = str(base_dir / "INPUT" / "HouseholdData")
    if RAW_INPUT_DIRECTORY == expected_input:
        print(f"  [OK] RAW_INPUT_DIRECTORY correct: {RAW_INPUT_DIRECTORY}")
    else:
        print(f"  [FAIL] RAW_INPUT_DIRECTORY wrong!")
        print(f"    Expected: {expected_input}")
        print(f"    Got: {RAW_INPUT_DIRECTORY}")
        tests_passed = False

    # Test OUTPUT paths
    if "OUTPUT" in OUTPUT_BASE_PATH:
        print(f"  [OK] OUTPUT_BASE_PATH goes to OUTPUT: {OUTPUT_BASE_PATH}")
    else:
        print(f"  [FAIL] OUTPUT_BASE_PATH doesn't go to OUTPUT: {OUTPUT_BASE_PATH}")
        tests_passed = False

    if "OUTPUT" in LOGS_DIRECTORY:
        print(f"  [OK] LOGS_DIRECTORY goes to OUTPUT: {LOGS_DIRECTORY}")
    else:
        print(f"  [FAIL] LOGS_DIRECTORY doesn't go to OUTPUT: {LOGS_DIRECTORY}")
        tests_passed = False

    if "OUTPUT" in ERRORS_DIRECTORY:
        print(f"  [OK] ERRORS_DIRECTORY goes to OUTPUT: {ERRORS_DIRECTORY}")
    else:
        print(f"  [FAIL] ERRORS_DIRECTORY doesn't go to OUTPUT: {ERRORS_DIRECTORY}")
        tests_passed = False

    return tests_passed

def test_csv_structure():
    """Test that example.csv has the correct structure"""
    print("\n" + "="*60)
    print("UNIT TEST: Input CSV structure")
    print("="*60)

    from data_util import RAW_INPUT_DIRECTORY

    csv_path = Path(RAW_INPUT_DIRECTORY) / "example.csv"

    if not csv_path.exists():
        print(f"  [FAIL] example.csv not found at {csv_path}")
        return False

    try:
        df = pd.read_csv(csv_path)

        # Check columns
        expected_cols = ['timestamp', '1', '2', '3', 'sum']
        if list(df.columns) == expected_cols:
            print(f"  [OK] CSV has correct columns: {list(df.columns)}")
        else:
            print(f"  [FAIL] CSV has wrong columns!")
            print(f"    Expected: {expected_cols}")
            print(f"    Got: {list(df.columns)}")
            return False

        # Check data
        if len(df) > 0:
            print(f"  [OK] CSV has {len(df)} rows")
        else:
            print(f"  [FAIL] CSV is empty!")
            return False

        # Check that numeric columns are numeric
        for col in ['1', '2', '3', 'sum']:
            if df[col].dtype in [np.int64, np.float64]:
                print(f"  [OK] Column '{col}' is numeric")
            else:
                print(f"  [FAIL] Column '{col}' is not numeric: {df[col].dtype}")
                return False

        return True

    except Exception as e:
        print(f"  [FAIL] Failed to read CSV: {e}")
        return False

def run_unit_tests():
    """Run all unit tests"""
    print("="*60)
    print("UNIT TEST SUITE")
    print("="*60)

    tests = [
        ("Data Util Paths", test_data_util_paths),
        ("Input CSV Structure", test_csv_structure),
        ("Expand Event Function", test_expand_event),
    ]

    results = {'passed': [], 'failed': []}

    for test_name, test_func in tests:
        try:
            if test_func():
                results['passed'].append(test_name)
            else:
                results['failed'].append(test_name)
        except Exception as e:
            print(f"\n[FAIL] UNEXPECTED ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results['failed'].append(test_name)

    # Print summary
    print("\n" + "="*60)
    print("UNIT TEST SUMMARY")
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

    if failed > 0:
        print("[FAIL] UNIT TESTS FAILED")
        return 1
    else:
        print("[OK] ALL UNIT TESTS PASSED")
        return 0

if __name__ == "__main__":
    exit_code = run_unit_tests()
    sys.exit(exit_code)
