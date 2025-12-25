"""
Master test runner - runs all test suites
Use this to verify the pipeline after refactoring
"""
import sys
import subprocess
from pathlib import Path

def run_test_file(test_file, description):
    """Run a test file and return exit code"""
    print("\n" + "="*60)
    print(f"= {description}")
    print("="*60 + "\n")

    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=False
    )

    return result.returncode

def main():
    print("\n")
    print("=" + "="*58 + "=")
    print("|" + " "*58 + "|")
    print("|" + "  PIPELINE TEST SUITE - COMPLETE VALIDATION".center(58) + "|")
    print("|" + " "*58 + "|")
    print("=" + "="*58 + "=")

    base_dir = Path(__file__).parent

    # List of test suites to run
    test_suites = [
        (base_dir / "test_unit.py", "UNIT TESTS - Testing individual functions"),
        (base_dir / "test_pipeline.py", "INTEGRATION TESTS - Testing full pipeline"),
    ]

    results = {}

    for test_file, description in test_suites:
        if not test_file.exists():
            print(f"\n❌ Test file not found: {test_file}")
            results[description] = 1
            continue

        exit_code = run_test_file(test_file, description)
        results[description] = exit_code

    # Final summary
    print("\n")
    print("=" + "="*58 + "=")
    print("|" + " "*58 + "|")
    print("|" + "  FINAL TEST SUMMARY".center(58) + "|")
    print("|" + " "*58 + "|")
    print("=" + "="*58 + "=")
    print()

    all_passed = True
    for suite_name, exit_code in results.items():
        status = "PASSED" if exit_code == 0 else "FAILED"
        print(f"{status:12} - {suite_name}")
        if exit_code != 0:
            all_passed = False

    print()
    print("="*60)

    if all_passed:
        print("*** ALL TEST SUITES PASSED! ***")
        print("="*60)
        print("\n[PASS] The pipeline is working correctly")
        print("[PASS] Safe to proceed with refactoring")
        print("[PASS] All outputs are being generated as expected")
        print()
        return 0
    else:
        print("*** SOME TESTS FAILED! ***")
        print("="*60)
        print("\n[FAIL] The pipeline has issues")
        print("[FAIL] Do NOT refactor until tests pass")
        print("[FAIL] Check the output above for details")
        print()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
