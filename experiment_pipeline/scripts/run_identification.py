"""
Run device identification (session grouping + classification) on existing
disaggregation output.

Usage:
    python scripts/run_identification.py --experiment_dir OUTPUT/experiments/exp010_XXX --house_id 305
    python scripts/run_identification.py --experiment_dir OUTPUT/experiments/exp010_XXX --house_id 305 --session_gap 30
"""
import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is importable
_SCRIPT_DIR = Path(__file__).parent.absolute()
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Run identification on disaggregation output"
    )
    parser.add_argument("--experiment_dir", required=True, type=str,
                        help="Root experiment output directory")
    parser.add_argument("--house_id", required=True, type=str,
                        help="House ID to process")
    parser.add_argument("--session_gap", type=int, default=30,
                        help="Gap threshold in minutes for session grouping (default: 30)")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        logger.error(f"Experiment directory does not exist: {experiment_dir}")
        sys.exit(1)

    # Read threshold_schedule from experiment metadata
    metadata_path = experiment_dir / "experiment_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        threshold_schedule = metadata.get('experiment', {}).get('threshold_schedule')
        if not threshold_schedule:
            threshold_schedule = metadata.get('threshold_schedule', [2000, 1500, 1100, 800])
    else:
        logger.warning("No experiment_metadata.json found, using default threshold schedule")
        threshold_schedule = [2000, 1500, 1100, 800]

    logger.info(f"Running identification for house {args.house_id}")
    logger.info(f"Experiment dir: {experiment_dir}")
    logger.info(f"Threshold schedule: {threshold_schedule}")
    logger.info(f"Session gap: {args.session_gap} min")

    from identification import (
        load_all_matches,
        deduplicate_cross_iteration,
        group_into_sessions,
        classify_sessions,
        build_session_json,
    )

    all_matches = load_all_matches(experiment_dir, args.house_id, threshold_schedule)
    if all_matches.empty:
        logger.error("No match files found. Ensure disaggregation has been run first.")
        sys.exit(1)

    deduped = deduplicate_cross_iteration(all_matches)
    sessions = group_into_sessions(deduped, gap_minutes=args.session_gap)
    classified = classify_sessions(sessions, deduped)
    json_path = build_session_json(
        classified_sessions=classified,
        house_id=args.house_id,
        threshold_schedule=threshold_schedule,
        experiment_dir=experiment_dir,
        session_gap_minutes=args.session_gap,
    )

    # Print summary
    total = sum(len(v) for v in classified.values())
    print(f"\nIdentification complete:")
    for dtype, sessions_list in classified.items():
        if sessions_list:
            print(f"  {dtype}: {len(sessions_list)} sessions")
    print(f"  Total: {total} sessions")
    print(f"  Output: {json_path}")


if __name__ == '__main__':
    main()
