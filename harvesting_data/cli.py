"""
Command-line interface for batch updating houses.
"""
import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import DATA_DIR, MAX_HOUSE_WORKERS, TOKEN_FILE, BACKUP_TOKEN_FILE
from .storage import load_house_tokens, load_backup_tokens
from .fetcher import update_house

HOUSE_LIST_CSV = str(TOKEN_FILE)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _update_house_wrapper(args: tuple) -> Tuple[str, bool, str]:
    """Wrapper for update_house that returns status info."""
    house_id, token, data_dir, backup_token = args
    try:
        success = update_house(house_id, token, data_dir, backup_token=backup_token)
        if success:
            return (house_id, True, "Updated successfully")
        else:
            return (house_id, False, "No data fetched")
    except Exception as e:
        return (house_id, False, f"Error: {e}")


def update_all_parallel(
    houses: List[Tuple[str, str]],
    data_dir: Path = DATA_DIR,
    max_workers: int = MAX_HOUSE_WORKERS,
    backup_tokens: dict = None
) -> dict:
    """Update all houses in parallel."""
    if not houses:
        logger.warning("No houses to update")
        return {"success": 0, "failed": 0, "total": 0}

    if backup_tokens is None:
        backup_tokens = {}

    logger.info(f"Updating {len(houses)} houses with {max_workers} workers")
    start_time = time.time()

    results = {"success": [], "failed": []}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_house = {
            executor.submit(_update_house_wrapper, (house_id, token, data_dir, backup_tokens.get(house_id))): house_id
            for house_id, token in houses
        }

        for i, future in enumerate(as_completed(future_to_house), 1):
            house_id = future_to_house[future]
            try:
                house_id, success, message = future.result()
                if success:
                    results["success"].append(house_id)
                    logger.info(f"[{i}/{len(houses)}] House {house_id}: {message}")
                else:
                    results["failed"].append((house_id, message))
                    logger.warning(f"[{i}/{len(houses)}] House {house_id}: {message}")
            except Exception as e:
                results["failed"].append((house_id, str(e)))
                logger.error(f"[{i}/{len(houses)}] House {house_id}: Exception - {e}")

    elapsed = time.time() - start_time
    _print_summary(results, len(houses), elapsed)

    return {
        "success": len(results["success"]),
        "failed": len(results["failed"]),
        "total": len(houses),
        "elapsed_seconds": elapsed,
        "failed_houses": results["failed"]
    }


def update_all_sequential(
    houses: List[Tuple[str, str]],
    data_dir: Path = DATA_DIR,
    backup_tokens: dict = None
) -> dict:
    """Update all houses sequentially."""
    if not houses:
        logger.warning("No houses to update")
        return {"success": 0, "failed": 0, "total": 0}

    if backup_tokens is None:
        backup_tokens = {}

    logger.info(f"Updating {len(houses)} houses sequentially")
    start_time = time.time()

    results = {"success": [], "failed": []}

    for i, (house_id, token) in enumerate(houses, 1):
        logger.info(f"[{i}/{len(houses)}] Processing house {house_id}...")
        try:
            backup_token = backup_tokens.get(house_id)
            success = update_house(house_id, token, data_dir, backup_token=backup_token)
            if success:
                results["success"].append(house_id)
            else:
                results["failed"].append((house_id, "No data fetched"))
        except Exception as e:
            results["failed"].append((house_id, str(e)))
            logger.error(f"House {house_id}: {e}")

    elapsed = time.time() - start_time
    _print_summary(results, len(houses), elapsed)

    return {
        "success": len(results["success"]),
        "failed": len(results["failed"]),
        "total": len(houses),
        "elapsed_seconds": elapsed,
        "failed_houses": results["failed"]
    }


def _print_summary(results: dict, total: int, elapsed: float):
    """Print summary of update operation."""
    logger.info(f"\n{'='*50}")
    logger.info("SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total: {total} | Success: {len(results['success'])} | Failed: {len(results['failed'])}")
    logger.info(f"Time: {elapsed:.1f}s")

    if results["failed"]:
        logger.info("\nFailed houses:")
        for house_id, reason in results["failed"]:
            logger.info(f"  - {house_id}: {reason}")


def main():
    parser = argparse.ArgumentParser(description="Batch update house data from EnergyHive API")
    parser.add_argument(
        "--houses",
        type=str,
        default=HOUSE_LIST_CSV,
        help=f"Path to CSV file with house tokens (default: {HOUSE_LIST_CSV})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=f"Output directory for data files (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_HOUSE_WORKERS,
        help=f"Number of parallel workers (default: {MAX_HOUSE_WORKERS})"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Process houses sequentially instead of in parallel"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only process these house IDs (comma-separated)"
    )

    args = parser.parse_args()

    # Load houses
    houses = load_house_tokens(args.houses)

    if not houses:
        logger.error("No houses to process")
        return

    # Load backup tokens
    backup_tokens = load_backup_tokens(str(BACKUP_TOKEN_FILE))
    logger.info(f"Loaded {len(backup_tokens)} backup tokens (used as primary when available)")

    # Filter if specified
    if args.filter:
        filter_ids = [x.strip() for x in args.filter.split(",")]
        houses = [(h, t) for h, t in houses if h in filter_ids]
        logger.info(f"Filtered to {len(houses)} houses: {filter_ids}")

    # Set output directory
    data_dir = Path(args.output) if args.output else DATA_DIR

    # Run update
    if args.sequential:
        update_all_sequential(houses, data_dir, backup_tokens)
    else:
        update_all_parallel(houses, data_dir, args.workers, backup_tokens)


if __name__ == "__main__":
    main()
