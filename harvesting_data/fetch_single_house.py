#!/usr/bin/env python3
"""
Fetch data for a single house.

Usage:
    python -m harvesting_data.fetch_single_house --house 140
    python -m harvesting_data.fetch_single_house --house 140 --token YOUR_TOKEN
    python -m harvesting_data.fetch_single_house --house 140 --token-file path/to/tokens.csv
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, TOKEN_FILE, BACKUP_TOKEN_FILE
from .fetcher import update_house

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_token_for_house(token_file: str, house_id: str) -> str:
    """Load token for a specific house from CSV file."""
    df = pd.read_csv(token_file)
    row = df[df['ID'].astype(str) == str(house_id)]
    if row.empty:
        raise ValueError(f"House {house_id} not found in {token_file}")
    return str(row['Token'].iloc[0]).strip().lstrip("'").lstrip("=")


def load_backup_token_for_house(house_id: str) -> str:
    """Load backup token for a specific house from backup CSV file."""
    try:
        df = pd.read_csv(BACKUP_TOKEN_FILE)
        row = df[df['ID'].astype(str) == str(house_id)]
        if row.empty:
            return None
        token = str(row['Token'].iloc[0]).strip().lstrip("'").lstrip("=")
        return token if token and token != 'nan' else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch data for a single house")
    parser.add_argument(
        "--house",
        type=str,
        required=True,
        help="House ID to fetch"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="API token (if not using token file)"
    )
    parser.add_argument(
        "--token-file", "-f",
        type=str,
        default=str(TOKEN_FILE),
        help=f"Path to CSV file with house tokens (default: {TOKEN_FILE})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help=f"Output directory (default: {DATA_DIR})"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential fetching instead of parallel"
    )

    args = parser.parse_args()

    house_id = args.house

    # Get token
    if args.token:
        token = args.token
    else:
        try:
            token = load_token_for_house(args.token_file, house_id)
        except Exception as e:
            logger.error(f"Failed to load token: {e}")
            sys.exit(1)

    # Set output directory
    data_dir = Path(args.output) if args.output else DATA_DIR

    # Load backup token
    backup_token = load_backup_token_for_house(house_id)
    if backup_token:
        logger.info(f"Backup token available for house {house_id}")

    logger.info(f"Fetching house {house_id}")
    logger.info(f"Output directory: {data_dir}")

    try:
        success = update_house(
            house_id=house_id,
            token=token,
            data_dir=data_dir,
            use_parallel=not args.sequential,
            backup_token=backup_token
        )

        if success:
            logger.info(f"Successfully updated house {house_id}")
            sys.exit(0)
        else:
            logger.warning(f"No new data fetched for house {house_id}")
            sys.exit(0)  # Not an error - just no new data

    except Exception as e:
        logger.error(f"Failed to update house {house_id}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
