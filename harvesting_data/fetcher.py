"""
Main fetching logic for household data.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from .config import (
    DATA_DIR,
    MAX_DAYS_PER_REQUEST,
    START_DATE,
    MAX_CONCURRENT_REQUESTS,
)
from .api import create_session, fetch_time_range, format_timestamp
from .storage import get_file_path, get_latest_timestamp, save_data

logger = logging.getLogger(__name__)


def generate_time_ranges(from_time: int, to_time: int) -> List[Tuple[int, int]]:
    """Generate list of time ranges to fetch."""
    ranges = []
    current_start = from_time
    chunk_seconds = MAX_DAYS_PER_REQUEST * 86400

    while current_start < to_time:
        current_end = min(current_start + chunk_seconds, to_time)
        ranges.append((current_start, current_end))
        current_start = current_end

    return ranges


def fetch_parallel(
    token: str,
    from_time: int,
    to_time: int,
    max_workers: int = MAX_CONCURRENT_REQUESTS
) -> Optional[pd.DataFrame]:
    """Fetch data using parallel requests for speed."""
    time_ranges = generate_time_ranges(from_time, to_time)

    if not time_ranges:
        return None

    logger.info(f"Fetching {len(time_ranges)} time ranges with {max_workers} workers")

    all_dfs = []
    session = create_session()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_range = {
            executor.submit(fetch_time_range, session, token, start, end): (start, end)
            for start, end in time_ranges
        }

        for future in as_completed(future_to_range):
            time_range = future_to_range[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    all_dfs.append(df)
            except Exception as e:
                logger.error(f"Error fetching {format_timestamp(time_range[0])}: {e}")

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"])
    combined = combined.sort_values("timestamp")

    logger.info(f"Fetched total of {len(combined)} unique rows")
    return combined


def fetch_sequential(
    token: str,
    from_time: int,
    to_time: int
) -> Optional[pd.DataFrame]:
    """Fetch data sequentially (fallback method)."""
    time_ranges = generate_time_ranges(from_time, to_time)

    if not time_ranges:
        return None

    logger.info(f"Fetching {len(time_ranges)} time ranges sequentially")

    all_dfs = []
    session = create_session()

    for start, end in time_ranges:
        df = fetch_time_range(session, token, start, end)
        if df is not None and not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"])
    combined = combined.sort_values("timestamp")

    return combined


def update_house(
    house_id: str,
    token: str,
    data_dir: Path = DATA_DIR,
    use_parallel: bool = True,
    backup_token: Optional[str] = None
) -> bool:
    """
    Update data for a single house.

    Fetches data incrementally and saves after each week to avoid losing
    progress if the job times out.

    If a backup_token is provided and different from the primary token,
    it will be used INSTEAD of the primary token (not as fallback).

    Returns True if successful, False otherwise.
    """
    # Use backup token as primary if available and different
    if backup_token and backup_token != token:
        logger.info(f"House {house_id}: Using backup token instead of primary")
        token_to_use = backup_token
    else:
        token_to_use = token

    return _try_update_house(house_id, token_to_use, data_dir)


def _try_update_house(
    house_id: str,
    token: str,
    data_dir: Path = DATA_DIR
) -> bool:
    """
    Internal function to attempt updating a house with a single token.
    """
    file_path = get_file_path(house_id, data_dir)
    latest_timestamp = get_latest_timestamp(file_path)

    # Determine time range
    if latest_timestamp is None:
        from_time = int(START_DATE.timestamp())
        logger.info(f"House {house_id}: No existing data, fetching from {START_DATE.strftime('%Y-%m-%d')}")
    else:
        from_time = int((latest_timestamp + timedelta(minutes=1)).timestamp())
        logger.info(f"House {house_id}: Updating from {latest_timestamp}")

    to_time = int(datetime.now().timestamp())

    # Skip if no new data needed
    if from_time >= to_time:
        logger.info(f"House {house_id}: Already up to date")
        return True

    # Generate time ranges and fetch incrementally
    time_ranges = generate_time_ranges(from_time, to_time)
    logger.info(f"House {house_id}: {len(time_ranges)} weeks to fetch")

    session = create_session()
    total_rows_saved = 0

    pbar = tqdm(
        enumerate(time_ranges, 1),
        total=len(time_ranges),
        desc=f"House {house_id}",
        unit="week",
        ncols=80
    )

    for i, (start, end) in pbar:
        df = fetch_time_range(session, token, start, end)

        if df is not None and not df.empty:
            # Save immediately after each week
            rows = save_data(house_id, df, data_dir)
            total_rows_saved = rows
            pbar.set_postfix(rows=rows)
        else:
            pbar.set_postfix(status="empty")

    if total_rows_saved == 0:
        logger.warning(f"House {house_id}: No new data fetched")
        return False

    logger.info(f"House {house_id}: Completed! Total rows: {total_rows_saved}")
    return True
