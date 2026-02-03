"""
Main fetching logic for household data.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from .config import (
    DATA_DIR,
    MAX_DAYS_PER_REQUEST,
    NUMBER_OF_YEARS,
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
    use_parallel: bool = True
) -> bool:
    """
    Update data for a single house.

    Returns True if successful, False otherwise.
    """
    file_path = get_file_path(house_id, data_dir)
    latest_timestamp = get_latest_timestamp(file_path)

    # Determine time range
    if latest_timestamp is None:
        from_time = int((datetime.now() - timedelta(days=365 * NUMBER_OF_YEARS)).timestamp())
        logger.info(f"House {house_id}: No existing data, fetching {NUMBER_OF_YEARS} years")
    else:
        from_time = int((latest_timestamp + timedelta(minutes=1)).timestamp())
        logger.info(f"House {house_id}: Updating from {latest_timestamp}")

    to_time = int(datetime.now().timestamp())

    # Skip if no new data needed
    if from_time >= to_time:
        logger.info(f"House {house_id}: Already up to date")
        return True

    # Fetch data
    if use_parallel:
        data = fetch_parallel(token, from_time, to_time)
    else:
        data = fetch_sequential(token, from_time, to_time)

    if data is None or data.empty:
        logger.warning(f"House {house_id}: No new data fetched")
        return False

    # Save data
    save_data(house_id, data, data_dir)
    return True
