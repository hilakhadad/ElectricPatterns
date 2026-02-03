"""
File storage operations for household data.
"""
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd

from .config import DATA_DIR

logger = logging.getLogger(__name__)


def get_file_path(house_id: str, data_dir: Path = DATA_DIR) -> Path:
    """Get the file path for a house's data."""
    return data_dir / f"{house_id}.csv"


def get_latest_timestamp(file_path: Path) -> Optional[datetime]:
    """Get the latest timestamp from existing data file."""
    try:
        if not file_path.exists():
            return None
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        if df.empty:
            return None
        return df["timestamp"].max()
    except Exception as e:
        logger.warning(f"Could not read latest timestamp from {file_path}: {e}")
        return None


def save_data(
    house_id: str,
    new_data: pd.DataFrame,
    data_dir: Path = DATA_DIR
) -> int:
    """
    Save new data, merging with existing if present.

    Returns the total number of rows saved.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / f"{house_id}.csv"

    if file_path.exists():
        try:
            existing = pd.read_csv(file_path, parse_dates=["timestamp"])
            # Remove legacy 'sum' column if present
            existing = existing.drop(columns=["sum"], errors='ignore')
            combined = pd.concat([existing, new_data], ignore_index=True)
            combined = combined.drop_duplicates(subset="timestamp")
            combined = combined.sort_values("timestamp")
        except Exception as e:
            logger.warning(f"Could not merge with existing data: {e}")
            combined = new_data.sort_values("timestamp")
    else:
        combined = new_data.sort_values("timestamp")

    combined.to_csv(file_path, index=False)
    logger.info(f"Saved {len(combined)} rows for house {house_id} to {file_path}")

    return len(combined)


def load_house_tokens(path: str) -> list:
    """
    Load house IDs and tokens from CSV file.

    Expected CSV columns: ID, Token
    Returns list of (house_id, token) tuples.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"House list file not found: {path}")
        return []

    house_tokens = []
    for _, row in df.iterrows():
        house_id = str(row["ID"]).strip()
        token = str(row["Token"]).strip().lstrip("'")
        if house_id and token and token != 'nan':
            house_tokens.append((house_id, token))

    logger.info(f"Loaded {len(house_tokens)} houses from {path}")
    return house_tokens
