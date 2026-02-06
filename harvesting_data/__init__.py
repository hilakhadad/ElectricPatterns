"""
Harvesting module for fetching household power data from EnergyHive API.

Usage:
    # Update a single house
    python -m harvesting_data.fetch_single_house --house 140

    # Update all houses
    python -m harvesting_data.cli

    # Programmatic usage
    from harvesting_data import update_house
    update_house("140", "your-token-here")
"""
from .config import DATA_DIR, TOKEN_FILE, BACKUP_TOKEN_FILE
from .fetcher import update_house
from .storage import load_house_tokens, load_backup_tokens, save_data, get_latest_timestamp
from .cli import update_all_parallel, update_all_sequential

__all__ = [
    "DATA_DIR",
    "TOKEN_FILE",
    "BACKUP_TOKEN_FILE",
    "update_house",
    "load_house_tokens",
    "load_backup_tokens",
    "save_data",
    "get_latest_timestamp",
    "update_all_parallel",
    "update_all_sequential",
]
