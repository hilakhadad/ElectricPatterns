"""
DEPRECATED: Use harvesting_data module directly.

This file is kept for backwards compatibility.
"""
from .config import DATA_DIR
from .fetcher import update_house as update_single_house
from .fetcher import fetch_parallel as fetch_data_from_api
from .storage import save_data as append_new_data

__all__ = [
    "DATA_DIR",
    "update_single_house",
    "fetch_data_from_api",
    "append_new_data",
]
