"""
API communication with EnergyHive.
"""
import time
import logging
from datetime import datetime
from typing import Optional, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import (
    API_BASE_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    INITIAL_RETRY_DELAY,
    MAX_RETRY_DELAY,
)

logger = logging.getLogger(__name__)


def create_session() -> requests.Session:
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def build_url(token: str, from_time: int, to_time: int) -> str:
    """Build API URL for data request."""
    return (
        f"{API_BASE_URL}?fromTime={from_time}&toTime={to_time}&"
        f"aggPeriod=minute&aggFunc=sum&offset=-180&type=PWER&period=custom&token={token}"
    )


def format_timestamp(ts: int) -> str:
    """Format unix timestamp to readable string."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def parse_response(data: List) -> Optional[pd.DataFrame]:
    """
    Parse API response data into DataFrame.

    Returns None if data is invalid or empty.
    """
    if not data or len(data) < 3:
        return None

    try:
        d1, d2, d3 = data[0]["data"], data[1]["data"], data[2]["data"]

        if len(d1) <= 1:
            return None

        rows = []
        for i in range(1, len(d1)):
            key = list(d1[i].keys())[0]
            timestamp = datetime.fromtimestamp(int(key[:-3]))

            row = [
                timestamp,
                None if d1[i].get(key) == 'undef' else d1[i].get(key),
                None if d2[i].get(key) == 'undef' else d2[i].get(key),
                None if d3[i].get(key) == 'undef' else d3[i].get(key)
            ]
            rows.append(row)

        if not rows:
            return None

        return pd.DataFrame(rows, columns=["timestamp", "1", "2", "3"])

    except (KeyError, IndexError, TypeError) as e:
        logger.warning(f"Error parsing API response: {e}")
        return None


def fetch_time_range(
    session: requests.Session,
    token: str,
    from_time: int,
    to_time: int
) -> Optional[pd.DataFrame]:
    """
    Fetch data for a specific time range with retry logic.

    Retries on:
    - Network errors
    - API errors
    - Empty responses (the website sometimes returns empty data)
    """
    delay = INITIAL_RETRY_DELAY
    url = build_url(token, from_time, to_time)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            json_data = response.json()

            # Check for API error
            if json_data.get('status') == 'error':
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES} ({format_timestamp(from_time)}): "
                    f"API error - {json_data.get('message', 'unknown')}"
                )
                time.sleep(delay)
                delay = min(delay * 2, MAX_RETRY_DELAY)
                continue

            # Parse the data
            data = json_data.get("data", [])
            df = parse_response(data)

            # Retry on empty response
            if df is None or df.empty:
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES} ({format_timestamp(from_time)}): "
                    f"Empty response, retrying..."
                )
                time.sleep(delay)
                delay = min(delay * 2, MAX_RETRY_DELAY)
                continue

            # Success
            logger.debug(f"Fetched {len(df)} rows for {format_timestamp(from_time)}")
            return df

        except requests.exceptions.Timeout:
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES} ({format_timestamp(from_time)}): "
                f"Request timeout"
            )
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Attempt {attempt}/{MAX_RETRIES} ({format_timestamp(from_time)}): "
                f"Request error: {e}"
            )
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)

        except Exception as e:
            logger.error(
                f"Attempt {attempt}/{MAX_RETRIES} ({format_timestamp(from_time)}): "
                f"Unexpected error: {e}"
            )
            time.sleep(delay)
            delay = min(delay * 2, MAX_RETRY_DELAY)

    logger.error(
        f"Failed to fetch data for {format_timestamp(from_time)} to "
        f"{format_timestamp(to_time)} after {MAX_RETRIES} attempts"
    )
    return None
