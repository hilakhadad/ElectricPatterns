"""
Configuration for the harvesting module.
"""
from pathlib import Path

# Paths
DATA_DIR = Path("/sise/shanigu-group/hilakese-dorins/PreprocessedData")

# API
API_BASE_URL = "http://www.energyhive.com/mobile_proxy/getHV"
REQUEST_TIMEOUT = 30  # seconds

# Retry settings
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 2  # seconds
MAX_RETRY_DELAY = 60  # seconds

# Fetching settings
MAX_DAYS_PER_REQUEST = 7
NUMBER_OF_YEARS = 6
MAX_CONCURRENT_REQUESTS = 4
MAX_HOUSE_WORKERS = 2
