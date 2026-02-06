"""
Configuration for the harvesting module.
"""
import os
from datetime import datetime
from pathlib import Path

# Paths - detect environment
if os.path.exists("/sise/shanigu-group"):
    # Cluster environment
    DATA_DIR = Path("/sise/shanigu-group/hilakese-dorins/UpdateHouseholdData")
else:
    # Local environment
    DATA_DIR = Path(__file__).parent.parent / "INPUT" / "UpdateHouseholdData"

TOKEN_FILE = Path(__file__).parent.parent / "INPUT" / "id_token.csv"
BACKUP_TOKEN_FILE = Path(__file__).parent.parent / "INPUT" / "mishkit_backup.csv"

# API
API_BASE_URL = "http://www.energyhive.com/mobile_proxy/getHV"
REQUEST_TIMEOUT = 10  # seconds

# Retry settings
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 0.5  # seconds
MAX_RETRY_DELAY = 15  # seconds

# Fetching settings
MAX_DAYS_PER_REQUEST = 7
START_DATE = datetime(2020, 1, 1)  # Fetch data from this date
MAX_CONCURRENT_REQUESTS = 1  # Sequential to avoid API rate limiting
MAX_HOUSE_WORKERS = 1
