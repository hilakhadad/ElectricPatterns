# Harvesting Data

Fetches household power consumption data from the EnergyHive API. Supports incremental updates (only fetches data since the last saved timestamp), weekly chunked requests, and batch processing of multiple houses.

## Quick Start

```bash
# All houses (sequential by default, 1 worker)
python -m harvesting_data.cli

# All houses with custom worker count
python -m harvesting_data.cli --workers 3

# Force sequential processing
python -m harvesting_data.cli --sequential

# Single house
python -m harvesting_data.fetch_single_house --house 140

# Single house with explicit token
python -m harvesting_data.fetch_single_house --house 140 --token YOUR_TOKEN

# Filter specific houses from batch
python -m harvesting_data.cli --filter 140,305,2008
```

## Structure

```
harvesting_data/
├── api.py                # EnergyHive API client (session, URL building, response parsing)
├── fetcher.py            # Fetch logic (incremental updates, weekly saves, retry)
├── storage.py            # Data I/O (load/save CSV, token loading, merge with existing)
├── config.py             # Configuration (paths, API settings, retry parameters)
├── cli.py                # Batch CLI (multi-house processing, parallel/sequential)
└── fetch_single_house.py # Single-house CLI
```

## Configuration

Defined in `config.py`:

```python
TOKEN_FILE = Path("../INPUT/id_token.csv")            # Primary token source
BACKUP_TOKEN_FILE = Path("../INPUT/mishkit_backup.csv")  # Backup tokens (used as primary when available)

DATA_DIR = Path("../INPUT/UpdateHouseholdData")   # Local output directory
# Cluster: /sise/shanigu-group/hilakese-dorins/UpdateHouseholdData

MAX_DAYS_PER_REQUEST = 7       # Days per API request chunk
MAX_CONCURRENT_REQUESTS = 1    # Sequential to avoid API rate limiting
MAX_HOUSE_WORKERS = 1          # Houses processed in parallel
REQUEST_TIMEOUT = 10           # Seconds per request

MAX_RETRIES = 5                # Retry count on failure
INITIAL_RETRY_DELAY = 0.5      # Seconds (doubles on each retry)
MAX_RETRY_DELAY = 15           # Seconds (cap for exponential backoff)

START_DATE = datetime(2020, 1, 1)  # Fetch data from this date for new houses
```

## Token Files

The module uses two token files. When a backup token exists for a house and differs from the primary token, the **backup token is used instead** of the primary (not as a fallback).

| File | Path | Format | Role |
|------|------|--------|------|
| `id_token.csv` | `INPUT/id_token.csv` | `ID, Token` | Primary token source |
| `mishkit_backup.csv` | `INPUT/mishkit_backup.csv` | `ID, Token` | Backup tokens (used as primary when available and different) |

## How It Works

1. **Load tokens** from CSV file (ID, Token columns)
2. **Check existing data** -- read last timestamp from house CSV file
3. **Generate time ranges** -- split the time gap into 7-day chunks
4. **Fetch each chunk** from the API with retry and exponential backoff
5. **Save after each chunk** -- merge with existing data, deduplicate, sort by timestamp
6. **Progress bar** -- tqdm shows per-house progress in weekly increments

The incremental approach means interrupted fetches resume from where they stopped, and each week's data is saved immediately to avoid losing progress on long jobs.

## CLI Arguments

### Batch mode (`python -m harvesting_data.cli`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--houses <path>` | `INPUT/id_token.csv` | CSV file with house tokens |
| `--output <path>` | `INPUT/UpdateHouseholdData` | Output directory |
| `--workers <n>` | 1 | Number of parallel workers |
| `--sequential` | false | Force sequential processing |
| `--filter <ids>` | all | Comma-separated house IDs to process |

### Single house (`python -m harvesting_data.fetch_single_house`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--house <id>` | (required) | House ID to fetch |
| `--token <token>` | from file | API token (overrides token file) |
| `--token-file <path>` | `INPUT/id_token.csv` | Token file path |
| `--output <path>` | `INPUT/UpdateHouseholdData` | Output directory |
| `--sequential` | false | Use sequential fetching |

## Output Format

Data saved to `INPUT/UpdateHouseholdData/{house_id}.csv`:

```csv
timestamp,1,2,3
2024-01-01 00:00:00,1234.5,2345.6,3456.7
```

- `timestamp`: YYYY-MM-DD HH:MM:SS
- `1`, `2`, `3`: Power per phase (Watts), 1-minute resolution
- `undef` values from the API are stored as NaN

## API Details

- **Endpoint**: `http://www.energyhive.com/mobile_proxy/getHV`
- **Aggregation**: per-minute sum
- **Response**: JSON with per-phase data arrays
- **Retry strategy**: HTTP-level retries (3x for 429/5xx) + application-level retries (5x with exponential backoff for timeouts, empty responses, API errors)
