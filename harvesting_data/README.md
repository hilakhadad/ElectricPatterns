# Harvesting Data

Fetches household power consumption data from the EnergyHive API. Supports incremental updates (only fetches data since last timestamp), parallel fetching, and SLURM cluster execution.

## Quick Start

```bash
# Single house
python -m harvesting_data.fetch_single_house --house 140

# All houses (parallel)
python -m harvesting_data.cli --parallel

# All houses (sequential, more stable)
python -m harvesting_data.cli --sequential

# SLURM cluster
sbatch slurm_fetch_house.sh 140
```

## Structure

```
harvesting_data/
├── api.py                # EnergyHive API client
├── fetcher.py            # Fetch logic with retry & exponential backoff
├── storage.py            # Data loading/saving
├── config.py             # Configuration (paths, API settings)
├── cli.py                # Command-line interface (multi-house)
├── fetch_single_house.py # Single house fetcher
├── slurm_fetch_house.sh  # SLURM job script
└── status.sh             # Check SLURM job status
```

## Configuration

Edit `config.py`:

```python
DATA_DIR = Path("../INPUT/UpdatatedHouseData")
TOKEN_FILE = Path("../INPUT/mishkit.csv")
BACKUP_TOKEN_FILE = Path("../INPUT/mishkit_backup.csv")

MAX_DAYS_PER_REQUEST = 30      # API limit per request
MAX_CONCURRENT_REQUESTS = 5    # Parallel fetch limit
REQUEST_TIMEOUT = 60           # Seconds
MAX_RETRIES = 5                # Retry count on failure
```

## Token Files

| File | Description |
|------|-------------|
| `mishkit_backup.csv` | **Primary** token source (ID, Token columns) |
| `mishkit.csv` | Fallback token source |

## Output Format

Data saved to `INPUT/UpdatatedHouseData/{house_id}.csv`:

```csv
timestamp,1,2,3
2024-01-01 00:00:00,1234.5,2345.6,3456.7
```

- `timestamp`: YYYY-MM-DD HH:MM:SS
- `1`, `2`, `3`: Power per phase (Watts), 1-minute resolution
