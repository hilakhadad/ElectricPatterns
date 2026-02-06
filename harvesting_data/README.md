# Harvesting Data

Fetches household power consumption data from the EnergyHive API.

## Quick Start

```bash
# Update single house
python -m harvesting_data.fetch_single_house --house 140

# Update all houses (parallel)
python -m harvesting_data.cli --parallel

# On SLURM cluster
sbatch slurm_fetch_house.sh 140
```

## Directory Structure

```
harvesting_data/
├── README.md
├── __init__.py          # Package exports
├── config.py            # Configuration (paths, API settings)
├── api.py               # EnergyHive API client
├── fetcher.py           # Fetch logic with retry handling
├── storage.py           # Data loading/saving
├── cli.py               # Command-line interface
├── fetch_single_house.py # Single house fetcher
├── slurm_fetch_house.sh  # SLURM job script
└── status.sh             # Check job status
```

## Usage

### Command Line

```bash
# Fetch single house
python -m harvesting_data.fetch_single_house --house 140

# Fetch all houses
python -m harvesting_data.cli --parallel    # Faster
python -m harvesting_data.cli --sequential  # More stable
python -m harvesting_data.cli --house 140   # Specific house
```

### SLURM Cluster

```bash
# Submit single house job
sbatch slurm_fetch_house.sh 140

# Check status
bash status.sh
```

### In Code

```python
from harvesting_data import update_house, load_backup_tokens

# Load tokens (backup tokens used as primary)
tokens = load_backup_tokens()

# Update a house
update_house("140", tokens["140"])
```

## Token Files

| File | Description |
|------|-------------|
| `mishkit_backup.csv` | **Primary** token source (ID, Token columns) |
| `mishkit.csv` | Fallback token source |

The system uses `mishkit_backup.csv` as the primary token source when available.

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

## Output Format

Data saved to `INPUT/UpdatatedHouseData/{house_id}.csv`:

```csv
timestamp,1,2,3
2024-01-01 00:00:00,1234.5,2345.6,3456.7
2024-01-01 00:01:00,1235.0,2346.0,3457.0
```

- `timestamp`: YYYY-MM-DD HH:MM:SS format
- `1`, `2`, `3`: Power per phase (Watts)

## Features

- **Incremental updates**: Only fetches data since last timestamp
- **Retry logic**: Exponential backoff on API failures
- **Parallel fetching**: Multiple houses simultaneously
- **Empty response handling**: Detects legitimately empty periods vs errors
- **Backup tokens**: Uses backup token file as primary source
