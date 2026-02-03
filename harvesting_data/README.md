# Harvesting Data

Data acquisition module for fetching household power consumption data from the EnergyHive API.

## Quick Start

```bash
# Parallel update (faster)
python -m harvesting_data.cli --parallel

# Sequential update (more stable)
python -m harvesting_data.cli --sequential

# Update specific house
python -m harvesting_data.cli --house 140
```

## Structure

```
harvesting_data/
├── cli.py                        # Command-line interface
├── api.py                        # EnergyHive API client
├── fetcher.py                    # Fetch logic & retry handling
├── storage.py                    # Data persistence
├── config.py                     # Configuration constants
└── batch_update_all_houses.py    # Legacy script (use CLI instead)
```

## Configuration

Edit `config.py`:

```python
DATA_DIR = "../INPUT/HouseholdData"  # Output directory
MAX_DAYS_PER_REQUEST = 30            # API limit
MAX_CONCURRENT_REQUESTS = 5          # Parallel limit
REQUEST_TIMEOUT = 60                 # Seconds
API_BASE_URL = "https://..."         # EnergyHive API
```

## House Tokens

Create `house_tokens.csv` with house IDs and API tokens:

```csv
house_id,token
140,abc123...
125,def456...
```

## Output

Data is saved to `INPUT/HouseholdData/`:

```
INPUT/HouseholdData/
├── 140/                          # Monthly folder structure
│   ├── 140_01_2024.csv
│   ├── 140_02_2024.csv
│   └── ...
├── 125/
│   └── ...
└── house_tokens.csv              # Token configuration
```

## CSV Format

```csv
timestamp,w1,w2,w3
01/01/2024 00:00,1234.5,2345.6,3456.7
01/01/2024 00:01,1235.0,2346.0,3457.0
...
```

- `timestamp`: DD/MM/YYYY HH:MM format
- `w1`, `w2`, `w3`: Power consumption per phase (Watts)

## Usage in Code

```python
from harvesting_data.fetcher import update_house
from harvesting_data.storage import load_house_tokens

# Load tokens
tokens = load_house_tokens("house_tokens.csv")

# Update single house
update_house(
    house_id="140",
    token=tokens["140"],
    data_dir="../INPUT/HouseholdData"
)
```

## Features

- **Incremental updates**: Only fetches new data since last update
- **Retry logic**: Exponential backoff on failures
- **Parallel fetching**: Multiple houses simultaneously
- **Time range splitting**: Handles large date ranges
