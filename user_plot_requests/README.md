# User Plot Requests

Flask web application for interactive visualization of household power data and segmentation results.

## Quick Start

```bash
# Run the web server
python app.py

# Open in browser
# http://localhost:5000
```

## Structure

```
user_plot_requests/
├── app.py                        # Flask server (main entry)
├── src/
│   ├── data_loader.py            # Load data from experiments/INPUT
│   └── plot_generator.py         # Generate Plotly plots
├── templates/                    # HTML templates
├── static/                       # CSS, JS assets
└── device_plots_colab.ipynb      # Colab notebook for device plots
```

## Features

### Web Interface
- Select house from dropdown
- Choose date and time window
- View interactive power consumption plots
- Download plots as HTML files

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main page with house selection |
| `/plot/view` | GET | Display interactive plot |
| `/plot` | POST | Generate plot (JSON response) |
| `/plot/download` | GET | Download plot as HTML |
| `/api/houses` | GET | List available houses |
| `/api/house/<id>/range` | GET | Date range for house |

## Data Sources

The app loads data from two sources (in priority order):

1. **Experiment Output**: `experiment_pipeline/OUTPUT/experiments/`
   - Includes segmentation results (original, remaining, segmented)
   - Shows matched events

2. **Raw Input**: `INPUT/HouseholdData/`
   - Original power data only
   - Used if no experiment results available

## Colab Notebook

`device_plots_colab.ipynb` - Generate plots for specific device activations:

1. Copy dates from experiment report ("Show Copyable Dates" button)
2. Paste into notebook
3. Run cells to generate 12-hour window plots

Supported date formats:
- `DD/MM/YYYY HH:MM-HH:MM` (e.g., `10/01/2024 08:30-14:15`)
- `YYYY-MM-DD HH:MM-HH:MM` (e.g., `2022-11-11 08:41-09:28`)

## Plot Structure

4-row layout matching experiment_pipeline:
1. **Original Data** - Raw power consumption
2. **After Segregation** - Remaining power
3. **Segregation Data** - Short/Medium/Long duration events
4. **Event Markers** - ON/OFF event indicators

## Usage in Code

```python
from src.data_loader import load_house_data, get_houses_info
from src.plot_generator import generate_plot

# Get available houses
houses = get_houses_info()

# Load data
df = load_house_data(house_id="140", run_number=0)

# Generate plot
fig = generate_plot(df, center_time, window_type='day')
fig.write_html("output.html")
```

## Configuration

Edit in `app.py`:
- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `5000`)
- `DEBUG`: Debug mode (default: `True`)
