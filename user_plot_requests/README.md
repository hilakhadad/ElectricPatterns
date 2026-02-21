# User Plot Requests

Interactive visualization tools for viewing household power data and segmentation results. Includes a Flask web app for local use and Colab notebooks for remote analysis.

## Quick Start

### Web App (Local)
```bash
python app.py
# Open http://localhost:5000
```

### Colab Notebooks
- `device_plots_colab.ipynb` - Plot device activations from static experiments
- `device_plots_dynamic_threshold.ipynb` - Plot device activations from dynamic threshold experiments (exp010/exp012)

## Structure

```
user_plot_requests/
├── app.py                                  # Flask server (main entry)
├── plot_by_request.py                      # CLI plot generation
├── plot_by_request_GUI.py                  # GUI plot generation
├── src/
│   ├── data_loader.py                      # Load data from experiments/INPUT
│   └── plot_generator.py                   # Generate Plotly plots
├── device_plots_colab.ipynb                # Colab notebook (static experiments)
└── device_plots_dynamic_threshold.ipynb    # Colab notebook (dynamic threshold)
```

## Web App Features

- Select house from dropdown
- Choose date and time window
- 4-row interactive plot: original power, remaining power, segregated devices, events
- Download plots as HTML

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

1. **Experiment output**: `experiment_pipeline/OUTPUT/experiments/` - includes segmentation results
2. **Raw input**: `INPUT/HouseholdData/` - original power data only (fallback)
