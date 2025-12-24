# Role-Based Segregation Project

## Project Overview
This project implements a household energy segregation and analysis pipeline. It processes household power consumption data, detects on/off events, segments consumption patterns, and generates visualizations.

## Directory Structure
- **experiment_pipeline/**: Main pipeline for data processing and analysis
  - `run_scripts.py`: Main orchestrator script that runs the complete pipeline
  - `on_off_log.py`: Detects on/off events in power consumption data
  - `new_matcher.py`: Matches and correlates detected events
  - `segmentation.py`: Segments power consumption data by detected events
  - `eval_segmentation.py`: Evaluates segmentation quality
  - `visualization_with_mark.py`: Creates interactive visualizations with event markers
  - `data_util.py`: Utility functions and configuration constants

- **harvesting_data/**: Scripts for data collection and management
  - `fetch_and_store_house_data.py`: Retrieves and stores household data
  - `batch_update_all_houses.py`: Batch updates data for multiple houses

- **user_plot_requests/**: GUI and plotting interfaces
  - `plot_by_request.py`: Plot generation based on user requests
  - `plot_by_request_GUI.py`: GUI for interactive plot requests

## Dependencies
- **pandas**: Data manipulation and CSV handling
- **numpy**: Numerical computing and array operations
- **matplotlib**: Static plotting and visualization
- **plotly**: Interactive visualizations
- **tqdm**: Progress bar utilities
- **requests**: HTTP library for API calls
- **tkcalendar**: Calendar widget for GUI

## Configuration
The main configuration is located in `experiment_pipeline/data_util.py`:
- `RAW_INPUT_DIRECTORY`: Source data directory
- `BASE`: Base output directory
- `OUTPUT_BASE_PATH`: Output directory for processed data
- `LOGS_DIRECTORY`: Logging directory
- `DEFAULT_THRESHOLD`: Default threshold for power segmentation (1600 W)
- `THRESHOLD_STEP`: Increment step for threshold adjustments (200 W)

## Pipeline Overview
1. **on_off_log.py**: Detects on/off events in raw power consumption data
2. **new_matcher.py**: Matches correlated on/off events
3. **segmentation.py**: Segments data based on matched events using threshold adjustments
4. **eval_segmentation.py**: Evaluates segmentation performance
5. **visualization_with_mark.py**: Creates visualizations of segmented data with event markers

## Logging
Logs are stored in the `logs/` directory with filenames based on house ID.
Error logs are stored in the `errors/` directory.

## Running the Project
```bash
cd experiment_pipeline
python run_scripts.py
```

## Setup Instructions

### Create and Activate Environment
```bash
# Using conda
conda create --name role_seg python=3.9
conda activate role_seg
pip install -r requirements.txt

# Or using venv
python -m venv venv
source venv/Scripts/activate  # On Windows
pip install -r requirements.txt
```

## Data Format
Input data should be CSV files with the following columns:
- `timestamp`: DateTime of the measurement
- `1`, `2`, `3`: Power consumption for phases W1, W2, W3
- `sum`: Total power consumption

## Output Files
The pipeline generates:
- `on_off_1600.csv`: Detected on/off events with 1600W threshold
- `matches_[house_id].csv`: Matched event pairs
- `segmented_[house_id].csv`: Segmented consumption data
- `summarized_[house_id].csv`: Summarized segregation results
- `separation_evaluation_[house_id].csv`: Evaluation metrics
- `plots/`: Directory containing visualization plots

## Notes
- The pipeline uses adaptive threshold adjustments based on evaluation metrics
- Paths in `data_util.py` may need to be updated for your local environment
- Currently configured for HPC cluster paths - update accordingly for local development
