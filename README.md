# ElectricPatterns - Household Energy Consumption Analysis

This repository contains the code and pipeline for analyzing electricity consumption patterns in households. The project focuses on event-based segmentation, device identification, and temporal consumption analysis.

## Project Objectives
- Identify distinct electricity usage patterns in household data
- Analyze device-specific energy consumption (water heaters, air conditioners, high-power appliances)
- Develop efficient segmentation and classification methods for power events
- Create visualizations to showcase consumption trends and anomalies

## Features
- **Event Detection**: Automatic detection of ON/OFF power events based on configurable thresholds
- **Event Matching**: Smart pairing of ON/OFF events with phase validation
- **Data Segmentation**: Role-based segregation of consumption into event-related and background power
- **Evaluation Tools**: Quality metrics for segmentation performance
- **Visualization**: Interactive plots showing consumption patterns across phases

## Repository Structure
```
.
├── experiment_pipeline/     # Main pipeline code
│   ├── INPUT/              # Input data directory (gitignored)
│   ├── OUTPUT/             # All outputs: results, logs, errors (gitignored)
│   ├── tests/              # Comprehensive test suite
│   ├── scripts/            # Helper scripts
│   ├── docs/               # Detailed documentation (gitignored, local only)
│   └── *.py               # Core modules (on_off_log, matcher, segmentation, etc.)
├── user_plot_requests/     # Interactive plotting tools
├── harvesting_data/        # Data collection utilities
└── requirements.txt        # Python dependencies
```

See [experiment_pipeline/README.md](experiment_pipeline/README.md) for detailed pipeline documentation.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- conda (recommended) or virtualenv
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hilakhadad/ElectricPatterns.git
   cd ElectricPatterns
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n electric_patterns python=3.9
   conda activate electric_patterns
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start
```bash
cd experiment_pipeline

# Run the test suite
python tests/run_all_tests.py

# Run on example data (requires data in INPUT/HouseholdData/)
python simple_test_example.py
```

For detailed usage instructions, see the [Pipeline README](experiment_pipeline/README.md).

## Pipeline Overview

The analysis pipeline consists of five main stages:

1. **On/Off Detection** - Identifies power events based on magnitude thresholds
2. **Event Matching** - Pairs ON/OFF events with temporal and phase validation
3. **Segmentation** - Segregates consumption into event-specific and background power
4. **Evaluation** - Calculates separation quality metrics
5. **Visualization** - Generates interactive plots

## Testing

The project includes comprehensive unit and integration tests:

```bash
# Run all tests
python experiment_pipeline/tests/run_all_tests.py

# Run specific test suites
python experiment_pipeline/tests/test_unit.py
python experiment_pipeline/tests/test_pipeline.py
```

## Configuration

All paths and parameters are configured in `experiment_pipeline/data_util.py`. The pipeline uses relative paths and works with local INPUT/OUTPUT directories by default.

## Contributing

When making changes:
1. Run tests before committing: `python tests/run_all_tests.py`
2. Ensure all tests pass before pushing
3. Use meaningful commit messages

## License

[Add license information]

## Contact

[Add contact information]
