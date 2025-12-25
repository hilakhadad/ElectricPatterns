# Development Guide

This guide helps you set up and work with the ElectricPatterns repository.

## Quick Start

1. **Clone and setup:**

```bash
git clone https://github.com/hilakhadad/ElectricPatterns.git
cd ElectricPatterns
conda create -n electric_patterns python=3.9
conda activate electric_patterns
pip install -r requirements.txt
```

2. **Verify installation:**

```bash
cd experiment_pipeline
python tests/run_all_tests.py
```

3. **Run example analysis:**

```bash
# Make sure you have data in experiment_pipeline/INPUT/HouseholdData/
python simple_test_example.py
```

## Development Workflow

### Working with Git

The repository uses a single `main` branch. For development:

```bash
# Check current status
git status

# Create a feature branch for your work
git checkout -b feat/my-feature

# Make changes, then commit
git add .
git commit -m "Description of changes"

# Push to remote
git push origin feat/my-feature
```

### Before Making Changes

**Always run tests first** to ensure everything works:

```bash
cd experiment_pipeline
python tests/run_all_tests.py
```

### After Making Changes

**Run tests again** to verify nothing broke:

```bash
python tests/run_all_tests.py
```

If tests pass, you're safe to commit!

## Project Structure

```
ElectricPatterns/
├── experiment_pipeline/        # Main analysis pipeline
│   ├── INPUT/                 # Your data goes here (gitignored)
│   │   └── HouseholdData/    # CSV files with household data
│   ├── OUTPUT/                # Results, logs, errors (gitignored)
│   ├── tests/                 # Test suite
│   ├── scripts/               # Helper scripts
│   ├── docs/                  # Documentation (gitignored, local only)
│   └── *.py                   # Core pipeline modules
├── user_plot_requests/         # Interactive plotting tools
├── harvesting_data/            # Data collection utilities
└── requirements.txt
```

## Running the Pipeline

### Option 1: Quick test with example data

```bash
cd experiment_pipeline
python simple_test_example.py
```

### Option 2: Run specific stages

```python
from on_off_log import process_house
from new_matcher import process_matches
from segmentation import process_segmentation

house_id = "example"
run_number = 0
threshold = 1600

# Run each stage
process_house(house_id, run_number, threshold)
process_matches(house_id, run_number, threshold)
process_segmentation(house_id, run_number)
```

### Option 3: Batch processing

```bash
cd experiment_pipeline/scripts
python run_scripts.py
```

## Testing

The project includes comprehensive tests:

```bash
# Run all tests
python tests/run_all_tests.py

# Run only unit tests
python tests/test_unit.py

# Run only integration tests
python tests/test_pipeline.py
```

## Configuration

Edit `experiment_pipeline/data_util.py` to change:
- Input/output paths
- Default thresholds
- Improvement criteria
- Logging settings

## Best Practices

1. **Always activate the environment:**
   ```bash
   conda activate electric_patterns
   ```

2. **Test before committing:**
   ```bash
   python tests/run_all_tests.py
   git add .
   git commit -m "Your message"
   ```

3. **Use meaningful commit messages:**
   - ✅ "Add support for multi-phase event detection"
   - ❌ "fix stuff"

4. **Keep commits small and focused:**
   - One logical change per commit
   - Makes debugging and review easier

## Troubleshooting

### "ModuleNotFoundError"
Make sure you're in the correct environment:
```bash
conda activate electric_patterns
pip install -r requirements.txt
```

### Tests failing
Check the test output for details. Common issues:
- Missing data files in INPUT/
- Incorrect paths in data_util.py
- Missing dependencies

### Git issues
If you need to reset your local changes:
```bash
git status  # Check what changed
git restore <file>  # Restore specific file
git restore .  # Restore all files
```

## Adding New Features

1. Create a feature branch
2. Add your code
3. Add tests for your code
4. Run all tests
5. Commit and push
6. Create a pull request (if using GitHub flow)

## Documentation

- Main README: Project overview
- Pipeline README: Detailed pipeline documentation
- Testing Guide: In docs/ (local only)
- This file: Development workflow

## Need Help?

- Check the [main README](../README.md)
- Check the [pipeline README](experiment_pipeline/README.md)
- Review test files for usage examples
- Check function docstrings in the code