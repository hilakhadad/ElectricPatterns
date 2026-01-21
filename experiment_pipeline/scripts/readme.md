# Scripts Directory

Execution scripts for running the pipeline.

## Available Scripts

### test_single_house.py

Run the complete pipeline on a single house.

**Configure (in script):**
```python
HOUSE_ID = "125"                           # House to process
EXPERIMENT_NAME = "exp005_asymmetric_windows"  # Experiment config
MAX_ITERATIONS = 2                          # Number of iterations
```

**Run:**
```bash
python scripts/test_single_house.py
```

**Import and use:**
```python
from test_single_house import run_pipeline_for_house

result = run_pipeline_for_house(
    house_id="125",
    experiment_name="exp005_asymmetric_windows",
    output_path="/path/to/output",
    max_iterations=2
)
# Returns: {'success': bool, 'iterations': int, 'error': str or None}
```

### test_array_of_houses.py

Run the pipeline on multiple houses in parallel.

**Features:**
- Auto-detects houses from `INPUT/HouseholdData/`
- Parallel processing with configurable workers
- Creates timestamped experiment output directory
- Progress tracking and error handling

**Run:**
```bash
python scripts/test_array_of_houses.py
```

## Related

- Source modules: `src/pipeline/` (orchestration), `src/core/` (config)
- Test suite: `tests/`
- Documentation: `docs/`
