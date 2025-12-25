# Role-Based Energy Segregation Pipeline

Pipeline for segregating household energy consumption into role-based components using event detection and matching.

---

## 🚀 Quick Start

**New here? Start with:** [Getting Started Guide](docs/getting-started.md)

```powershell
# 1. Activate environment
conda activate role_seg_env

# 2. Run tests
cd experiment_pipeline
python tests/run_all_tests.py

# 3. Run on example data
python simple_test_example.py
```

---

## 📁 Project Structure

```
experiment_pipeline/
│
├── INPUT/                    # Input data
│   └── HouseholdData/       # 166 household CSV files
│       ├── example.csv
│       ├── 1.csv
│       └── ...
│
├── OUTPUT/                   # All outputs (results, logs, errors)
│   ├── run_0/               # First run results
│   ├── run_1/               # Second run results (if iterating)
│   ├── logs/                # Execution logs
│   └── errors/              # Error files
│
├── tests/                    # Test suite
│   ├── test_unit.py         # Unit tests
│   ├── test_pipeline.py     # Integration tests
│   └── run_all_tests.py     # Run all tests
│
├── docs/                     # Documentation
│   ├── getting-started.md   # Start here!
│   ├── quick-start.md       # Quick reference
│   ├── testing-guide.md     # Testing documentation
│   ├── refactoring-workflow.md  # How to refactor safely
│   └── output-structure.md  # Output directory structure
│
├── Core modules:
│   ├── data_util.py         # Configuration & paths
│   ├── on_off_log.py        # Event detection
│   ├── new_matcher.py       # Event matching
│   ├── segmentation.py      # Data segmentation
│   ├── eval_segmentation.py # Evaluation
│   └── visualization_with_mark.py  # Visualization
│
└── simple_test_example.py   # Simple test script
```

---

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| **[Getting Started](docs/getting-started.md)** | **Start here tomorrow!** |
| [Quick Start](docs/quick-start.md) | Quick reference guide |
| [Testing Guide](docs/testing-guide.md) | How to use the test suite |
| [Refactoring Workflow](docs/refactoring-workflow.md) | Safe refactoring process |
| [Output Structure](docs/output-structure.md) | OUTPUT directory explained |
| [Summary](docs/summary.md) | Complete project summary |

---

## 🧪 Testing

The project includes a comprehensive test suite to ensure safe refactoring:

```powershell
# Run all tests
python tests/run_all_tests.py

# Run specific test suites
python tests/test_unit.py         # Unit tests only
python tests/test_pipeline.py     # Integration tests only
```

**Test coverage:**
- ✅ 3 unit tests (functions, paths, configuration)
- ✅ 4 integration tests (full pipeline validation)
- ✅ All tests passing!

---

## 🔄 Pipeline Stages

1. **On/Off Detection** (`on_off_log.py`)
   - Detects power ON/OFF events
   - Output: `on_off_{threshold}.csv`

2. **Event Matching** (`new_matcher.py`)
   - Matches ON/OFF event pairs
   - Output: `matches_{house_id}.csv`

3. **Segmentation** (`segmentation.py`)
   - Segregates consumption by events
   - Output: `segmented_{house_id}.csv`

4. **Evaluation** (`eval_segmentation.py`)
   - Evaluates segregation quality
   - Output: `separation_evaluation_{house_id}.csv`

5. **Visualization** (`visualization_with_mark.py`)
   - Creates visual plots
   - Output: `plots/`

---

## 💻 Usage

### Run on example data:
```powershell
python simple_test_example.py
```

### Run on specific house:
```python
# Edit simple_test_example.py:
HOUSE_ID = "1"  # Change to desired house ID
```

### Run full pipeline:
```python
from on_off_log import process_house
from new_matcher import process_matches
from segmentation import process_segmentation

house_id = "example"
run_number = 0
threshold = 1600

process_house(house_id, run_number, threshold)
process_matches(house_id, run_number, threshold)
process_segmentation(house_id, run_number)
```

---

## ⚙️ Configuration

All paths are configured in `data_util.py`:

```python
RAW_INPUT_DIRECTORY  # INPUT/HouseholdData/
OUTPUT_BASE_PATH     # OUTPUT/
LOGS_DIRECTORY       # OUTPUT/logs/
ERRORS_DIRECTORY     # OUTPUT/errors/
```

---

## 🔧 Development

### Before refactoring:
```powershell
python tests/run_all_tests.py  # Ensure everything works
```

### After refactoring:
```powershell
python tests/run_all_tests.py  # Verify nothing broke
```

See [Refactoring Workflow](docs/refactoring-workflow.md) for detailed process.

---

## 📊 Requirements

```
Python >= 3.8
pandas >= 1.3.0
numpy >= 1.20.0
matplotlib >= 3.4.0
plotly >= 5.0.0
tqdm >= 4.62.0
```

Install: `pip install -r requirements.txt`

---

## 🐛 Troubleshooting

### "No module named 'X'"
```powershell
# Make sure you're in the correct environment:
conda activate role_seg_env
```

### Tests failing
```powershell
# Check the test output for details
python tests/run_all_tests.py

# Read the testing guide:
# docs/testing-guide.md
```

### Where are my outputs?
Everything is in `OUTPUT/`:
- Results: `OUTPUT/run_{N}/house_{ID}/`
- Logs: `OUTPUT/logs/`
- Errors: `OUTPUT/errors/`

---

## 📝 License

[Add your license here]

---

## 👥 Contributors

[Add contributors here]

---

**Need help?** Start with [Getting Started Guide](docs/getting-started.md)
