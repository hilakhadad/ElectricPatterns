# Scripts Directory

Helper and utility scripts for running the pipeline.

---

## 📄 Available Scripts

### test_single_house.py
Run the complete pipeline on a single house with customizable parameters.

**Configure:**
```python
LOCAL_INPUT_PATH = r"C:\path\to\input"
LOCAL_OUTPUT_PATH = r"C:\path\to\output"
HOUSE_ID = "1"
DEFAULT_THRESHOLD = 1600
RUN_NUMBER = 0
```

**Run:**
```powershell
python scripts/test_single_house.py
```

---

### run_scripts.py
Orchestrates the complete pipeline with automatic threshold adjustment.

**Features:**
- Input/output validation
- Automatic threshold adjustment based on evaluation
- Multi-house batch processing capability

**Run:**
```powershell
python scripts/run_scripts.py
```

---

## 🚀 Quick Examples

### Run on a single house:
```powershell
# Edit test_single_house.py to set your paths and house ID
python scripts/test_single_house.py
```

### Run full pipeline orchestration:
```powershell
python scripts/run_scripts.py
```

---

## 📝 Notes

- These scripts are higher-level wrappers around the core modules
- For testing and development, use the test suite in `tests/`
- For production runs, use these scripts

---

## 🔗 Related

- Core modules: `on_off_log.py`, `new_matcher.py`, `segmentation.py`
- Test suite: `tests/`
- Documentation: `docs/`
