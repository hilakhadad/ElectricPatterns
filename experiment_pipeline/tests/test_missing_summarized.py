"""
Regression test for Bug #22: Missing summarized files cause NaN inflation.

When no events are detected for a month in iteration N, that month must still
get a summarized file (copied from iteration N-1).  Without this, the dynamic
report's inner merge drops all timestamps from the missing month, inflating
the "No Data" percentage.

The test creates a two-iteration layout where run_1 has NO events in any
month, then verifies:
  1. All months still receive a summarized file in run_1.
  2. The copied files contain no NaN in the power columns.
  3. An inner merge (same logic as the report) loses zero rows.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest import mock

# ── path setup ──────────────────────────────────────────────────────────
_src_dir = str(Path(__file__).resolve().parent.parent / 'src')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# ── constants ───────────────────────────────────────────────────────────
PHASES = ['w1', 'w2', 'w3']


# ── helpers ─────────────────────────────────────────────────────────────
def _make_summarized_df(month, year, minutes=60, base_power=500.0):
    """Create a synthetic summarized DataFrame for one month."""
    start = pd.Timestamp(f'{year}-{month:02d}-01')
    timestamps = pd.date_range(start=start, periods=minutes, freq='min')
    data = {'timestamp': timestamps}
    for phase in PHASES:
        power = np.full(minutes, base_power)
        data[f'original_{phase}'] = power
        data[f'remaining_{phase}'] = power          # passthrough
        data[f'short_duration_{phase}'] = np.zeros(minutes)
        data[f'medium_duration_{phase}'] = np.zeros(minutes)
        data[f'long_duration_{phase}'] = np.zeros(minutes)
    return pd.DataFrame(data)


# ── tests ───────────────────────────────────────────────────────────────
class TestMissingSummarizedCopy:
    """Bug #22: months with no events must still get summarized files."""

    HOUSE_ID = 'test'
    MONTHS = [1, 2, 3]
    YEAR = 1990

    @pytest.fixture(autouse=True)
    def setup_dirs(self, tmp_path):
        self.tmp = tmp_path

        # ── run_0: all 3 months have summarized files ──
        run0_summ = tmp_path / 'run_0' / f'house_{self.HOUSE_ID}' / 'summarized'
        run0_summ.mkdir(parents=True)
        for m in self.MONTHS:
            _make_summarized_df(m, self.YEAR).to_pickle(
                run0_summ / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            )

        # ── run_1: matches dir exists but is EMPTY (no events detected) ──
        run1_house = tmp_path / 'run_1' / f'house_{self.HOUSE_ID}'
        (run1_house / 'matches').mkdir(parents=True)

        # logging directory
        (tmp_path / 'logs').mkdir(exist_ok=True)

    # ── helper: run segmentation with patched core paths ──
    def _run_segmentation(self):
        import core
        with mock.patch.object(core, 'OUTPUT_BASE_PATH', str(self.tmp)), \
             mock.patch.object(core, 'RAW_INPUT_DIRECTORY', str(self.tmp / 'raw')), \
             mock.patch.object(core, 'LOGS_DIRECTORY', str(self.tmp / 'logs')), \
             mock.patch.object(core, 'ERRORS_DIRECTORY', str(self.tmp / 'errors')):
            from disaggregation.pipeline.segmentation_step import process_segmentation
            process_segmentation(house_id=self.HOUSE_ID, run_number=1)

    def _run1_summ_dir(self):
        return self.tmp / 'run_1' / f'house_{self.HOUSE_ID}' / 'summarized'

    # ── test 1: every month has a summarized file ──
    def test_all_months_have_summarized_after_empty_run(self):
        """Even when run_1 has zero events, every month gets a summarized file."""
        self._run_segmentation()
        for m in self.MONTHS:
            f = self._run1_summ_dir() / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            assert f.exists(), (
                f"summarized file missing for month {m:02d}/{self.YEAR} in run_1"
            )

    # ── test 2: no NaN in copied files ──
    def test_copied_months_have_no_nan_in_power_columns(self):
        """Copied summarized files contain no NaN in original/remaining columns."""
        self._run_segmentation()
        for m in self.MONTHS:
            df = pd.read_pickle(
                self._run1_summ_dir() / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            )
            for phase in PHASES:
                assert df[f'remaining_{phase}'].notna().all(), (
                    f"NaN in remaining_{phase} for month {m:02d}"
                )
                assert df[f'original_{phase}'].notna().all(), (
                    f"NaN in original_{phase} for month {m:02d}"
                )

    # ── test 3: report merge loses zero rows ──
    def test_report_merge_no_nan_inflation(self):
        """
        Simulate the dynamic report merge (baseline inner-join final).
        All baseline timestamps must survive — zero row loss.
        """
        self._run_segmentation()

        run0_summ = self.tmp / 'run_0' / f'house_{self.HOUSE_ID}' / 'summarized'
        run1_summ = self._run1_summ_dir()

        baseline = pd.concat(
            [pd.read_pickle(f) for f in sorted(run0_summ.glob('*.pkl'))],
            ignore_index=True,
        )
        final = pd.concat(
            [pd.read_pickle(f) for f in sorted(run1_summ.glob('*.pkl'))],
            ignore_index=True,
        )

        for phase in PHASES:
            merged = baseline[['timestamp', f'original_{phase}']].merge(
                final[['timestamp', f'remaining_{phase}']],
                on='timestamp',
                how='inner',
            )
            assert len(merged) == len(baseline), (
                f"Inner merge lost {len(baseline) - len(merged)} rows for {phase} "
                f"— missing summarized files cause NaN inflation in reports"
            )
            assert merged[f'original_{phase}'].notna().all()
            assert merged[f'remaining_{phase}'].notna().all()
