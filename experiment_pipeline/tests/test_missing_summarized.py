"""
Regression test for Bug #22: Missing summarized files cause NaN inflation.

When no events are detected for a month in iteration N, that month must still
get a summarized file — either via passthrough or copied from a previous run.
Without this, the dynamic report's inner merge drops all timestamps from the
missing month, inflating the "No Data" percentage.

Scenarios tested:
  A. run_1 has no events, run_0 has all months → copied from run_0.
  B. run_3 has no events, and run_1/run_2 also lack that month → copied
     backwards from run_0 (multi-hop search).
  C. matches dir doesn't exist at all → still creates summarized files.
  D. Report merge loses zero rows after the fix.
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


def _run_segmentation(tmp_path, house_id, run_number):
    """Run process_segmentation with patched core paths."""
    import core
    with mock.patch.object(core, 'OUTPUT_BASE_PATH', str(tmp_path)), \
         mock.patch.object(core, 'RAW_INPUT_DIRECTORY', str(tmp_path / 'raw')), \
         mock.patch.object(core, 'LOGS_DIRECTORY', str(tmp_path / 'logs')), \
         mock.patch.object(core, 'ERRORS_DIRECTORY', str(tmp_path / 'errors')):
        from disaggregation.pipeline.segmentation_step import process_segmentation
        process_segmentation(house_id=house_id, run_number=run_number)


# ── Scenario A: single empty run ────────────────────────────────────────
class TestMissingSummarizedCopy:
    """run_1 has zero events — all months must be copied from run_0."""

    HOUSE_ID = 'test'
    MONTHS = [1, 2, 3]
    YEAR = 1990

    @pytest.fixture(autouse=True)
    def setup_dirs(self, tmp_path):
        self.tmp = tmp_path

        # run_0: all 3 months have summarized files
        run0_summ = tmp_path / 'run_0' / f'house_{self.HOUSE_ID}' / 'summarized'
        run0_summ.mkdir(parents=True)
        for m in self.MONTHS:
            _make_summarized_df(m, self.YEAR).to_pickle(
                run0_summ / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            )

        # run_1: matches dir exists but is EMPTY (no events detected)
        run1_house = tmp_path / 'run_1' / f'house_{self.HOUSE_ID}'
        (run1_house / 'matches').mkdir(parents=True)

        (tmp_path / 'logs').mkdir(exist_ok=True)

    def _run1_summ_dir(self):
        return self.tmp / 'run_1' / f'house_{self.HOUSE_ID}' / 'summarized'

    def test_all_months_have_summarized_after_empty_run(self):
        _run_segmentation(self.tmp, self.HOUSE_ID, run_number=1)
        for m in self.MONTHS:
            f = self._run1_summ_dir() / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            assert f.exists(), f"summarized missing for month {m:02d}"

    def test_copied_months_have_no_nan_in_power_columns(self):
        _run_segmentation(self.tmp, self.HOUSE_ID, run_number=1)
        for m in self.MONTHS:
            df = pd.read_pickle(
                self._run1_summ_dir() / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            )
            for phase in PHASES:
                assert df[f'remaining_{phase}'].notna().all()
                assert df[f'original_{phase}'].notna().all()

    def test_report_merge_no_nan_inflation(self):
        _run_segmentation(self.tmp, self.HOUSE_ID, run_number=1)

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
                on='timestamp', how='inner',
            )
            assert len(merged) == len(baseline), (
                f"Merge lost {len(baseline) - len(merged)} rows for {phase}"
            )


# ── Scenario B: multi-hop backwards search ──────────────────────────────
class TestMultiHopBackwardsSearch:
    """
    run_0 has months 1-3.  run_1 and run_2 have only month 1.
    run_3 must find months 2-3 by searching back to run_0.
    """

    HOUSE_ID = 'hop'
    YEAR = 1990

    @pytest.fixture(autouse=True)
    def setup_dirs(self, tmp_path):
        self.tmp = tmp_path

        # run_0: all 3 months
        run0_summ = tmp_path / 'run_0' / f'house_{self.HOUSE_ID}' / 'summarized'
        run0_summ.mkdir(parents=True)
        for m in [1, 2, 3]:
            _make_summarized_df(m, self.YEAR).to_pickle(
                run0_summ / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            )

        # run_1: only month 1 (months 2-3 missing — simulates broken old code)
        run1_summ = tmp_path / 'run_1' / f'house_{self.HOUSE_ID}' / 'summarized'
        run1_summ.mkdir(parents=True)
        _make_summarized_df(1, self.YEAR).to_pickle(
            run1_summ / f'summarized_{self.HOUSE_ID}_01_{self.YEAR}.pkl'
        )

        # run_2: only month 1 (same gap)
        run2_summ = tmp_path / 'run_2' / f'house_{self.HOUSE_ID}' / 'summarized'
        run2_summ.mkdir(parents=True)
        _make_summarized_df(1, self.YEAR).to_pickle(
            run2_summ / f'summarized_{self.HOUSE_ID}_01_{self.YEAR}.pkl'
        )

        # run_3: matches dir empty (no events)
        run3_house = tmp_path / 'run_3' / f'house_{self.HOUSE_ID}'
        (run3_house / 'matches').mkdir(parents=True)

        (tmp_path / 'logs').mkdir(exist_ok=True)

    def test_missing_months_found_from_run_0(self):
        """Safeguard searches run_2 → run_1 → run_0 to find months 2-3."""
        _run_segmentation(self.tmp, self.HOUSE_ID, run_number=3)

        run3_summ = self.tmp / 'run_3' / f'house_{self.HOUSE_ID}' / 'summarized'
        for m in [1, 2, 3]:
            f = run3_summ / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            assert f.exists(), (
                f"month {m:02d} missing in run_3 — backwards search failed"
            )

    def test_backwards_copy_has_no_nan(self):
        _run_segmentation(self.tmp, self.HOUSE_ID, run_number=3)

        run3_summ = self.tmp / 'run_3' / f'house_{self.HOUSE_ID}' / 'summarized'
        for m in [2, 3]:
            df = pd.read_pickle(
                run3_summ / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            )
            for phase in PHASES:
                assert df[f'remaining_{phase}'].notna().all()
                assert df[f'original_{phase}'].notna().all()


# ── Scenario C: no matches directory at all ──────────────────────────────
class TestNoMatchesDirectory:
    """
    matches/ dir doesn't exist (e.g. detection found zero events).
    Segmentation must NOT return early — must still create summarized.
    """

    HOUSE_ID = 'nodir'
    YEAR = 1990

    @pytest.fixture(autouse=True)
    def setup_dirs(self, tmp_path):
        self.tmp = tmp_path

        # run_0: all 3 months
        run0_summ = tmp_path / 'run_0' / f'house_{self.HOUSE_ID}' / 'summarized'
        run0_summ.mkdir(parents=True)
        for m in [1, 2, 3]:
            _make_summarized_df(m, self.YEAR).to_pickle(
                run0_summ / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            )

        # run_1: house dir exists but NO matches/ subdirectory at all
        run1_house = tmp_path / 'run_1' / f'house_{self.HOUSE_ID}'
        run1_house.mkdir(parents=True)
        # intentionally NOT creating matches/ dir

        (tmp_path / 'logs').mkdir(exist_ok=True)

    def test_summarized_created_despite_no_matches_dir(self):
        _run_segmentation(self.tmp, self.HOUSE_ID, run_number=1)

        run1_summ = self.tmp / 'run_1' / f'house_{self.HOUSE_ID}' / 'summarized'
        for m in [1, 2, 3]:
            f = run1_summ / f'summarized_{self.HOUSE_ID}_{m:02d}_{self.YEAR}.pkl'
            assert f.exists(), (
                f"month {m:02d} missing — segmentation returned early "
                f"because matches/ dir doesn't exist"
            )
