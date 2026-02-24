"""
Wave recovery I/O helpers.

Handles loading rectangle matches, loading/saving remaining power,
and saving wave match records.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

PHASES = ['w1', 'w2', 'w3']


def _load_rectangle_matches(
    output_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load all rectangle matches (not wave) from M1 run directories."""
    all_dfs = []

    for run_number, threshold in enumerate(threshold_schedule):
        # Try both naming conventions
        for pattern in [f"run_{run_number}", f"run_{run_number}_{threshold}w"]:
            run_dir = output_dir / pattern
            if run_dir.is_dir():
                break
        else:
            continue

        matches_dir = run_dir / f"house_{house_id}" / "matches"
        if not matches_dir.exists():
            continue

        for pkl_file in sorted(matches_dir.glob(f"matches_{house_id}_*.pkl")):
            try:
                df = pd.read_pickle(pkl_file)
                if df.empty:
                    continue
                df['iteration'] = run_number
                df['threshold'] = threshold
                all_dfs.append(df)
            except Exception as exc:
                logger.warning(f"Failed to load {pkl_file}: {exc}")

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def _load_remaining(
    output_dir: Path,
    house_id: str,
    last_run: int,
    logger: logging.Logger,
) -> tuple:
    """Load remaining power from last M1 run's summarized files.

    Returns
    -------
    (remaining_by_phase, monthly_data) : tuple
        remaining_by_phase: {phase: pd.Series indexed by timestamp}
        monthly_data: list of (month_tag, DataFrame) for saving later
    """
    # Find the last run directory
    summarized_dir = None
    for run_idx in range(last_run, -1, -1):
        candidate = output_dir / f"run_{run_idx}" / f"house_{house_id}" / "summarized"
        if candidate.is_dir():
            summarized_dir = candidate
            break

    if summarized_dir is None:
        return {}, []

    pkl_files = sorted(summarized_dir.glob(f"summarized_{house_id}_*.pkl"))
    if not pkl_files:
        return {}, []

    all_dfs = []
    monthly_data = []
    for f in pkl_files:
        df = pd.read_pickle(f)
        # Extract month tag from filename: summarized_{house_id}_{MM}_{YYYY}.pkl
        month_tag = f.stem.replace(f"summarized_{house_id}_", "")
        monthly_data.append((month_tag, df))
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Ensure timestamp is datetime
    if 'timestamp' in combined.columns:
        combined['timestamp'] = pd.to_datetime(combined['timestamp'])
        combined = combined.sort_values('timestamp').drop_duplicates('timestamp', keep='first')
        combined = combined.set_index('timestamp')

    remaining_by_phase = {}
    for phase in PHASES:
        col = f'remaining_{phase}'
        if col in combined.columns:
            remaining_by_phase[phase] = combined[col].dropna()

    logger.info(f"Wave recovery: loaded remaining from {summarized_dir} "
                f"({len(combined)} rows, phases: {list(remaining_by_phase.keys())})")

    return remaining_by_phase, monthly_data


def _save_wave_matches(
    output_dir: Path,
    house_id: str,
    match_records: List[dict],
    threshold_schedule: List[int],
    logger: logging.Logger,
):
    """Save wave match records grouped by month to run_post/house_{id}/matches/."""
    matches_df = pd.DataFrame(match_records)

    # Add iteration/threshold columns for M2 compatibility
    # Wave matches get iteration = len(threshold_schedule) (after all M1 iterations)
    matches_df['iteration'] = len(threshold_schedule)
    matches_df['threshold'] = 0

    # Group by month for per-month pickle files
    matches_df['_month'] = matches_df['on_start'].dt.strftime('%m_%Y')

    post_dir = output_dir / "run_post" / f"house_{house_id}" / "matches"
    post_dir.mkdir(parents=True, exist_ok=True)

    for month_tag, month_df in matches_df.groupby('_month'):
        month_df = month_df.drop(columns=['_month'])
        out_path = post_dir / f"matches_{house_id}_{month_tag}.pkl"
        month_df.to_pickle(out_path)
        logger.info(f"Wave recovery: saved {len(month_df)} matches to {out_path}")


def _save_updated_remaining(
    output_dir: Path,
    house_id: str,
    monthly_data: list,
    updated_remaining: Dict[str, pd.Series],
    logger: logging.Logger,
):
    """Save updated remaining (after wave extraction) to run_post/house_{id}/summarized/."""
    post_dir = output_dir / "run_post" / f"house_{house_id}" / "summarized"
    post_dir.mkdir(parents=True, exist_ok=True)

    for month_tag, original_df in monthly_data:
        df = original_df.copy()

        # Ensure timestamp for joining
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Update remaining columns from the extracted results
        for phase in PHASES:
            col = f'remaining_{phase}'
            if col in df.columns and phase in updated_remaining:
                updated = updated_remaining[phase]
                # Map updated values back by timestamp
                if 'timestamp' in df.columns:
                    ts_to_val = updated.to_dict()
                    df[col] = df['timestamp'].map(ts_to_val).fillna(df[col])

        out_path = post_dir / f"summarized_{house_id}_{month_tag}.pkl"
        df.to_pickle(out_path)

    logger.info(f"Wave recovery: saved updated remaining for {len(monthly_data)} months to {post_dir}")


def _remove_repaired_matches(
    output_dir: Path,
    house_id: str,
    threshold_schedule: List[int],
    repaired_ids: List[tuple],
    logger: logging.Logger,
):
    """
    Remove original rectangle matches that were replaced by wave repairs.

    For each (run_number, on_event_id), finds the match file and removes that row.
    """
    # Group by run_number for efficiency
    by_run = {}
    for run_number, on_event_id in repaired_ids:
        by_run.setdefault(run_number, set()).add(on_event_id)

    for run_number, event_ids in by_run.items():
        threshold = threshold_schedule[run_number] if run_number < len(threshold_schedule) else 0

        # Find run directory
        run_dir = None
        for pattern in [f"run_{run_number}", f"run_{run_number}_{threshold}w"]:
            candidate = output_dir / pattern
            if candidate.is_dir():
                run_dir = candidate
                break

        if run_dir is None:
            continue

        matches_dir = run_dir / f"house_{house_id}" / "matches"
        if not matches_dir.exists():
            continue

        for pkl_file in sorted(matches_dir.glob(f"matches_{house_id}_*.pkl")):
            try:
                df = pd.read_pickle(pkl_file)
                if df.empty:
                    continue

                original_len = len(df)
                mask = df['on_event_id'].isin(event_ids)
                if mask.any():
                    removed = mask.sum()
                    df = df[~mask]
                    df.to_pickle(pkl_file)
                    logger.info(
                        f"  Hole repair: removed {removed} rectangle match(es) from {pkl_file.name}"
                    )
            except Exception as exc:
                logger.warning(f"Failed to update {pkl_file}: {exc}")
