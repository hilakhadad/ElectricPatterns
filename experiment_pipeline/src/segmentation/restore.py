"""
Restore skipped matches back to unmatched files.

Bug #14: When segmentation skips a match (extracting it would create negative
remaining power), the match was deleted from matches but NOT returned to
unmatched_on/unmatched_off, causing events to disappear.
"""
import os
import pandas as pd
from pathlib import Path


def restore_skipped_to_unmatched(skipped_matches, output_dir, house_id, month, year, logger):
    """Restore skipped matches back to unmatched_on and unmatched_off files.

    Args:
        skipped_matches: DataFrame of match rows that were skipped by segmentation
        output_dir: Base output directory for this house/run
        house_id: House identifier
        month: Month number
        year: Year number
        logger: Logger instance
    """
    restored_on = []
    restored_off = []

    for _, row in skipped_matches.iterrows():
        correction = row.get('correction', 0) or 0

        # Restore original magnitudes (undo correction applied in _format_match)
        if 'original_on_magnitude' in row and pd.notna(row.get('original_on_magnitude')):
            on_mag = row['original_on_magnitude']
            off_mag = -row['original_off_magnitude']
        else:
            on_mag = row['on_magnitude'] + correction if correction > 0 else row['on_magnitude']
            off_mag = row['off_magnitude'] - correction if correction > 0 else row['off_magnitude']

        restored_on.append({
            'event_id': row['on_event_id'],
            'start': row['on_start'], 'end': row['on_end'],
            'magnitude': on_mag,
            'duration': (row['on_end'] - row['on_start']).total_seconds() / 60,
            'phase': row['phase'], 'event': 'on'
        })
        restored_off.append({
            'event_id': row['off_event_id'],
            'start': row['off_start'], 'end': row['off_end'],
            'magnitude': off_mag,
            'duration': (row['off_end'] - row['off_start']).total_seconds() / 60,
            'phase': row['phase'], 'event': 'off'
        })

    for events_list, subdir, prefix in [
        (restored_on, 'unmatched_on', 'unmatched_on'),
        (restored_off, 'unmatched_off', 'unmatched_off'),
    ]:
        dir_path = Path(output_dir) / subdir
        os.makedirs(dir_path, exist_ok=True)
        file_path = dir_path / f"{prefix}_{house_id}_{month:02d}_{year}.pkl"
        new_df = pd.DataFrame(events_list)
        if file_path.exists():
            existing = pd.read_pickle(file_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_pickle(file_path)

    logger.info(f"  Restored {len(restored_on)} ON + {len(restored_off)} OFF events to unmatched files")
