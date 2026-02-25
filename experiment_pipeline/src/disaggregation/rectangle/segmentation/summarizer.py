"""
Summarization of segmented data.

Groups event power by duration category (short/medium/long).
"""
import numpy as np
import pandas as pd
from typing import List


def summarize_segmentation(
    data: pd.DataFrame,
    phases: List[str] = ['w1', 'w2', 'w3'],
    logger=None
) -> pd.DataFrame:
    """
    Summarize segmented data into duration categories.

    Duration categories:
    - short: <= 2 minutes
    - medium: 3-24 minutes
    - long: >= 25 minutes

    Args:
        data: DataFrame with event_power columns
        phases: List of phase names

    Returns:
        DataFrame with summarized columns
    """
    event_columns = [col for col in data.columns if 'event_power' in col]
    summarized = {'timestamp': data['timestamp']}

    for phase in phases:
        summarized[f'original_{phase}'] = data[phase].copy()
        summarized[f'remaining_{phase}'] = data[f'remaining_power_{phase}']

        short_sum = np.zeros(len(data))
        medium_sum = np.zeros(len(data))
        long_sum = np.zeros(len(data))

        phase_columns = [col for col in event_columns if col.endswith(f'_m_{phase}')]

        for col in phase_columns:
            duration = int(col.split('_')[2])
            if duration <= 2:
                short_sum += data[col]
            elif duration <= 24:
                medium_sum += data[col]
            else:
                long_sum += data[col]

        summarized[f'short_duration_{phase}'] = short_sum
        summarized[f'medium_duration_{phase}'] = medium_sum
        summarized[f'long_duration_{phase}'] = long_sum

    if logger:
        logger.debug(f"Summarized {len(event_columns)} event columns into duration categories")

    return pd.DataFrame(summarized)
