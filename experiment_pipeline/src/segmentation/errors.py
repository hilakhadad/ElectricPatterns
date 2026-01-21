"""
Error logging for segmentation.

Tracks and saves negative values and other anomalies.
"""
import os
import pandas as pd
from typing import List


def log_negative_values(
    house_id: str,
    run_number: int,
    data: pd.DataFrame,
    columns: List[str],
    source: str,
    errors_directory: str,
    logger
) -> None:
    """
    Log negative values from the dataset to a CSV file.

    Args:
        house_id: House identifier
        run_number: Current run number
        data: DataFrame to check
        columns: List of column names to check
        source: Source identifier (e.g., 'original', 'remaining')
        errors_directory: Directory to save error logs
        logger: Logger instance
    """
    os.makedirs(errors_directory, exist_ok=True)

    # Check for any negative values
    has_negatives = (data[columns] < 0).any().any()

    if not has_negatives:
        return

    # Melt data to get phase and power columns
    error_df = data.melt(
        id_vars=['timestamp'],
        value_vars=columns,
        var_name='phase',
        value_name='power'
    )

    # Keep only negative rows
    error_df = error_df[error_df['power'] < 0].copy()
    error_df['house_id'] = house_id
    error_df['run_number'] = run_number
    error_df['source'] = source

    error_file_path = os.path.join(errors_directory, f"errors_{house_id}_run{run_number}.csv")

    try:
        file_exists = os.path.isfile(error_file_path)
        error_df.to_csv(error_file_path, index=False, mode='a', header=not file_exists)
        logger.info(f"Errors saved to {error_file_path} from source: {source}")
    except Exception as e:
        logger.error(f"Failed to save error log: {e}")
