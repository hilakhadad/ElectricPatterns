"""
Centralized logging setup for experiment pipeline.

Provides consistent logging across all modules with clear source identification.
"""
import logging
import os
from pathlib import Path
from typing import Optional

from .paths import LOGS_DIRECTORY


def setup_logging(house_id: str, run_number: int, logs_directory: str = LOGS_DIRECTORY) -> logging.Logger:
    """
    Set up logging for a specific house and run number.

    This is the main logging function used by pipeline steps.

    Args:
        house_id: ID of the house being processed
        run_number: Run number of the pipeline
        logs_directory: Base directory for logs

    Returns:
        Configured logger instance
    """
    os.makedirs(logs_directory, exist_ok=True)

    log_file = os.path.join(logs_directory, f"{house_id}.log")

    logger = logging.getLogger(f"{house_id}_run_{run_number}")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def setup_pipeline_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.

    Args:
        name: Logger name (usually module name like 'detection.sharp')
        log_file: Optional path to log file
        level: Logging level
        console: Whether to also log to console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a basic one."""
    return logging.getLogger(name)


class PipelineLogger:
    """
    Context-aware logger that includes house/run info in messages.

    Usage:
        logger = PipelineLogger('detection', house_id='2', run_number=0)
        logger.info("Found 10 events")  # Outputs: [house_2/run_0] Found 10 events
    """

    def __init__(self, module_name: str, house_id: str, run_number: int,
                 log_file: Optional[Path] = None):
        self.house_id = house_id
        self.run_number = run_number
        self.prefix = f"[house_{house_id}/run_{run_number}]"
        self._logger = setup_pipeline_logger(
            f"{module_name}.house_{house_id}",
            log_file=log_file,
            console=False
        )

    def _format(self, msg: str) -> str:
        return f"{self.prefix} {msg}"

    def info(self, msg: str):
        self._logger.info(self._format(msg))

    def warning(self, msg: str):
        self._logger.warning(self._format(msg))

    def error(self, msg: str):
        self._logger.error(self._format(msg))

    def debug(self, msg: str):
        self._logger.debug(self._format(msg))
