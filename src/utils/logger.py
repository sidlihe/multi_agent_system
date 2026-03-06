import logging
import os
from pathlib import Path

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

class FilenameFilter(logging.Filter):
    """Custom filter to strip .py extension from filename."""
    def filter(self, record: logging.LogRecord) -> bool:
        # record.filename is like "groq_client.py"
        record.filename_noext = Path(record.filename).stem
        return True

# Formatter: use our custom field
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(filename_noext)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M"  # minutes only
)

def get_logger():
    """
    Returns a logger that shows the file name (without extension).
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(FilenameFilter())

    # File handler
    file_handler = logging.FileHandler("logs/project.log")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(FilenameFilter())

    # Attach handlers only once
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
