import logging
import os
import sys
from pathlib import Path

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

class FilenameFilter(logging.Filter):
    """Custom filter to strip .py extension from filename."""
    def filter(self, record: logging.LogRecord) -> bool:
        # record.filename is like "groq_client.py"
        record.filename_noext = Path(record.filename).stem
        return True

# Formatter: use our custom field (simplified to reduce noise)
formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(filename_noext)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M"
)

def get_logger(name: str = __name__):
    """
    Returns a logger that shows the file name (without extension).
    Handles Unicode properly on Windows with UTF-8 encoding and error handling.
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)  # Changed from DEBUG to INFO to reduce noise

    # Only add handlers once
    if logger.handlers:
        return logger

    # Console handler with UTF-8 encoding and error handling
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(FilenameFilter())
    # Force UTF-8 encoding with 'replace' error handling to prevent UnicodeEncodeError
    console_handler.setStream(sys.stdout)
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler("logs/project.log", encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.addFilter(FilenameFilter())

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
