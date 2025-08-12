"""Logging utilities for the application"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format: Optional[str] = None,
    log_file: Optional[str] = None
):
    """
    Setup logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format string (uses default if None)
        log_file: Optional file to write logs to
    """
    if format is None:
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format,
        handlers=handlers
    )
    
    # Set specific loggers to WARNING to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)