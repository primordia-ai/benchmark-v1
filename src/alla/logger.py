"""
Call this before other scripts to use the alla logger
"""
from alla.utils import PROJECT_ROOT

# From logging module
CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0


def reset_logger(level=INFO, log_file="out.log"):
    """Re-initialize ALLA logger with one of the above options"""
    import logging
    # Create a custom logger
    _logger = logging.getLogger("alla")

    # Remove any existing handlers
    _logger.handlers.clear()

    _logger.setLevel(level)  # Set the minimum level for the logger

    # Add a StreamHandler to the logger
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)  # Set the minimum level for the handler

    # Add a FileHandler to the logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)  # Set the minimum level for the handler

    # Add an optional formatter for better output
    formatter = logging.Formatter('[%(name)s|%(levelname)s]\t%(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    _logger.addHandler(stream_handler)
    _logger.addHandler(file_handler)

    return _logger


def get_logger():
    """Initialize ALLA logger with one of the above options"""
    import logging
    # Create a custom logger
    _logger = logging.getLogger("alla")

    # Check if the logger already has handlers attached
    if not _logger.hasHandlers():
        return reset_logger()

    return _logger


logger = get_logger()