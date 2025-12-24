"""
Simple logging utility using loguru.
Logs to both console and file.
"""

from loguru import logger
import sys


def setup_logger(log_file=None, level="INFO"):
    """
    Setup logger for training/evaluation.

    Args:
        log_file: Path to log file. If None, only logs to console.
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()

    # Add console handler (colored, INFO level)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
        colorize=True
    )

    # Add file handler if specified (DEBUG level, all details)
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="30 days"  # Keep logs for 30 days
        )

    return logger


if __name__ == "__main__":
    """Test logger"""
    test_logger = setup_logger("test.log")
    test_logger.info("This is an info message")
    test_logger.debug("This is a debug message")
    test_logger.warning("This is a warning")
    test_logger.error("This is an error")
    test_logger.success("This is a success message")
    print("\n✓ Logger test complete. Check test.log")
