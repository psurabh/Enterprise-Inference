"""Simple timing decorator for performance tracking."""
import time
import functools
import logging

logger = logging.getLogger(__name__)

def track_time(label: str, log_level: str = "INFO"):
    """
    Decorator to track execution time of functions.
    
    Args:
        label: Label for the timing log
        log_level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                log_func = getattr(logger, log_level.lower(), logger.info)
                log_func(f"{label}: {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"{label} failed after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator
