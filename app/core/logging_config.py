import logging
import sys


def setup_logging(default_level: int = logging.INFO) -> None:
    """Configures the root logger with a clean format and outputs to stdout."""
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"

    logging.basicConfig(
        level=default_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_logger(name: str) -> logging.Logger:
    """Returns a logger instance with the given name."""
    return logging.getLogger(name)
