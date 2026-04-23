import logging
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOGS_DIR = os.path.join(_ROOT, "logs")
os.makedirs(_LOGS_DIR, exist_ok=True)

_FMT = logging.Formatter(
    "%(asctime)s | %(name)-22s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(_FMT)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(_LOGS_DIR, "pipeline.log"), encoding="utf-8")
    fh.setFormatter(_FMT)
    logger.addHandler(fh)

    return logger
