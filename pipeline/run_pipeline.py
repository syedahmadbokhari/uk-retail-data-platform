"""
Pipeline orchestrator — runs all steps in sequence.

Usage:
    python pipeline/run_pipeline.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger
from src.data_generator.generate_events import generate_events
from src.etl.ingest                      import ingest_raw
from src.etl.ingest_events               import ingest_incremental
from src.etl.clean                       import clean_tables
from src.etl.aggregate                   import build_analytics
from src.features.build_features         import build_features
from src.recommender                     import build_similarity_matrix

logger = get_logger("pipeline")

STEPS = [
    ("1/7  Generate     synthetic events",     lambda: generate_events(n_events=200)),
    ("2/7  Ingest       static source tables", ingest_raw),
    ("3/7  Ingest       new events (incr.)",   ingest_incremental),
    ("4/7  Clean        clean layer",          clean_tables),
    ("5/7  Aggregate    analytics layer",      build_analytics),
    ("6/7  Features     feature table",        build_features),
    ("7/7  Similarity   model artifact",       build_similarity_matrix),
]


def run():
    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("PIPELINE START")
    logger.info("=" * 60)

    for label, fn in STEPS:
        step_start = time.time()
        logger.info(f"--- {label} ---")
        try:
            fn()
            logger.info(f"    done in {time.time() - step_start:.2f}s")
        except Exception as exc:
            logger.error(f"FAILED at step [{label}]: {exc}", exc_info=True)
            sys.exit(1)

    total = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE — {total:.2f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()
