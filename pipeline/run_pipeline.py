"""
Pipeline orchestrator — runs all steps in sequence.

Usage:
    python pipeline/run_pipeline.py
"""

import os
import sys
import time

# Ensure project root is on the path so src.* imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import get_logger
from src.etl.ingest import ingest_raw
from src.etl.clean import clean_tables
from src.etl.aggregate import build_analytics
from src.features.build_features import build_features
from src.recommender import build_similarity_matrix

logger = get_logger("pipeline")

STEPS = [
    ("1/5  Ingest       raw layer",      ingest_raw),
    ("2/5  Clean        clean layer",    clean_tables),
    ("3/5  Aggregate    analytics layer",build_analytics),
    ("4/5  Features     feature table",  build_features),
    ("5/5  Similarity   model artifact", build_similarity_matrix),
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
