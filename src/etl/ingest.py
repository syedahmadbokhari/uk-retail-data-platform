import time
import pandas as pd
from src.utils.db import get_connection
from src.utils.logger import get_logger
from src.utils.validation import validate

logger = get_logger("etl.ingest")

_SOURCE_TABLES = ["finance", "brands", "info", "reviews", "traffic"]


def ingest_raw():
    start = time.time()
    logger.info("=== Ingestion: source → raw layer ===")

    with get_connection() as conn:
        for tbl in _SOURCE_TABLES:
            df = pd.read_sql(f"SELECT * FROM {tbl}", conn)
            validate(df, tbl)
            df.to_sql(f"raw_{tbl}", conn, if_exists="replace", index=False)
            logger.info(f"  raw_{tbl}: {len(df)} rows loaded")

    logger.info(f"Ingestion complete in {time.time() - start:.2f}s")
