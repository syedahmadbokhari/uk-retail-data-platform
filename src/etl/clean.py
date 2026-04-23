import time
import pandas as pd
from src.utils.db import get_connection
from src.utils.logger import get_logger
from src.utils.validation import validate

logger = get_logger("etl.clean")


def _clean_finance(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "product_id", "modified_listing_price", "modified_sale_price",
        "modified_discount", "modified_revenue",
    ]
    df = df[cols].copy()
    df = df.dropna(subset=["product_id", "modified_revenue"])
    df = df[df["modified_revenue"] >= 0]
    df["modified_discount"] = df["modified_discount"].clip(0, 1).fillna(0)
    df["modified_listing_price"] = df["modified_listing_price"].fillna(0)
    df["modified_sale_price"] = df["modified_sale_price"].fillna(0)
    return df.reset_index(drop=True)


def _clean_brands(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["product_id", "modified_brand"]].copy()
    df = df.dropna(subset=["product_id", "modified_brand"])
    df["modified_brand"] = df["modified_brand"].str.strip().str.title()
    return df.reset_index(drop=True)


def _clean_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["product_id", "modified_product_name", "modified_description"]].copy()
    df = df.dropna(subset=["product_id", "modified_product_name"])
    df["modified_product_name"] = df["modified_product_name"].str.strip()
    df["modified_description"] = df["modified_description"].fillna("")
    return df.reset_index(drop=True)


def _clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["product_id", "real_rating", "real_reviews"]].copy()

    # Drop rows with empty/null product_id (trailing garbage rows in source)
    df = df.dropna(subset=["product_id"])
    df = df[df["product_id"].astype(str).str.strip() != ""]

    # real_rating is stored in European decimal format ("3,3" means 3.3)
    df["real_rating"] = (
        df["real_rating"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df["real_rating"] = pd.to_numeric(df["real_rating"], errors="coerce").clip(0, 5)

    df["real_reviews"] = pd.to_numeric(df["real_reviews"], errors="coerce").fillna(0).clip(lower=0)

    return df.reset_index(drop=True)


def _clean_traffic(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["product_id", "modified_last_visited"]].copy()
    df = df.dropna(subset=["product_id", "modified_last_visited"])
    df["modified_last_visited"] = df["modified_last_visited"].astype(str).str.strip()
    df = df[df["modified_last_visited"] != ""]
    return df.reset_index(drop=True)


_CLEANERS = {
    "finance": (_clean_finance, ["product_id"]),
    "brands":  (_clean_brands,  ["product_id"]),
    "info":    (_clean_info,    ["product_id"]),
    "reviews": (_clean_reviews, ["product_id"]),
    "traffic": (_clean_traffic, ["product_id"]),
}


def clean_tables():
    start = time.time()
    logger.info("=== Cleaning: raw → clean layer ===")

    with get_connection() as conn:
        for tbl, (cleaner, keys) in _CLEANERS.items():
            raw = pd.read_sql(f"SELECT * FROM raw_{tbl}", conn)
            cleaned = cleaner(raw)
            validate(cleaned, f"clean_{tbl}", critical_cols=["product_id"], key_cols=keys)
            cleaned.to_sql(f"clean_{tbl}", conn, if_exists="replace", index=False)
            logger.info(f"  clean_{tbl}: {len(cleaned)} rows (was {len(raw)})")

    logger.info(f"Cleaning complete in {time.time() - start:.2f}s")
