import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.db import get_connection, get_root
from src.utils.logger import get_logger
from src.utils.validation import validate

logger = get_logger("features.build")

# One row per product: aggregate finance metrics, join reviews + brand.
# Note: r.rating is NOT wrapped in COALESCE — NULLs are intentionally
# preserved so Python can impute with the column median rather than
# treating unknown ratings as 0 (which would distort cosine similarity).
_FEATURE_QUERY = """
WITH product_finance AS (
    SELECT product_id,
           AVG(modified_listing_price) AS listing_price,
           AVG(modified_discount)      AS discount,
           SUM(modified_revenue)       AS revenue
    FROM clean_finance
    GROUP BY product_id
),
product_reviews AS (
    SELECT product_id,
           AVG(real_rating)    AS rating,
           SUM(real_reviews)   AS review_count
    FROM clean_reviews
    GROUP BY product_id
)
SELECT
    f.product_id,
    i.modified_product_name AS product_name,
    b.modified_brand        AS brand,
    f.listing_price,
    f.discount,
    f.revenue,
    r.rating,
    COALESCE(r.review_count, 0) AS review_count
FROM product_finance f
JOIN clean_info   i ON f.product_id = i.product_id
JOIN clean_brands b ON f.product_id = b.product_id
LEFT JOIN product_reviews r ON f.product_id = r.product_id
"""

FEATURE_COLS = ["brand_encoded", "listing_price", "discount", "revenue", "rating", "review_count"]


def build_features() -> pd.DataFrame:
    start = time.time()
    logger.info("=== Feature engineering ===")

    with get_connection() as conn:
        df = pd.read_sql(_FEATURE_QUERY, conn)

    logger.info(f"Raw join result: {len(df)} products")

    # Impute with median rather than 0.
    # Filling with 0 collapses feature vectors: products with no price data
    # all become identical on that dimension, driving cosine similarity to ~100%.
    df["listing_price"] = df["listing_price"].fillna(df["listing_price"].median())
    df["review_count"]  = df["review_count"].fillna(df["review_count"].median())
    df["discount"]      = df["discount"].fillna(0)  # 0 is a valid default for discount

    # rating=0 means no rating data, not a genuine 0-star product (ratings run 1–5).
    # Impute both NaN and 0 with the median so these products don't cluster together.
    df["rating"] = df["rating"].replace(0, np.nan)
    df["rating"] = df["rating"].fillna(df["rating"].median())

    # Drop rows where listing_price is still <= 0 after imputation.
    # A £0 price cannot be real — these are corrupted source rows that would
    # cluster together and produce artificially high similarity scores.
    n_before = len(df)
    df = df[df["listing_price"] > 0].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(f"Dropped {n_dropped} rows with listing_price <= 0 (corrupted source data)")

    # Log-transform revenue before storing.
    # Revenue is right-skewed (£0–£37,150): without log transform StandardScaler
    # compresses all low-revenue products together, reducing their discrimination.
    # log1p handles the zero-revenue edge case (log(0) is undefined).
    df["revenue"] = np.log1p(df["revenue"])

    # Fail fast if any invariant is violated before writing to DB
    assert df["listing_price"].min() > 0,          "listing_price must be positive after cleaning"
    assert df["listing_price"].isna().sum() == 0,  "listing_price must have no nulls"
    assert df["review_count"].isna().sum() == 0,   "review_count must have no nulls"
    assert df["rating"].isna().sum() == 0,         "rating must have no nulls after imputation"

    # Label-encode brand
    le = LabelEncoder()
    df["brand_encoded"] = le.fit_transform(df["brand"])
    mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
    logger.info(f"Brand encoding: {mapping}")

    feature_df = df[[
        "product_id", "product_name", "brand", "brand_encoded",
        "listing_price", "discount", "revenue", "rating", "review_count",
    ]].copy()

    validate(
        feature_df, "features_products",
        critical_cols=["product_id", "product_name"],
    )

    # Persist to DB table
    with get_connection() as conn:
        feature_df.to_sql("features_products", conn, if_exists="replace", index=False)

    # Persist to CSV
    out_dir = os.path.join(get_root(), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "features_products.csv")
    feature_df.to_csv(csv_path, index=False)

    logger.info(
        f"Feature table: {len(feature_df)} products saved "
        f"(DB + {csv_path}) — {time.time() - start:.2f}s"
    )
    return feature_df
