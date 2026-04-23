import os
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.db import get_connection, get_root
from src.utils.logger import get_logger
from src.utils.validation import validate

logger = get_logger("features.build")

# One row per product: aggregate finance metrics, join reviews + brand
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
    COALESCE(r.rating,       0) AS rating,
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

    # Fill remaining nulls
    df["rating"]        = df["rating"].fillna(df["rating"].median())
    df["review_count"]  = df["review_count"].fillna(0)
    df["listing_price"] = df["listing_price"].fillna(0)
    df["discount"]      = df["discount"].fillna(0)

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
