import time
import pandas as pd
from src.utils.db import get_connection
from src.utils.logger import get_logger

logger = get_logger("etl.aggregate")


def _brand_revenue(conn) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT b.modified_brand AS brand,
               SUM(f.modified_revenue) AS total_revenue,
               COUNT(DISTINCT f.product_id) AS product_count
        FROM clean_finance f
        JOIN clean_brands b ON f.product_id = b.product_id
        GROUP BY b.modified_brand
        ORDER BY total_revenue DESC
        """,
        conn,
    )
    grand = df["total_revenue"].sum()
    df["revenue_share_pct"] = (df["total_revenue"] / grand * 100).round(2)
    return df


def _product_revenue(conn) -> pd.DataFrame:
    df = pd.read_sql(
        """
        SELECT i.modified_product_name AS product_name,
               b.modified_brand        AS brand,
               SUM(f.modified_revenue) AS total_revenue
        FROM clean_finance f
        JOIN clean_info   i ON f.product_id = i.product_id
        JOIN clean_brands b ON f.product_id = b.product_id
        GROUP BY i.modified_product_name
        ORDER BY total_revenue DESC
        """,
        conn,
    )
    df["revenue_rank"] = df["total_revenue"].rank(method="min", ascending=False).astype(int)
    return df


def _monthly_traffic(conn) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT SUBSTR(modified_last_visited, 1, 7) AS month,
               COUNT(*) AS visit_count
        FROM clean_traffic
        GROUP BY month
        ORDER BY month
        """,
        conn,
    )


def _discount_impact(conn) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT CASE WHEN modified_discount > 0 THEN 'Discounted' ELSE 'No Discount' END AS category,
               SUM(modified_revenue)   AS total_revenue,
               AVG(modified_revenue)   AS avg_revenue,
               COUNT(*)                AS product_count
        FROM clean_finance
        GROUP BY category
        """,
        conn,
    )


_BUILDERS = {
    "analytics_brand_revenue":   _brand_revenue,
    "analytics_product_revenue": _product_revenue,
    "analytics_monthly_traffic": _monthly_traffic,
    "analytics_discount_impact": _discount_impact,
}


def build_analytics():
    start = time.time()
    logger.info("=== Aggregation: clean → analytics layer ===")

    with get_connection() as conn:
        for name, builder in _BUILDERS.items():
            df = builder(conn)
            df.to_sql(name, conn, if_exists="replace", index=False)
            logger.info(f"  {name}: {len(df)} rows")

    logger.info(f"Aggregation complete in {time.time() - start:.2f}s")
