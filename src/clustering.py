import time
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.utils.db import get_connection
from src.utils.logger import get_logger

logger = get_logger("clustering")

# Features used for distance-based clustering.
# Excludes brand_encoded (label-encoded ordinal — integer gaps are meaningless
# for Euclidean distance) and review_count (highly correlated with revenue,
# adding it would double-weight the popularity signal without new information).
# revenue is already log-transformed in features_products.
CLUSTER_COLS = ["revenue", "listing_price", "discount", "rating"]
N_CLUSTERS   = 3
_LABEL_ORDER = ["Low Performer", "Mid Tier", "Premium"]


# ── Elbow-method utility — run manually to validate k=3 ──────────────────────
# Usage: python -c "from src.clustering import plot_elbow; plot_elbow()"
#
# def plot_elbow() -> None:
#     import os
#     import matplotlib.pyplot as plt
#     from src.utils.db import get_connection, get_root
#     with get_connection() as conn:
#         df = pd.read_sql("SELECT * FROM features_products", conn)
#     X = StandardScaler().fit_transform(df[CLUSTER_COLS].dropna())
#     ks, inertias = range(2, 11), []
#     for k in ks:
#         inertias.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_)
#     plt.plot(list(ks), inertias, marker="o")
#     plt.xlabel("k"); plt.ylabel("Inertia"); plt.title("KMeans elbow — features_products")
#     plt.tight_layout()
#     out = os.path.join(get_root(), "outputs", "elbow_plot.png")
#     plt.savefig(out); plt.close()
#     print(f"Saved: {out}")


def build_clusters() -> pd.DataFrame:
    start = time.time()
    logger.info("=== K-means clustering (k=3) ===")

    with get_connection() as conn:
        df = pd.read_sql("SELECT * FROM features_products", conn)

    if df.empty:
        logger.warning("features_products is empty — skipping clustering, no table written")
        return pd.DataFrame()

    logger.info(f"Loaded {len(df)} products")

    # features_products stores median-imputed but unscaled values
    # (revenue is log-transformed; listing_price, discount, rating are raw).
    # StandardScaler is required so no single feature dominates by magnitude.
    X = StandardScaler().fit_transform(df[CLUSTER_COLS])

    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df["_raw_id"] = km.fit_predict(X)

    # Map raw KMeans integers to stable business labels ordered by mean
    # log-revenue: lowest mean → "Low Performer", highest → "Premium".
    # This makes the label assignment independent of which integer KMeans
    # happened to assign first, so re-runs on the same data are consistent.
    mean_rev = df.groupby("_raw_id")["revenue"].mean().sort_values()
    raw_to_stable = {raw_id: rank for rank, raw_id in enumerate(mean_rev.index)}

    df["cluster_id"]    = df["_raw_id"].map(raw_to_stable)
    df["cluster_label"] = df["cluster_id"].map(dict(enumerate(_LABEL_ORDER)))

    result = (
        df[["product_id", "cluster_label", "cluster_id", "revenue"]]
        .copy()
        .rename(columns={"revenue": "log_revenue"})
    )

    counts = result["cluster_label"].value_counts().sort_index()
    for label, count in counts.items():
        logger.info(f"  {label}: {count} products")

    with get_connection() as conn:
        result.to_sql("analytics_product_clusters", conn, if_exists="replace", index=False)

    logger.info(
        f"Clustering complete — {len(result)} products → analytics_product_clusters "
        f"({time.time() - start:.2f}s)"
    )
    return result
