import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ── project root on path so src.* imports work when app is launched directly ──
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE_DIR)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Revenue Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Retail Revenue Dashboard")
st.markdown("Interactive analytics dashboard for retail revenue performance.")

# ── Database connection ────────────────────────────────────────────────────────
DB_PATH = os.path.join(_BASE_DIR, "data", "retailDB.sqlite")
def load_connection():
    if not os.path.exists(DB_PATH):
        st.error("❌ Database file not found.")
        st.stop()
    return sqlite3.connect(DB_PATH, check_same_thread=False)


conn = load_connection()

cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
if len(cursor.fetchall()) == 0:
    st.error("❌ Database has no tables.")
    st.stop()

# ── KPI queries ────────────────────────────────────────────────────────────────
total_revenue_query = "SELECT SUM(modified_revenue) AS total_revenue FROM finance;"

brand_share_query = """
WITH brand_revenue AS (
    SELECT b.brand,
           SUM(f.modified_revenue) AS total_revenue
    FROM finance f
    JOIN brands b ON f.product_id = b.product_id
    GROUP BY b.brand
)
SELECT brand,
       total_revenue,
       ROUND(total_revenue * 100.0 / SUM(total_revenue) OVER (), 2) AS revenue_share_percentage
FROM brand_revenue
ORDER BY total_revenue DESC;
"""

try:
    total_revenue = pd.read_sql(total_revenue_query, conn)
    brand_share   = pd.read_sql(brand_share_query, conn)
except Exception as e:
    st.error(f"SQL Error: {e}")
    st.stop()

# ── KPI display ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
col1.metric("💰 Total Revenue",   f"£{int(total_revenue['total_revenue'][0]):,}")
col2.metric("🏆 Top Brand Share", f"{brand_share['revenue_share_percentage'][0]}%")

st.divider()

# ── Revenue by Brand ───────────────────────────────────────────────────────────
brand_query = """
SELECT b.brand,
       SUM(f.modified_revenue) AS total_revenue
FROM finance f
JOIN brands b ON f.product_id = b.product_id
GROUP BY b.brand
ORDER BY total_revenue DESC;
"""
brand_df  = pd.read_sql(brand_query, conn)
fig_brand = px.bar(brand_df, x="brand", y="total_revenue",
                   title="Revenue by Brand", color="brand")

# ── Discount category ──────────────────────────────────────────────────────────
discount_query = """
SELECT CASE WHEN modified_discount > 0 THEN 'Discounted' ELSE 'No Discount' END AS discount_category,
       SUM(modified_revenue) AS total_revenue
FROM finance
GROUP BY discount_category;
"""
discount_df  = pd.read_sql(discount_query, conn)
fig_discount = px.pie(discount_df, names="discount_category", values="total_revenue",
                      title="Revenue by Discount Category")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_brand, use_container_width=True)
with col2:
    st.plotly_chart(fig_discount, use_container_width=True)

st.divider()

# ── Revenue vs Rating ──────────────────────────────────────────────────────────
rating_query = """
SELECT r.real_rating,
       SUM(f.modified_revenue) AS total_revenue
FROM finance f
JOIN reviews r ON f.product_id = r.product_id
GROUP BY r.real_rating
ORDER BY r.real_rating DESC;
"""
rating_df = pd.read_sql(rating_query, conn).dropna()
fig_rating = px.scatter(rating_df, x="real_rating", y="total_revenue",
                        title="Revenue vs Product Rating", size="total_revenue")

# ── Monthly traffic ────────────────────────────────────────────────────────────
traffic_query = """
SELECT SUBSTR(modified_last_visited, 1, 7) AS month,
       COUNT(*) AS visit_count
FROM traffic
GROUP BY month
ORDER BY month;
"""
traffic_df  = pd.read_sql(traffic_query, conn)
fig_traffic = px.line(traffic_df, x="month", y="visit_count",
                      markers=True, title="Monthly Website Traffic")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_rating, use_container_width=True)
with col2:
    st.plotly_chart(fig_traffic, use_container_width=True)

st.divider()

# ── Top 10 products ────────────────────────────────────────────────────────────
top_products_query = """
SELECT i.product_name,
       SUM(f.modified_revenue) AS total_revenue
FROM finance f
JOIN info i ON f.product_id = i.product_id
GROUP BY i.product_name
ORDER BY total_revenue DESC
LIMIT 10;
"""
top_products_df = pd.read_sql(top_products_query, conn)
st.subheader("🏆 Top 10 Products by Revenue")
st.dataframe(top_products_df, use_container_width=True, hide_index=True)

st.divider()

# ── Recommendation engine ──────────────────────────────────────────────────────
st.header("🎯 Product Recommendations")
st.markdown(
    "Content-based filtering using price, discount, revenue, rating, and review count."
)


@st.cache_resource(show_spinner="Building recommendation model…")
def _load_artifact():
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics.pairwise import cosine_similarity
        from src.utils.db import get_connection

        FEATURE_COLS = [
            "brand_encoded", "listing_price", "discount",
            "revenue", "rating", "review_count",
        ]
        with get_connection() as _conn:
            feat_df = pd.read_sql("SELECT * FROM features_products", _conn)

        feat_df = feat_df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
        X = StandardScaler().fit_transform(feat_df[FEATURE_COLS])
        matrix = cosine_similarity(X)
        return {"matrix": matrix, "df": feat_df}
    except Exception as e:
        st.error(f"Could not build recommendation model: {e}")
        return None


def _get_recommendations(
    product_id: str,
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    top_n: int,
) -> pd.DataFrame:
    matches = df.index[df["product_id"] == product_id].tolist()
    if not matches:
        return pd.DataFrame()
    idx    = matches[0]
    scores = sorted(enumerate(sim_matrix[idx]), key=lambda x: x[1], reverse=True)

    # Fetch extra candidates before deduplication: the same product_name can
    # appear under multiple product_ids (different colourways), so we need a
    # larger pool to guarantee top_n unique names after deduplication.
    candidates = [(i, s) for i, s in scores if i != idx][:top_n * 3]

    rows = []
    for i, score in candidates:
        r = df.iloc[i]
        rows.append({
            "Product":     r["product_name"],
            "Brand":       r["brand"],
            "Price (£)":   f"{r['listing_price']:.0f}",
            "Rating":      f"{r['rating']:.2f}",
            "Revenue (£)": f"{r['revenue']:,.0f}",
            "Similarity":  f"{score:.1%}",
            "_score":      score,
        })

    result = pd.DataFrame(rows)
    # Keep one row per product name — the highest-similarity entry
    result = (result
              .sort_values("_score", ascending=False)
              .drop_duplicates(subset="Product", keep="first")
              .drop(columns=["_score"])
              .head(top_n)
              .reset_index(drop=True))
    return result


artifact = _load_artifact()

if artifact is None:
    st.info("⚠️ Recommendation model could not be loaded.")
else:
    feat_df    = artifact["df"]
    sim_matrix = artifact["matrix"]

    product_names = sorted(feat_df["product_name"].dropna().unique().tolist())

    col1, col2 = st.columns([4, 1])
    with col1:
        selected_name = st.selectbox("Select a product:", product_names)
    with col2:
        top_n = st.slider("Results", min_value=3, max_value=10, value=5)

    if st.button("Get Recommendations", type="primary"):
        pid  = feat_df.loc[feat_df["product_name"] == selected_name, "product_id"].iloc[0]
        recs = _get_recommendations(pid, feat_df, sim_matrix, top_n)
        if recs.empty:
            st.warning("No recommendations found for this product.")
        else:
            st.dataframe(recs, use_container_width=True, hide_index=True)

# ── close connection ───────────────────────────────────────────────────────────
