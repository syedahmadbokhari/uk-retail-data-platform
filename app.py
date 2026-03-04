import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# ------------------------------------------------
# Page Settings
# ------------------------------------------------
st.set_page_config(
    page_title="Retail Revenue Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Revenue Dashboard")
st.markdown("Interactive analytics dashboard for retail revenue performance.")

# ------------------------------------------------
# Database Connection
# ------------------------------------------------
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "retailDB.sqlite")

def load_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

conn = load_connection()

# ------------------------------------------------
# KPI Queries
# ------------------------------------------------
total_revenue_query = """
SELECT SUM(modified_revenue) AS total_revenue
FROM finance;
"""

brand_share_query = """
WITH brand_revenue AS (
    SELECT 
        b.brand,
        SUM(f.modified_revenue) AS total_revenue
    FROM finance f
    JOIN brands b ON f.product_id = b.product_id
    GROUP BY b.brand
)
SELECT 
    brand,
    total_revenue,
    ROUND(
        total_revenue * 100.0 / SUM(total_revenue) OVER (), 
        2
    ) AS revenue_share_percentage
FROM brand_revenue
ORDER BY total_revenue DESC;
"""

total_revenue = pd.read_sql(total_revenue_query, conn)
brand_share = pd.read_sql(brand_share_query, conn)

# ------------------------------------------------
# KPI DISPLAY
# ------------------------------------------------
col1, col2 = st.columns(2)

col1.metric(
    "💰 Total Revenue",
    f"${int(total_revenue['total_revenue'][0]):,}"
)

col2.metric(
    "🏆 Top Brand Share",
    f"{brand_share['revenue_share_percentage'][0]}%"
)

st.divider()

# ------------------------------------------------
# Revenue by Brand
# ------------------------------------------------
brand_query = """
SELECT 
    b.brand,
    SUM(f.modified_revenue) AS total_revenue
FROM finance f
JOIN brands b ON f.product_id = b.product_id
GROUP BY b.brand
ORDER BY total_revenue DESC;
"""

brand_df = pd.read_sql(brand_query, conn)

fig_brand = px.bar(
    brand_df,
    x="brand",
    y="total_revenue",
    title="Revenue by Brand",
    color="brand"
)

# ------------------------------------------------
# Discount Category
# ------------------------------------------------
discount_query = """
SELECT 
    CASE 
        WHEN modified_discount > 0 THEN 'Discounted'
        ELSE 'No Discount'
    END AS discount_category,
    SUM(modified_revenue) AS total_revenue
FROM finance
GROUP BY discount_category;
"""

discount_df = pd.read_sql(discount_query, conn)

fig_discount = px.pie(
    discount_df,
    names="discount_category",
    values="total_revenue",
    title="Revenue by Discount Category"
)

# ------------------------------------------------
# Display Charts in Columns
# ------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_brand, width="stretch")

with col2:
    st.plotly_chart(fig_discount, width="stretch")

st.divider()

# ------------------------------------------------
# Review vs Revenue
# ------------------------------------------------
rating_query = """
SELECT 
    r.real_rating,
    SUM(f.modified_revenue) AS total_revenue
FROM finance f
JOIN reviews r ON f.product_id = r.product_id
GROUP BY r.real_rating
ORDER BY r.real_rating DESC;
"""

rating_df = pd.read_sql(rating_query, conn)

# Remove rows with missing values
rating_df = rating_df.dropna()

fig_rating = px.scatter(
    rating_df,
    x="real_rating",
    y="total_revenue",
    title="Revenue vs Product Rating",
    size="total_revenue"
)
# ------------------------------------------------
# Monthly Traffic
# ------------------------------------------------
traffic_query = """
SELECT 
    SUBSTR(modified_last_visited, 1, 7) AS month,
    COUNT(*) AS visit_count
FROM traffic
GROUP BY month
ORDER BY month;
"""

traffic_df = pd.read_sql(traffic_query, conn)

fig_traffic = px.line(
    traffic_df,
    x="month",
    y="visit_count",
    markers=True,
    title="Monthly Website Traffic"
)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_rating, width="stretch")

with col2:
    st.plotly_chart(fig_traffic, width="stretch")

st.divider()

# ------------------------------------------------
# Top Products
# ------------------------------------------------
top_products_query = """
SELECT 
    i.product_name,
    SUM(f.modified_revenue) AS total_revenue
FROM finance f
JOIN info i ON f.product_id = i.product_id
GROUP BY i.product_name
ORDER BY total_revenue DESC
LIMIT 10;
"""

top_products_df = pd.read_sql(top_products_query, conn)

st.subheader("🏆 Top 10 Products by Revenue")
st.dataframe(top_products_df)

conn.close()