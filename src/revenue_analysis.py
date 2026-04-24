import sqlite3
import os

_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "retailDB.sqlite")

try:
    conn = sqlite3.connect(_DB)
    cursor = conn.cursor()

    print("Connected successfully.\n")

    # ==============================
    # 1️⃣ Total Revenue
    # ==============================

    query_total = """
    SELECT 
        SUM(modified_revenue) AS total_revenue
    FROM finance;
    """

    cursor.execute(query_total)
    total_revenue = cursor.fetchone()

    print("Total Revenue:")
    print(total_revenue)


    # ==============================
    # 2️⃣ Revenue by Brand
    # ==============================

    query_brand = """
    SELECT 
        b.brand,
        SUM(f.modified_revenue) AS total_revenue
    FROM finance f
    JOIN brands b ON f.product_id = b.product_id
    GROUP BY b.brand
    ORDER BY total_revenue DESC;
    """

    cursor.execute(query_brand)
    brand_results = cursor.fetchall()

    print("\nRevenue by Brand:")
    for row in brand_results:
        print(row)


    # ==============================
    # 3️⃣ Top 10 Products by Revenue
    # ==============================

    query_top_products = """
    SELECT 
        i.product_name,
        SUM(f.modified_revenue) AS total_revenue
    FROM finance f
    JOIN info i ON f.product_id = i.product_id
    GROUP BY i.product_name
    ORDER BY total_revenue DESC
    LIMIT 10;
    """

    cursor.execute(query_top_products)
    top_products = cursor.fetchall()

    print("\nTop 10 Products by Revenue:")
    for row in top_products:
        print(row)

    conn.close()

except Exception as e:
    print("ERROR:", e)