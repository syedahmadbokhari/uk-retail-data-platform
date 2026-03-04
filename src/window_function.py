import sqlite3

try:
    conn = sqlite3.connect("retailDB.sqlite")
    cursor = conn.cursor()

    print("Connected successfully.\n")

    # ==============================
    # 1️⃣ Rank Products by Revenue
    # ==============================

    query_rank = """
    SELECT 
        i.product_name,
        SUM(f.modified_revenue) AS total_revenue,
        RANK() OVER (ORDER BY SUM(f.modified_revenue) DESC) AS revenue_rank
    FROM finance f
    JOIN info i ON f.product_id = i.product_id
    GROUP BY i.product_name
    ORDER BY revenue_rank;
    """

    cursor.execute(query_rank)
    rank_results = cursor.fetchall()

    print("Product Revenue Ranking:")
    for row in rank_results[:10]:   # Show top 10 only
        print(row)


    # ==============================
    # 2️⃣ Brand Revenue Share
    # ==============================

    query_share = """
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

    cursor.execute(query_share)
    share_results = cursor.fetchall()

    print("\nBrand Revenue Share:")
    for row in share_results:
        print(row)

    conn.close()

except Exception as e:
    print("ERROR:", e)