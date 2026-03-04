import sqlite3

try:
    conn = sqlite3.connect("retailDB.sqlite")
    cursor = conn.cursor()

    print("Connected successfully.\n")

    # ==============================
    # 1️⃣ Revenue by Discount Category
    # ==============================

    query_category = """
    SELECT 
        CASE 
            WHEN modified_discount > 0 THEN 'Discounted'
            ELSE 'No Discount'
        END AS discount_category,
        SUM(modified_revenue) AS total_revenue,
        AVG(modified_revenue) AS avg_revenue
    FROM finance
    GROUP BY discount_category;
    """

    cursor.execute(query_category)
    category_results = cursor.fetchall()

    print("Revenue by Discount Category:")
    for row in category_results:
        print(row)


    # ==============================
    # 2️⃣ Revenue by Discount Range
    # ==============================

    query_range = """
    SELECT 
        CASE 
            WHEN modified_discount BETWEEN 0 AND 0.20 THEN 'Low Discount'
            WHEN modified_discount BETWEEN 0.21 AND 0.50 THEN 'Medium Discount'
            WHEN modified_discount > 0.50 THEN 'High Discount'
            ELSE 'No Discount'
        END AS discount_range,
        SUM(modified_revenue) AS total_revenue
    FROM finance
    GROUP BY discount_range
    ORDER BY total_revenue DESC;
    """

    cursor.execute(query_range)
    range_results = cursor.fetchall()

    print("\nRevenue by Discount Range:")
    for row in range_results:
        print(row)

    conn.close()

except Exception as e:
    print("ERROR:", e)