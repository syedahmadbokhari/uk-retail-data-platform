import sqlite3

try:
    conn = sqlite3.connect("retailDB.sqlite")
    cursor = conn.cursor()

    print("Connected successfully.\n")

    # ==============================
    # 1️⃣ Revenue by Review Volume
    # ==============================

    query_review_volume = """
    SELECT 
        CASE 
            WHEN r.real_reviews < 100 THEN 'Low Reviews'
            WHEN r.real_reviews BETWEEN 100 AND 500 THEN 'Medium Reviews'
            ELSE 'High Reviews'
        END AS review_volume,
        SUM(f.modified_revenue) AS total_revenue
    FROM finance f
    JOIN reviews r ON f.product_id = r.product_id
    GROUP BY review_volume
    ORDER BY total_revenue DESC;
    """

    cursor.execute(query_review_volume)
    review_volume_results = cursor.fetchall()

    print("Revenue by Review Volume:")
    for row in review_volume_results:
        print(row)


    # ==============================
    # 2️⃣ Revenue vs Rating
    # ==============================

    query_rating = """
    SELECT 
        r.real_rating,
        SUM(f.modified_revenue) AS total_revenue
    FROM finance f
    JOIN reviews r ON f.product_id = r.product_id
    GROUP BY r.real_rating
    ORDER BY r.real_rating DESC;
    """

    cursor.execute(query_rating)
    rating_results = cursor.fetchall()

    print("\nRevenue by Rating:")
    for row in rating_results:
        print(row)

    conn.close()

except Exception as e:
    print("ERROR:", e)