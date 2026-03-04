import sqlite3

try:
    # Connect to database (make sure file is in same folder)
    conn = sqlite3.connect(r"C:\Users\ahmad\Downloads\SQL\retailDB.sqlite")
    cursor = conn.cursor()

    print("Connected successfully.")

    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print("Tables:", cursor.fetchall())

    # Check traffic table row count
    cursor.execute("SELECT COUNT(*) FROM traffic;")
    print("Traffic rows:", cursor.fetchall())

    # Run main query
    query = """
    SELECT 
        SUBSTR(modified_last_visited, 1, 7) AS month,
        COUNT(*) AS visit_count
    FROM traffic
    GROUP BY month
    ORDER BY month;
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    print("\nMonthly Traffic Trend:")
    for row in rows:
        print(row)

    conn.close()

except Exception as e:
    print("ERROR:", e)