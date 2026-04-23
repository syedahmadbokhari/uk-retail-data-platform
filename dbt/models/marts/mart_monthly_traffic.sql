-- Mart: monthly traffic
-- Groups page visits by YYYY-MM month bucket.

SELECT
    LEFT(last_visited, 7) AS month,
    COUNT(*)              AS visit_count
FROM {{ ref('stg_traffic') }}
WHERE last_visited IS NOT NULL
GROUP BY LEFT(last_visited, 7)
ORDER BY month
