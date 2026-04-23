-- Mart: discount impact
-- Compares revenue and volume between discounted and full-price products.

SELECT
    CASE
        WHEN discount > 0 THEN 'Discounted'
        ELSE 'No Discount'
    END          AS category,
    SUM(revenue) AS total_revenue,
    AVG(revenue) AS avg_revenue,
    COUNT(*)     AS product_count
FROM {{ ref('stg_finance') }}
GROUP BY 1
