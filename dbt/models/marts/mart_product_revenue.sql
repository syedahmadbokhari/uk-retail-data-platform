-- Mart: product revenue
-- One row per product: total revenue, brand, and dense revenue rank.

SELECT
    i.product_name,
    b.brand,
    SUM(f.revenue)                                          AS total_revenue,
    RANK() OVER (ORDER BY SUM(f.revenue) DESC)              AS revenue_rank
FROM {{ ref('stg_finance') }} f
JOIN {{ ref('stg_info') }}    i ON f.product_id = i.product_id
JOIN {{ ref('stg_brands') }}  b ON f.product_id = b.product_id
GROUP BY i.product_name, b.brand
ORDER BY total_revenue DESC
