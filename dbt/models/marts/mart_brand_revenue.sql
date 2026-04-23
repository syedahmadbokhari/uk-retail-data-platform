-- Mart: brand revenue
-- Aggregates total revenue per brand with market-share percentage.
-- Window function computes share across all brands in a single pass.

SELECT
    b.brand,
    SUM(f.revenue)                                          AS total_revenue,
    COUNT(DISTINCT f.product_id)                            AS product_count,
    ROUND(
        SUM(f.revenue) * 100.0 / SUM(SUM(f.revenue)) OVER (),
        2
    )                                                       AS revenue_share_pct
FROM {{ ref('stg_finance') }} f
JOIN {{ ref('stg_brands') }}  b ON f.product_id = b.product_id
GROUP BY b.brand
ORDER BY total_revenue DESC
