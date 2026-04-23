-- Staging: finance
-- Casts string columns to numeric, clips discount to [0, 1], drops invalid rows.
-- Mirrors the logic in src/etl/clean.py::_clean_finance but executed inside Postgres.

SELECT
    product_id,
    COALESCE(modified_listing_price::FLOAT, 0)  AS listing_price,
    COALESCE(modified_sale_price::FLOAT,    0)  AS sale_price,
    LEAST(GREATEST(
        COALESCE(modified_discount::FLOAT, 0), 0
    ), 1)                                        AS discount,
    modified_revenue::FLOAT                      AS revenue
FROM {{ source('retail_raw', 'raw_finance') }}
WHERE product_id   IS NOT NULL
  AND modified_revenue IS NOT NULL
  AND modified_revenue::FLOAT >= 0
