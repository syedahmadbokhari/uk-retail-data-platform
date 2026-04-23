-- Staging: brands
-- Strips whitespace and title-cases brand names.
-- INITCAP is a Postgres built-in (equivalent to Python's str.title()).

SELECT
    product_id,
    INITCAP(TRIM(modified_brand)) AS brand
FROM {{ source('retail_raw', 'raw_brands') }}
WHERE product_id    IS NOT NULL
  AND modified_brand IS NOT NULL
