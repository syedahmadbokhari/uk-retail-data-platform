-- Staging: product info
-- Trims product names; fills NULL descriptions with empty string.

SELECT
    product_id,
    TRIM(modified_product_name)              AS product_name,
    COALESCE(modified_description, '')       AS description
FROM {{ source('retail_raw', 'raw_info') }}
WHERE product_id            IS NOT NULL
  AND modified_product_name IS NOT NULL
