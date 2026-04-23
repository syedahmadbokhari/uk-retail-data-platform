-- Staging: traffic
-- Strips whitespace from visit timestamps; drops rows with no date.

SELECT
    product_id,
    TRIM(modified_last_visited::TEXT) AS last_visited
FROM {{ source('retail_raw', 'raw_traffic') }}
WHERE product_id            IS NOT NULL
  AND modified_last_visited IS NOT NULL
  AND TRIM(modified_last_visited::TEXT) != ''
