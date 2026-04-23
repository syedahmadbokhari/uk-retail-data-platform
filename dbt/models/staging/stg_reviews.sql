-- Staging: reviews
-- Handles the European decimal bug: source stores ratings as "3,3" (text).
-- REPLACE converts the comma to a period before casting to FLOAT.
-- Rating is clipped to [0, 5]; review count floored at 0.

SELECT
    product_id,
    LEAST(GREATEST(
        REPLACE(real_rating::TEXT, ',', '.')::FLOAT,
        0
    ), 5)                                        AS rating,
    GREATEST(
        COALESCE(real_reviews::FLOAT, 0), 0
    )                                            AS review_count
FROM {{ source('retail_raw', 'raw_reviews') }}
WHERE product_id IS NOT NULL
  AND TRIM(product_id::TEXT) != ''
