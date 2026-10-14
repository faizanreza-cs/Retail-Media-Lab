-- Ad type overlap
-- How many shoppers each combination of ad types reached, and how often they purchased.
WITH reach AS (
    SELECT user_id, 'DSP' AS ad_type FROM dsp_impressions
    UNION
    SELECT user_id,
           CASE ad_product_type
               WHEN 'sponsored_products' THEN 'SP'
               WHEN 'sponsored_brands' THEN 'SB'
               ELSE 'SD'
           END
    FROM sponsored_ads_traffic
),
combos AS (
    SELECT user_id, LISTAGG(ad_type, ' + ') WITHIN GROUP (ORDER BY ad_type) AS ad_types
    FROM reach
    GROUP BY user_id
),
buyers AS (
    SELECT DISTINCT user_id FROM conversions
)
SELECT c.ad_types,
       COUNT(*) AS users,
       COUNT(b.user_id) AS purchasers,
       ROUND(COUNT(b.user_id) / COUNT(*), 4) AS purchase_rate
FROM combos c
LEFT JOIN buyers b ON b.user_id = c.user_id
GROUP BY c.ad_types
HAVING COUNT(*) >= 100
ORDER BY users DESC
