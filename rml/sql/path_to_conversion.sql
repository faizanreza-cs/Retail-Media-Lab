-- Path to conversion
-- The most common orders of ad types shoppers saw before their first purchase. Repeated
-- touches of the same ad type in a row count once (DSP > DSP > SP reads as DSP > SP).
WITH touches AS (
    SELECT user_id, impression_dt_utc AS touch_ts, 'DSP' AS ad_type
    FROM dsp_impressions
    UNION ALL
    SELECT user_id, event_dt_utc,
           CASE ad_product_type
               WHEN 'sponsored_products' THEN 'SP'
               WHEN 'sponsored_brands' THEN 'SB'
               ELSE 'SD'
           END
    FROM sponsored_ads_traffic
),
first_purchase AS (
    SELECT user_id, MIN(conversion_event_dt_utc) AS purchase_ts, SUM(product_sales) AS sales
    FROM conversions
    GROUP BY user_id
),
ordered AS (
    SELECT t.user_id, t.ad_type, t.touch_ts,
           LAG(t.ad_type) OVER (PARTITION BY t.user_id ORDER BY t.touch_ts) AS previous_type
    FROM touches t
    JOIN first_purchase p ON p.user_id = t.user_id AND t.touch_ts < p.purchase_ts
),
paths AS (
    SELECT user_id, LISTAGG(ad_type, ' > ') WITHIN GROUP (ORDER BY touch_ts) AS path
    FROM ordered
    WHERE previous_type IS NULL OR previous_type <> ad_type
    GROUP BY user_id
)
SELECT p.path,
       COUNT(DISTINCT p.user_id) AS users,
       ROUND(SUM(f.sales), 2) AS sales
FROM paths p
JOIN first_purchase f ON f.user_id = p.user_id
GROUP BY p.path
HAVING COUNT(DISTINCT p.user_id) >= 100
ORDER BY users DESC
