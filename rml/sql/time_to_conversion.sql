-- Time to conversion
-- Days from a shopper's first ad to their first purchase, by the ad type that came first.
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
first_touch AS (
    SELECT user_id, ad_type AS first_ad_type, touch_ts
    FROM touches
    QUALIFY ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY touch_ts) = 1
),
first_purchase AS (
    SELECT user_id, MIN(conversion_event_dt_utc) AS purchase_ts
    FROM conversions
    GROUP BY user_id
),
lags AS (
    SELECT t.user_id, t.first_ad_type, DATEDIFF(day, t.touch_ts, p.purchase_ts) AS days
    FROM first_touch t
    JOIN first_purchase p ON p.user_id = t.user_id AND t.touch_ts < p.purchase_ts
)
SELECT first_ad_type,
       CASE
           WHEN days <= 7 THEN '00-07 days'
           WHEN days <= 14 THEN '08-14 days'
           WHEN days <= 21 THEN '15-21 days'
           ELSE '22+ days'
       END AS days_to_purchase,
       COUNT(*) AS users
FROM lags
GROUP BY 1, 2
HAVING COUNT(*) >= 100
ORDER BY 1, 2
