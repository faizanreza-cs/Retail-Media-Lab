-- Campaign reach and purchase rate
-- Shoppers reached, cost and purchase rate by campaign. Purchase rate is the share of reached
-- shoppers who bought in the window; it is not the campaign's incremental effect.
WITH reach AS (
    SELECT campaign, user_id, SUM(impressions) AS impressions, 0 AS clicks,
           SUM(total_cost) AS cost
    FROM dsp_impressions
    GROUP BY campaign, user_id
    UNION ALL
    SELECT campaign, user_id, SUM(impressions), SUM(clicks), SUM(spend)
    FROM sponsored_ads_traffic
    GROUP BY campaign, user_id
),
buyers AS (
    SELECT DISTINCT user_id FROM conversions
)
SELECT r.campaign,
       COUNT(*) AS users,
       SUM(r.impressions) AS impressions,
       SUM(r.clicks) AS clicks,
       ROUND(SUM(r.cost), 2) AS cost,
       COUNT(b.user_id) AS purchasers,
       ROUND(COUNT(b.user_id) / COUNT(*), 4) AS purchase_rate
FROM reach r
LEFT JOIN buyers b ON b.user_id = r.user_id
GROUP BY r.campaign
HAVING COUNT(*) >= 100
ORDER BY cost DESC
