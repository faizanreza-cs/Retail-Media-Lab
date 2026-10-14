-- New-to-brand purchasers by ad exposure
-- Share of purchasers who had not bought the brand in the previous 12 months, by which ads
-- they saw. Upper-funnel ads should bring in more new customers.
WITH exposure AS (
    SELECT user_id,
           MAX(IFF(source = 'DSP', 1, 0)) AS saw_dsp,
           MAX(IFF(source = 'SPONSORED', 1, 0)) AS saw_sponsored
    FROM (
        SELECT user_id, 'DSP' AS source FROM dsp_impressions
        UNION ALL
        SELECT user_id, 'SPONSORED' FROM sponsored_ads_traffic
    )
    GROUP BY user_id
),
buyers AS (
    SELECT user_id,
           MAX(IFF(new_to_brand, 1, 0)) AS ntb,
           SUM(product_sales) AS sales
    FROM conversions
    GROUP BY user_id
)
SELECT CASE
           WHEN e.saw_dsp = 1 AND e.saw_sponsored = 1 THEN 'DSP + sponsored ads'
           WHEN e.saw_dsp = 1 THEN 'DSP only'
           WHEN e.saw_sponsored = 1 THEN 'Sponsored ads only'
           ELSE 'No ad exposure'
       END AS exposure,
       COUNT(*) AS users,
       SUM(b.ntb) AS new_to_brand_users,
       ROUND(SUM(b.ntb) / COUNT(*), 4) AS new_to_brand_rate,
       ROUND(SUM(b.sales), 2) AS sales
FROM buyers b
LEFT JOIN exposure e ON e.user_id = b.user_id
GROUP BY 1
HAVING COUNT(*) >= 100
ORDER BY users DESC
