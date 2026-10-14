-- DSP frequency vs purchase rate
-- Purchase rate of shoppers by how many DSP impressions they saw. Shows where extra frequency
-- stops paying off. Observational: heavier-exposed shoppers can differ in other ways.
WITH frequency AS (
    SELECT user_id, SUM(impressions) AS impressions
    FROM dsp_impressions
    GROUP BY user_id
),
buyers AS (
    SELECT DISTINCT user_id FROM conversions
)
SELECT CASE
           WHEN f.impressions = 1 THEN '01'
           WHEN f.impressions <= 3 THEN '02-03'
           WHEN f.impressions <= 6 THEN '04-06'
           WHEN f.impressions <= 10 THEN '07-10'
           ELSE '11+'
       END AS frequency,
       COUNT(*) AS users,
       COUNT(b.user_id) AS purchasers,
       ROUND(COUNT(b.user_id) / COUNT(*), 4) AS purchase_rate
FROM frequency f
LEFT JOIN buyers b ON b.user_id = f.user_id
GROUP BY 1
HAVING COUNT(*) >= 100
ORDER BY 1
