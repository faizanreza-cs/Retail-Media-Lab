"""Dataiku Python recipe: multi-touch attribution.

Inputs: dsp_impressions, sponsored_ads_traffic, conversions
Output: attribution (channel, method, credited_purchasers, share)
"""

import dataiku

from rml.attribution import compare, journeys, touches

dsp = dataiku.Dataset("dsp_impressions").get_dataframe(columns=["user_id", "impression_dt_utc"])
sponsored = dataiku.Dataset("sponsored_ads_traffic").get_dataframe(
    columns=["user_id", "event_dt_utc", "ad_product_type"]
)
conversions = dataiku.Dataset("conversions").get_dataframe(
    columns=["user_id", "conversion_event_dt_utc"]
)

paths = journeys(touches(dsp, sponsored), conversions)
dataiku.Dataset("attribution").write_with_schema(compare(paths))
