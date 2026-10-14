"""Dataiku Python recipe: fit the marketing mix model.

Inputs: daily_geo_sales, daily_geo_media, calendar, geo_test
Outputs: channel_roas (ROAS and marginal ROAS with intervals), mmm_curves (ensemble members)

"Current" spend is the 91 days before the geo test, as in ``rml build``, so the budget plan
matches the API's.
"""

import json

import dataiku
import pandas as pd

from rml.mmm import days_before, fit, national_daily

df = national_daily(
    dataiku.Dataset("daily_geo_sales").get_dataframe(),
    dataiku.Dataset("daily_geo_media").get_dataframe(),
    dataiku.Dataset("calendar").get_dataframe(),
)
test_start = pd.Timestamp(dataiku.Dataset("geo_test").get_dataframe()["test_start"].iloc[0])
model = fit(df, bau_days=days_before(df, test_start))
dataiku.Dataset("channel_roas").write_with_schema(model.channels)
members = [{"member": i, "params": json.dumps(m)} for i, m in enumerate(model.curves.to_dict())]
dataiku.Dataset("mmm_curves").write_with_schema(pd.DataFrame(members))
