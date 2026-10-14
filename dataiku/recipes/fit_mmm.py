"""Dataiku Python recipe: fit the marketing mix model.

Inputs: daily_geo_sales, daily_geo_media, calendar
Outputs: channel_roas (ROAS and marginal ROAS with intervals), mmm_curves (ensemble members)
"""

import json

import dataiku
import pandas as pd

from rml.mmm import fit, national_daily

df = national_daily(
    dataiku.Dataset("daily_geo_sales").get_dataframe(),
    dataiku.Dataset("daily_geo_media").get_dataframe(),
    dataiku.Dataset("calendar").get_dataframe(),
)
model = fit(df)
dataiku.Dataset("channel_roas").write_with_schema(model.channels)
members = [{"member": i, "params": json.dumps(m)} for i, m in enumerate(model.curves.to_dict())]
dataiku.Dataset("mmm_curves").write_with_schema(pd.DataFrame(members))
