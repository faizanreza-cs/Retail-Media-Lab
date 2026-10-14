"""Dataiku Python recipe: read the DSP geo holdout.

Inputs: daily_geo_sales, daily_geo_media, geo_test
Output: geo_test_results (one row per method: lift, lost sales, spend saved, iROAS)
"""

import dataiku
import pandas as pd

from rml.incrementality import analyse

result = analyse(
    dataiku.Dataset("daily_geo_sales").get_dataframe(),
    dataiku.Dataset("daily_geo_media").get_dataframe(),
    dataiku.Dataset("geo_test").get_dataframe(),
)
rows = [{k: v for k, v in r.items() if not isinstance(v, list | dict)} for r in result["results"]]
dataiku.Dataset("geo_test_results").write_with_schema(pd.DataFrame(rows))
