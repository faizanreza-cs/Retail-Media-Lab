"""Dataiku Python recipe: recommend daily spend per channel.

Inputs: channel_roas, mmm_curves
Output: budget_plan (current vs recommended daily spend per channel)
Project variables (optional): max_change (default 0.3), total_daily_budget
"""

import json

import dataiku
import pandas as pd

from rml.mmm import Curves
from rml.optimize import optimize

variables = dataiku.get_custom_variables()
roas = dataiku.Dataset("channel_roas").get_dataframe()
members = dataiku.Dataset("mmm_curves").get_dataframe()
curves = Curves.from_dict([json.loads(p) for p in members["params"]])

current = dict(zip(roas["channel"], roas["bau_daily_spend"], strict=True))
total = variables.get("total_daily_budget")
plan = optimize(
    curves,
    current,
    total=float(total) if total else None,
    max_change=float(variables.get("max_change", 0.3)),
)
dataiku.Dataset("budget_plan").write_with_schema(pd.DataFrame(plan["channels"]))
