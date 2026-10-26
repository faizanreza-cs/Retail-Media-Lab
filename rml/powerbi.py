"""Power BI kit: export a star schema as CSV files for Power BI Desktop.

``rml export-powerbi`` writes ``powerbi/data/*.csv``. Load them with Get data > Text/CSV, create
the relationships listed in ``powerbi/README.md``, paste the measures from
``powerbi/measures.dax`` and apply ``powerbi/theme.json``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rml import pipeline
from rml.synth import CHANNEL_NAMES, CHANNELS, FUNNEL_STAGE
from rml.warehouse import get_warehouse


def tables() -> dict[str, pd.DataFrame]:
    wh = get_warehouse()
    res = pipeline.load_results()
    media = wh.query("SELECT date, geo_id, channel, spend, impressions FROM daily_geo_media")
    sales = wh.query("SELECT date, geo_id, sales, orders FROM daily_geo_sales")
    cal = wh.query("SELECT date, event FROM calendar")
    geos = wh.query("SELECT g.geo_id, g.geo_name, g.region, g.population_share, t.test_group "
                    "FROM geos g JOIN geo_test t ON t.geo_id = g.geo_id")  # fmt: skip
    geo = res["incrementality"]

    d = pd.to_datetime(cal["date"])
    dim_date = pd.DataFrame(
        {
            "date": d.dt.date,
            "week_start": (d - pd.to_timedelta(d.dt.day_of_week, unit="D")).dt.date,
            "month": d.dt.strftime("%Y-%m"),
            "quarter": d.dt.year.astype(str) + "-Q" + d.dt.quarter.astype(str),
            "year": d.dt.year,
            "weekday": d.dt.day_name(),
            "event": cal["event"].replace("", "None"),
            "is_geo_test_period": (d >= pd.Timestamp(geo["start"]))
            & (d <= pd.Timestamp(geo["end"])),
        }
    )
    dim_channel = pd.DataFrame(
        {
            "channel": list(CHANNELS),
            "channel_name": [CHANNEL_NAMES[c] for c in CHANNELS],
            "funnel_stage": [FUNNEL_STAGE[c] for c in CHANNELS],
            "sort_order": range(1, len(CHANNELS) + 1),
        }
    )
    curves = pipeline.curves(res)
    plan = pd.DataFrame(res["budget"]["channels"])
    grid = []
    for c in CHANNELS:
        cur = float(plan.loc[plan["channel"] == c, "current_daily_spend"].iloc[0])
        xs = np.linspace(0, 2 * cur, 41)
        inc = curves.response(c, xs).round(2)
        grid.append(
            pd.DataFrame({"channel": c, "daily_spend": xs.round(2), "incremental_sales": inc})
        )
    weekly = pd.DataFrame(res["mmm"]["weekly"])
    contribution = (
        weekly[["date", "baseline", *CHANNELS]]
        .melt(id_vars="date", var_name="component", value_name="contribution")
        .rename(columns={"date": "week_start"})
    )
    geo_weekly = pd.DataFrame(geo["weekly"]).rename(
        columns={"week": "week_start", "actual": "holdout_actual", "synthetic": "synthetic_control"}
    )
    geo_weekly["in_test"] = pd.to_datetime(geo_weekly["week_start"]) >= pd.Timestamp(geo["start"])
    return {
        "dim_date": dim_date,
        "dim_geo": geos,
        "dim_channel": dim_channel,
        "fact_media_daily": media.assign(date=pd.to_datetime(media["date"]).dt.date),
        "fact_sales_daily": sales.assign(date=pd.to_datetime(sales["date"]).dt.date),
        "fact_attribution": pd.DataFrame(res["attribution"]["table"]),
        "fact_channel_roas": pd.DataFrame(res["mmm"]["channels"]),
        "fact_response_curve": pd.concat(grid, ignore_index=True),
        "fact_budget_plan": plan,
        "fact_mmm_contribution": contribution,
        "fact_geo_test_weekly": geo_weekly,
    }


def export(out: str | Path = "powerbi/data") -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in tables().items():
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths
