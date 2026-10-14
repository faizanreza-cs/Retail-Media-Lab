"""End to end: data -> warehouse -> every analysis -> ``results.json``.

``build`` generates the synthetic data first. ``analyse`` only reads tables through the
warehouse, so it runs the same way on DuckDB or on Snowflake once the tables are loaded there.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from rml import attribution, incrementality, mmm, optimize, synth
from rml.config import Settings, get_settings, results_path
from rml.synth import CHANNELS
from rml.warehouse import DuckDBWarehouse, Warehouse, reset_cache


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    return json.loads(out.to_json(orient="records", double_precision=6))


def analyse(wh: Warehouse, truth: dict | None = None, n_search: int = 500) -> dict[str, Any]:
    # --- attribution on shopper journeys
    touch = attribution.touches(
        wh.query("SELECT user_id, impression_dt_utc FROM dsp_impressions"),
        wh.query("SELECT user_id, event_dt_utc, ad_product_type FROM sponsored_ads_traffic"),
    )
    conv = wh.query("SELECT user_id, conversion_event_dt_utc FROM conversions")
    j = attribution.journeys(touch, conv)
    table = attribution.compare(j, truth["journeys"] if truth else None)
    attr: dict[str, Any] = {
        "table": _records(table),
        "exposed_shoppers": len(j),
        "exposed_buyers": int(j["converted"].sum()),
        "unexposed_buyers": attribution.unexposed_buyers(j, conv),
    }
    if truth:
        attr["share_error"] = attribution.share_error(table).to_dict()
        attr["true_incremental_purchasers"] = round(
            sum(truth["journeys"]["incremental_purchasers"].values()), 1
        )

    # --- DSP geo holdout
    sales = wh.query("SELECT * FROM daily_geo_sales")
    media = wh.query("SELECT * FROM daily_geo_media")
    geo_test = wh.query("SELECT * FROM geo_test")
    inc = incrementality.analyse(sales, media, geo_test, truth["geo_test"] if truth else None)
    inc["weekly"] = _records(inc.pop("series"))

    # --- marketing mix model; "current" = the 13 weeks before the geo test
    df = mmm.national_daily(sales, media, wh.query("SELECT * FROM calendar"))
    test_start = pd.Timestamp(inc["start"])
    before = df.index[df["date"] < test_start]
    model = mmm.fit(df, n_search=n_search, bau_days=slice(before[-91], before[-1] + 1))
    channels = model.channels.copy()
    if truth:
        channels["true_roas"] = [truth["media"][c]["roas"] for c in channels["channel"]]
        channels["true_mroas"] = [truth["media"][c]["mroas_at_bau"] for c in channels["channel"]]
    weekly = (
        model.decomposition.set_index("date")
        .resample("W-MON", label="left", closed="left")  # weeks starting Monday
        .sum()
    )
    mix = {
        "fit": model.fit,
        "channels": _records(channels),
        "curves": model.curves.to_dict(),
        "weekly": _records(weekly.reset_index()),
    }

    # --- budget plan at the current total
    current = {r["channel"]: r["bau_daily_spend"] for r in model.channels.to_dict("records")}
    plan = optimize.optimize(model.curves, current)
    if truth:
        plan["truth"] = optimize.score_with_truth(
            plan, optimize.TrueCurves(truth["media"], truth["geo_weights"])
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": wh.name,
        "channels": list(CHANNELS),
        "attribution": attr,
        "incrementality": inc,
        "mmm": mix,
        "budget": plan,
        "truth": truth,
    }


def build(
    settings: Settings | None = None, cfg: synth.SynthConfig | None = None, n_search: int = 500
) -> dict[str, Any]:
    """Generate the data, write the DuckDB file, run every analysis, save results.json."""
    s = settings or get_settings()
    data = synth.generate(cfg)
    synth.write(data, s.db_path)
    reset_cache()
    results = analyse(DuckDBWarehouse(s.db_path), data.truth, n_search=n_search)
    path = results_path(s.db_path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=1, default=str))
    tmp.replace(path)
    _load.cache_clear()
    return results


@lru_cache(maxsize=2)
def _load(path: str, mtime: float) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text()))


def load_results(settings: Settings | None = None) -> dict[str, Any]:
    s = settings or get_settings()
    path = results_path(s.db_path)
    if not path.exists():
        raise FileNotFoundError(f"No results at {path}. Run `rml build` first.")
    return _load(str(path), path.stat().st_mtime)


def curves(results: dict[str, Any]) -> mmm.Curves:
    return mmm.Curves.from_dict(results["mmm"]["curves"])
