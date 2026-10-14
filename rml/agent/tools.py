"""Tools the Ads Copilot agent can call. Every number in an answer comes from one of these.

Analyses read ``results.json`` (written by ``rml build``), so answers are fast and consistent;
the budget optimizer re-runs on the saved response curves with the requested constraints, and
SQL runs live against the warehouse (DuckDB or Snowflake).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pandas as pd

from rml import optimize, pipeline
from rml.agent.sql_guard import UnsafeSQLError, check_result, validate
from rml.config import get_settings
from rml.queries import MIN_USERS, run_saved, saved_queries
from rml.synth import CHANNEL_NAMES, CHANNELS
from rml.warehouse import get_warehouse


def _ok(rows: list[dict[str, Any]] | pd.DataFrame, note: str = "", **extra: Any) -> dict:
    if isinstance(rows, pd.DataFrame):
        rows = json.loads(rows.to_json(orient="records", date_format="iso", double_precision=4))
    return {"rows": rows, "row_count": len(rows), "note": note, **extra}


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# --------------------------------------------------------------------------- tools


def t_list_saved_queries() -> dict:
    rows = [
        {"name": q.name, "title": q.title, "description": q.description}
        for q in saved_queries().values()
    ]
    return _ok(rows, note=f"AMC-style saved queries; outputs keep rows with >= {MIN_USERS} users.")


def t_run_saved_query(name: str) -> dict:
    q = saved_queries().get(name)
    if q is None:
        raise ValueError(f"unknown query {name!r}; choose from {', '.join(saved_queries())}")
    df = run_saved(name, get_warehouse())
    return _ok(df, note=f"{q.title}: {q.description}", query=name)


def t_run_sql(sql: str) -> dict:
    safe = validate(sql, get_settings().sql_row_limit)
    df = get_warehouse().query(safe)
    check_result(df)
    return _ok(df, note="Read-only query, row-capped.", sql=safe)


def t_compare_attribution() -> dict:
    a = pipeline.load_results()["attribution"]
    wide = pd.DataFrame(a["table"]).pivot(index="channel", columns="method", values="share")
    wide = wide.reindex(list(CHANNELS)).reset_index()
    errors = a.get("share_error", {})
    note = (
        f"Share of credited purchases by method ({a['exposed_buyers']:,} purchases by shoppers "
        "who saw an ad). Attribution shows who saw ads before buying; it can't tell whether the "
        "ad caused the purchase."
    )
    if "true_incremental_purchasers" in a:
        note += (
            f" The simulator's truth: ads caused {a['true_incremental_purchasers']:,.0f} purchases, "
            f"and the 'truth' column is their causal split. Total share error vs truth: "
            + ", ".join(f"{m} {v:.4f}" for m, v in sorted(errors.items(), key=lambda kv: kv[1]))
            + "."
        )
    return _ok(wide, note=note)


def t_geo_test_results() -> dict:
    inc = pipeline.load_results()["incrementality"]
    rows = [
        {
            "method": r["method"],
            "sales_change_pct": r["lift_pct"],
            "lost_sales": r["lost_sales"],
            "lost_sales_low": r["lost_sales_ci"][0],
            "lost_sales_high": r["lost_sales_ci"][1],
            "iroas": r["iroas"],
            "iroas_low": r["iroas_ci"][0],
            "iroas_high": r["iroas_ci"][1],
            "p_value": r["p_value"],
        }
        for r in inc["results"]
    ]
    saved = inc["results"][0]["spend_saved"]
    if "truth" in inc:
        t = inc["truth"]
        rows.append(
            {"method": "simulator truth", "lost_sales": t["lost_sales"], "iroas": t["iroas"]}
        )
    note = (
        f"DSP was switched off in {len(inc['test_geos'])} test geos ({', '.join(inc['test_geos'])}) "
        f"from {inc['start']} to {inc['end']}; control geos kept it. Spend saved: ${saved:,.0f}. "
        "iROAS = sales lost / spend saved. Intervals: 95% (DiD, clustered by geo; synthetic "
        "control, placebo tests)."
    )
    return _ok(rows, note=note)


def t_channel_roas() -> dict:
    m = pipeline.load_results()["mmm"]
    keep = ["channel", "roas", "roas_low", "roas_high", "mroas_at_bau", "mroas_low",
            "mroas_high", "bau_daily_spend", "true_roas", "true_mroas"]  # fmt: skip
    rows = [{k: r[k] for k in keep if k in r} for r in m["channels"]]
    for r in rows:
        r["channel_name"] = CHANNEL_NAMES[r["channel"]]
    f = m["fit"]
    note = (
        f"Marketing mix model, ensemble of the {f['models']} best fits (R2 {f['r2']}, MAPE "
        f"{_pct(f['mape'])}, cross-validated MAPE {_pct(f['cv_mape'])}). ROAS = incremental sales "
        "per $1 over two years; mROAS = sales from the next $1 at current spend. Intervals are 90% "
        "and combine model and noise uncertainty: a wide one means the data can't pin it down."
    )
    return _ok(rows, note=note)


def t_optimize_budget(
    total_daily_budget: float | None = None, max_change: float = 0.3, min_roas: float | None = None
) -> dict:
    res = pipeline.load_results()
    current = {r["channel"]: r["bau_daily_spend"] for r in res["mmm"]["channels"]}
    plan = optimize.optimize(
        pipeline.curves(res), current, total_daily_budget, max_change, min_roas
    )
    extra: dict[str, Any] = {k: v for k, v in plan.items() if k != "channels"}
    note = (
        f"Daily budget ${plan['total_daily_budget']:,.0f}; each channel may move up to "
        f"{_pct(plan['max_change'])} from its current share. Model-expected incremental sales "
        f"{_pct(plan['uplift_pct'])} vs the current plan (${plan['current_daily_incremental_sales']:,.0f}"
        f" -> ${plan['recommended_daily_incremental_sales']:,.0f} per day)."
    )
    if res.get("truth"):
        t = res["truth"]
        score = optimize.score_with_truth(plan, optimize.TrueCurves(t["media"], t["geo_weights"]))
        extra["truth"] = score
        note += (
            f" Checked against the simulator's true curves: {_pct(score['uplift_pct'])} "
            f"(best possible {_pct(score['best_possible_uplift_pct'])})."
        )
    return _ok(plan["channels"], note=note, **extra)


# --------------------------------------------------------------------------- registry

_CHANNEL_HELP = "Channels: SP (Sponsored Products), SB (Sponsored Brands), SD (Sponsored Display), DSP (Amazon DSP)."

TOOLS: dict[str, tuple[Callable[..., dict], dict[str, Any]]] = {
    "list_saved_queries": (t_list_saved_queries, {
        "description": "List the AMC-style saved queries (name, title, description).",
        "input_schema": {"type": "object", "properties": {}},
    }),
    "run_saved_query": (t_run_saved_query, {
        "description": "Run an AMC-style saved query on shopper journeys: path_to_conversion (common ad paths before purchase), new_to_brand (new customers by exposure), frequency_vs_conversion (DSP impression frequency vs purchase rate), ad_type_overlap (reach by ad type combination), time_to_conversion (days from first ad to purchase), campaign_reach (users, cost and purchase rate by campaign). Rows cover at least 100 users.",
        "input_schema": {"type": "object", "properties": {
            "name": {"type": "string", "enum": sorted(saved_queries())}},
            "required": ["name"]},
    }),
    "run_sql": (t_run_sql, {
        "description": "Run one read-only Snowflake SQL SELECT when no other tool fits. Tables: dsp_impressions(user_id, impression_dt_utc, campaign_id, campaign, impressions, total_cost), sponsored_ads_traffic(user_id, event_dt_utc, ad_product_type, campaign_id, campaign, impressions, clicks, spend), conversions(user_id, conversion_event_dt_utc, tracked_asin, purchases, product_sales, new_to_brand), asins(tracked_asin, product_name, category, list_price), daily_geo_media(date, geo_id, channel, spend, impressions), daily_geo_sales(date, geo_id, sales, orders), geos(geo_id, geo_name, region, population_share), geo_test(geo_id, test_group, channel, test_start, test_end), calendar(date, event, new_year, coupon). Never return user_id: aggregate (COUNT(DISTINCT user_id) is fine). Results are capped at 200 rows.",
        "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}},
                         "required": ["sql"]},
    }),
    "compare_attribution": (t_compare_attribution, {
        "description": f"Multi-touch attribution: share of purchases credited to each channel by last touch, first touch, linear, Markov removal effect and Shapley, next to the true causal split from the simulator. {_CHANNEL_HELP}",
        "input_schema": {"type": "object", "properties": {}},
    }),
    "geo_test_results": (t_geo_test_results, {
        "description": "Results of the DSP geo holdout experiment: sales lost when DSP was switched off in test geos, spend saved and incremental ROAS (iROAS), by difference-in-differences and synthetic control, with intervals.",
        "input_schema": {"type": "object", "properties": {}},
    }),
    "channel_roas": (t_channel_roas, {
        "description": f"Marketing mix model results per channel: average ROAS and marginal ROAS at current spend (mROAS) with 90% intervals, and current daily spend. Use for ROAS, saturation and diminishing-returns questions. {_CHANNEL_HELP}",
        "input_schema": {"type": "object", "properties": {}},
    }),
    "optimize_budget": (t_optimize_budget, {
        "description": "Recommend daily spend per channel that maximises incremental sales on the MMM response curves. Defaults: current total budget, each channel moves at most 30%.",
        "input_schema": {"type": "object", "properties": {
            "total_daily_budget": {"type": "number", "description": "USD per day; default: current total"},
            "max_change": {"type": "number", "minimum": 0, "maximum": 1, "description": "max relative change per channel (0.3 = 30%)"},
            "min_roas": {"type": "number", "description": "optional floor on the plan's ROAS"}}},
    }),
}  # fmt: skip


def tool_definitions() -> list[dict[str, Any]]:
    return [{"name": name, **spec} for name, (_, spec) in TOOLS.items()]


def execute(name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Run a tool; returns (json_string, is_error). Errors go back to the model, not up."""
    if name not in TOOLS:
        return json.dumps({"error": f"unknown tool {name!r}"}), True
    fn, _ = TOOLS[name]
    try:
        return json.dumps(fn(**args), default=str), False
    except (UnsafeSQLError, ValueError, TypeError, KeyError, FileNotFoundError) as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
    except Exception as e:  # warehouse errors and the like: tell the model, keep the run alive
        return json.dumps({"error": f"{type(e).__name__}: {e}"}), True
