"""Evaluate the Ads Copilot on golden questions.

Metrics
-------
* **tool_accuracy**   - the agent called at least one of the expected tools
* **answer_accuracy** - every ground-truth fact (number or label) appears in the answer
* **tool_error_rate** - questions where a tool returned an error
* latency and token usage per question

Run ``python -m rml.evals.run_evals`` (mock provider in CI) or ``--provider anthropic`` (or
openai / azure) to measure a real model. Writes ``reports/eval_results_<provider>.json`` and
``reports/eval_summary_<provider>.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from rml import optimize, pipeline
from rml.agent.core import Agent
from rml.agent.provider import get_provider
from rml.queries import run_saved, saved_queries
from rml.warehouse import get_warehouse

GOLDEN = Path(__file__).with_name("golden.yaml")
Fact = float | int | str


# --------------------------------------------------------------------------- ground truth


def _saved(name: str) -> pd.DataFrame:
    return run_saved(name, get_warehouse())


def _query_names() -> list[Fact]:
    names: list[Fact] = [n for n in saved_queries()]
    return names[:3]


def _plan(**kw: Any) -> dict:
    res = pipeline.load_results()
    current = {r["channel"]: r["bau_daily_spend"] for r in res["mmm"]["channels"]}
    return optimize.optimize(pipeline.curves(res), current, **kw)


def _rec(plan: dict, channel: str) -> float:
    return next(r["recommended_daily_spend"] for r in plan["channels"] if r["channel"] == channel)


def _mmm(channel: str, key: str) -> float:
    rows = pipeline.load_results()["mmm"]["channels"]
    return float(next(r[key] for r in rows if r["channel"] == channel))


def _attr() -> dict:
    return pipeline.load_results()["attribution"]


def _share(method: str, channel: str) -> float:
    return next(
        r["share"] for r in _attr()["table"] if r["method"] == method and r["channel"] == channel
    )


def _geo(method: str, key: str) -> float:
    rows = pipeline.load_results()["incrementality"]["results"]
    return float(next(r[key] for r in rows if r["method"] == method))


def _sql(sql: str, col: str) -> Fact:
    v = get_warehouse().query(sql)[col].iloc[0]
    return v if isinstance(v, str) else float(v)


TRUTH: dict[str, Callable[[], list[Fact]]] = {
    "top_path": lambda: [str(_saved("path_to_conversion")["path"].iloc[0]),
                         int(_saved("path_to_conversion")["users"].iloc[0])],
    "ntb_dsp_only": lambda: [float(_saved("new_to_brand").set_index("exposure")
                                   .loc["DSP only", "new_to_brand_rate"])],
    "frequency_top_bucket": lambda: [float(_saved("frequency_vs_conversion")["purchase_rate"]
                                           .iloc[-1])],
    "overlap_dsp_sp": lambda: [int(_saved("ad_type_overlap").set_index("ad_types")
                                   .loc["DSP + SP", "users"])],
    "time_dsp_late": lambda: [int(_saved("time_to_conversion").set_index(
        ["first_ad_type", "days_to_purchase"]).loc[("DSP", "22+ days"), "users"])],
    "top_campaign": lambda: [str(_saved("campaign_reach")["campaign"].iloc[0]),
                             float(_saved("campaign_reach")["purchase_rate"].iloc[0])],
    "last_touch_sp": lambda: [_share("last_touch", "SP")],
    "attribution_errors": lambda: [float(_attr()["share_error"]["shapley"])],
    "attribution_totals": lambda: [int(_attr()["exposed_buyers"]),
                                   float(_attr()["true_incremental_purchasers"])],
    "geo_iroas": lambda: [_geo("difference_in_differences", "iroas"),
                          _geo("synthetic_control", "iroas")],
    "geo_lost_sales": lambda: [_geo("difference_in_differences", "lost_sales")],
    "mmm_roas": lambda: [_mmm("DSP", "roas"), _mmm("SB", "roas")],
    "mmm_mroas_dsp": lambda: ["DSP", _mmm("DSP", "mroas_at_bau")],
    "mmm_mroas_sp": lambda: [_mmm("SP", "mroas_at_bau")],
    "plan_default": lambda: [_rec(_plan(), "DSP")],
    "plan_20k": lambda: [_rec(_plan(total=20000.0), "DSP")],
    "plan_10pct": lambda: [_rec(_plan(max_change=0.1), "DSP")],
    "sql_purchasers": lambda: [_sql("SELECT COUNT(DISTINCT user_id) AS n FROM conversions", "n")],
    "sql_spend_sp": lambda: [_sql("SELECT ROUND(SUM(spend), 0) AS s FROM daily_geo_media "
                                  "WHERE channel = 'SP'", "s")],
    "sql_top_product": lambda: [_sql(
        "SELECT a.product_name AS p FROM conversions c JOIN asins a "
        "ON a.tracked_asin = c.tracked_asin GROUP BY 1 ORDER BY SUM(c.product_sales) DESC "
        "LIMIT 1", "p")],
    "saved_query_names": _query_names,
}  # fmt: skip


# --------------------------------------------------------------------------- grading

_NUM = re.compile(r"(-?\d[\d,]*\.?\d*)\s*([kKmM%]?)")


def extract_numbers(text: str) -> list[float]:
    out = []
    for raw, suffix in _NUM.findall(text):
        try:
            n = float(raw.replace(",", ""))
        except ValueError:
            continue
        if suffix in "kK" and suffix:
            n *= 1_000
        elif suffix in "mM" and suffix:
            n *= 1_000_000
        out.append(n)
    return out


def fact_present(fact: Fact, answer: str) -> bool:
    if isinstance(fact, str):
        return fact.lower() in answer.lower()
    nums = extract_numbers(answer)
    v = float(fact)
    if abs(v) <= 1.5 and not float(v).is_integer():  # a rate / share
        return any(abs(n - v) <= 0.0015 or abs(n - 100 * v) <= 0.15 for n in nums)
    tol = max(abs(v) * 0.005, 0.6)  # counts and volumes: 0.5% (allows "62.7K")
    return any(abs(n - v) <= tol for n in nums)


@dataclass
class EvalRow:
    id: str
    question: str
    expected_tools: list[str]
    tools_used: list[str]
    tool_ok: bool
    facts: list[Fact]
    facts_found: list[bool]
    answer_ok: bool
    tool_error: bool
    latency_ms: int
    input_tokens: int
    output_tokens: int
    answer: str


def run(provider_name: str = "mock", only: list[str] | None = None) -> list[EvalRow]:
    cases = yaml.safe_load(GOLDEN.read_text())
    if only:
        cases = [c for c in cases if c["id"] in only]
    agent = Agent(get_provider(provider_name))
    rows = []
    for case in cases:
        res = agent.ask(case["question"])
        facts = TRUTH[case["truth"]]()
        found = [fact_present(f, res.answer) for f in facts]
        rows.append(
            EvalRow(
                id=case["id"],
                question=case["question"],
                expected_tools=case["expected_tools"],
                tools_used=res.tools_used,
                tool_ok=bool(set(case["expected_tools"]) & set(res.tools_used)),
                facts=facts,
                facts_found=found,
                answer_ok=all(found),
                tool_error=any(s.is_error for s in res.steps),
                latency_ms=res.latency_ms,
                input_tokens=res.usage.get("input_tokens", 0),
                output_tokens=res.usage.get("output_tokens", 0),
                answer=res.answer,
            )
        )
        mark = "PASS" if rows[-1].tool_ok and rows[-1].answer_ok else "FAIL"
        print(f"[{mark}] {case['id']:<20} tools={res.tools_used} facts={found}", flush=True)
    return rows


def summarise(rows: list[EvalRow], provider: str) -> dict[str, Any]:
    n = len(rows)
    return {
        "provider": provider,
        "questions": n,
        "tool_accuracy": round(sum(r.tool_ok for r in rows) / n, 4),
        "answer_accuracy": round(sum(r.answer_ok for r in rows) / n, 4),
        "pass_rate": round(sum(r.tool_ok and r.answer_ok for r in rows) / n, 4),
        "tool_error_rate": round(sum(r.tool_error for r in rows) / n, 4),
        "median_latency_ms": int(statistics.median(r.latency_ms for r in rows)),
        "total_input_tokens": sum(r.input_tokens for r in rows),
        "total_output_tokens": sum(r.output_tokens for r in rows),
    }


def write_reports(rows: list[EvalRow], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = summary["provider"]
    (out_dir / f"eval_results_{tag}.json").write_text(
        json.dumps({"summary": summary, "rows": [r.__dict__ for r in rows]}, indent=2, default=str)
    )
    lines = [
        f"# Eval summary ({tag})",
        "",
        "| metric | value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in summary.items()],
        "",
        "| id | tool ok | answer ok | tools used |",
        "|---|---|---|---|",
        *[
            f"| {r.id} | {'✅' if r.tool_ok else '❌'} | {'✅' if r.answer_ok else '❌'} "
            f"| {', '.join(r.tools_used)} |"
            for r in rows
        ],
    ]
    (out_dir / f"eval_summary_{tag}.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--provider", default="mock", help="mock | anthropic | openai | azure")
    p.add_argument("--only", nargs="*", help="run only these case ids")
    p.add_argument("--out", default="reports")
    p.add_argument("--min-pass-rate", type=float, default=0.0, help="exit 1 below this")
    a = p.parse_args(argv)
    rows = run(a.provider, a.only)
    summary = summarise(rows, a.provider)
    write_reports(rows, summary, Path(a.out))
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass_rate"] >= a.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
