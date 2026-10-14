"""Command line: ``rml build | query | sql | ask | optimize | export-powerbi | charts | load``."""

from __future__ import annotations

import argparse
import json
import sys

from rml.config import get_settings


def _print_table(rows: list[dict]) -> None:
    import pandas as pd

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(pd.DataFrame(rows).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rml", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate data, run every analysis, write results.json")
    b.add_argument("--search", type=int, default=500, help="MMM random-search size")
    q = sub.add_parser("query", help="run an AMC-style saved query (no name: list them)")
    q.add_argument("name", nargs="?")
    s = sub.add_parser("sql", help="run a guarded read-only Snowflake SQL query")
    s.add_argument("sql")
    a = sub.add_parser("ask", help="ask the Ads Copilot a question")
    a.add_argument("question")
    a.add_argument("--provider", default=None, help="anthropic | openai | azure | mock")
    o = sub.add_parser("optimize", help="recommend daily spend per channel")
    o.add_argument("--budget", type=float, default=None, help="total daily budget, USD")
    o.add_argument("--max-change", type=float, default=0.3)
    o.add_argument("--min-roas", type=float, default=None)
    e = sub.add_parser("export-powerbi", help="write the Power BI star schema as CSV files")
    e.add_argument("--out", default="powerbi/data")
    c = sub.add_parser("charts", help="render the README charts")
    c.add_argument("--out", default="docs/img")
    ld = sub.add_parser("load", help="copy the tables to Snowflake (SNOWFLAKE_* variables)")
    ld.add_argument("--backend", default="snowflake", choices=["snowflake"])
    args = p.parse_args(argv)

    if args.cmd == "build":
        from rml.pipeline import build

        res = build(n_search=args.search)
        print(json.dumps({"attribution_error": res["attribution"].get("share_error"),
                          "geo_test": [{k: r[k] for k in ("method", "iroas", "iroas_ci")}
                                       for r in res["incrementality"]["results"]],
                          "mmm_fit": res["mmm"]["fit"],
                          "budget": res["budget"].get("truth", res["budget"]["uplift_pct"])},
                         indent=2))  # fmt: skip
        print(f"wrote {get_settings().db_path} and results.json")
        return 0

    from rml.agent import tools

    if args.cmd == "query":
        out, err = tools.execute(
            "run_saved_query" if args.name else "list_saved_queries",
            {"name": args.name} if args.name else {},
        )
    elif args.cmd == "sql":
        out, err = tools.execute("run_sql", {"sql": args.sql})
    elif args.cmd == "optimize":
        out, err = tools.execute(
            "optimize_budget",
            {"total_daily_budget": args.budget, "max_change": args.max_change,
             "min_roas": args.min_roas},
        )  # fmt: skip
    elif args.cmd == "ask":
        from rml.agent.core import Agent
        from rml.agent.provider import get_provider

        r = Agent(get_provider(args.provider)).ask(args.question)
        print(r.answer)
        print(f"\n[{r.provider} | tools: {', '.join(r.tools_used) or 'none'} | {r.latency_ms} ms]")
        return 0
    elif args.cmd == "export-powerbi":
        from rml.powerbi import export

        for path in export(args.out):
            print(f"wrote {path}")
        return 0
    elif args.cmd == "charts":
        from rml.charts import render

        for path in render(args.out):
            print(f"wrote {path}")
        return 0
    else:  # load
        from rml.warehouse import TABLES, DuckDBWarehouse, SnowflakeWarehouse

        local, remote = DuckDBWarehouse(get_settings().db_path), SnowflakeWarehouse()
        for t in TABLES:
            n = remote.write(t, local.query(f"SELECT * FROM {t}"))
            print(f"{t}: {n:,} rows")
        return 0

    payload = json.loads(out)
    if err:
        print(payload["error"], file=sys.stderr)
        return 1
    _print_table(payload["rows"])
    if payload.get("note"):
        print(f"\n{payload['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
