"""Guardrails for model-written SQL (Snowflake dialect).

The agent may run ad-hoc SQL, but only:

* one read-only SELECT (CTEs and UNION allowed),
* against the allow-listed tables,
* with no table functions, LATERAL, or file / network / session functions,
* never returning shopper ids, in the spirit of AMC, which only releases aggregates: the output
  may not list ``user_id`` outside an aggregate, and ``check_result`` rejects any result whose
  values contain a shopper id (which also catches aliases, CTEs and string tricks),
* with a row cap.

This is the first line of defence. The engine is the second: DuckDB runs read-only with
external access disabled, and on Snowflake the agent should use the read-only role in
``snowflake/setup.sql``.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
import sqlglot
from sqlglot import exp

from rml.warehouse import TABLES

ALLOWED_TABLES = frozenset(TABLES)
SHOPPER_TABLES = frozenset({"dsp_impressions", "sponsored_ads_traffic", "conversions"})
BLOCKED_FUNCTIONS = frozenset(
    {
        "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "read_json",
        "read_json_auto", "read_json_objects", "read_ndjson", "read_text", "read_blob", "glob",
        "sniff_csv", "query", "query_table", "getenv", "json_execute_serialized_sql",
        "result_scan", "infer_schema", "get_ddl", "system$whitelist", "flatten",
    }
)  # fmt: skip


USER_ID = re.compile(r"u\d{7}")  # format of the synthetic shopper ids


class UnsafeSQLError(ValueError):
    pass


def _outputs(node: Any) -> list[exp.Expression]:
    if isinstance(node, exp.Union | exp.Intersect | exp.Except):
        return _outputs(node.left) + _outputs(node.right)
    if isinstance(node, exp.Subquery):
        return _outputs(node.this)
    return list(node.expressions) if isinstance(node, exp.Select) else []


def validate(sql: str, row_limit: int = 200) -> str:
    """Return a safe, row-capped version of ``sql`` (Snowflake dialect) or raise."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="snowflake") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"could not parse SQL: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQLError("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Select | exp.Union | exp.Intersect | exp.Except):
        raise UnsafeSQLError(f"only SELECT queries are allowed (got {tree.key.upper()})")
    for node in tree.walk():
        if isinstance(node, exp.Insert | exp.Update | exp.Delete | exp.Drop | exp.Create
                      | exp.Alter | exp.Command | exp.Copy | exp.Merge):  # fmt: skip
            raise UnsafeSQLError("write and DDL statements are not allowed")
        if isinstance(node, exp.Lateral):
            raise UnsafeSQLError("LATERAL is not allowed")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    used = set()
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if isinstance(table.this, exp.Func) or not name:
            raise UnsafeSQLError("table functions are not allowed")
        if table.args.get("db") or table.args.get("catalog"):
            raise UnsafeSQLError("schema-qualified table names are not allowed")
        if name not in ALLOWED_TABLES and name not in cte_names:
            raise UnsafeSQLError(f"table {table.name!r} is not allowed")
        used.add(name)

    for func in tree.find_all(exp.Func):
        fname = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
        if fname in BLOCKED_FUNCTIONS:
            raise UnsafeSQLError(f"function {fname!r} is not allowed")

    if used & SHOPPER_TABLES:
        for out in _outputs(tree):
            for col in out.find_all(exp.Column):
                if col.name.lower() == "user_id" and not col.find_ancestor(exp.AggFunc):
                    raise UnsafeSQLError("queries may not return user_id; aggregate instead")

    limit = tree.args.get("limit")
    options = limit.args.get("limit_options") if limit is not None else None
    plain = isinstance(tree, exp.Select) and (
        limit is None
        or (
            isinstance(limit, exp.Limit)
            and isinstance(limit.expression, exp.Literal)
            and not limit.expression.is_string
            and not (options and options.args.get("percent"))
        )
    )
    if plain:
        n = int(limit.expression.name) if limit is not None else row_limit
        # capped in place, so a top-level ORDER BY keeps working on every engine
        return tree.limit(min(n, row_limit)).sql(dialect="snowflake")
    # PERCENT, TOP/FETCH variants, set operations: wrap with an outer cap.
    # shortcut: Snowflake may not keep an inner ORDER BY through the wrapper; order rows in the
    # caller if a set-operation query needs ordering.
    return exp.select("*").from_(tree.subquery("q")).limit(row_limit).sql(dialect="snowflake")


def check_result(df: pd.DataFrame) -> None:
    """Reject results that contain shopper ids, however the query produced them."""
    for col in df.columns:
        s = df[col]
        if not (pd.api.types.is_string_dtype(s) or s.dtype == object):
            continue
        if s.dropna().astype(str).str.contains(USER_ID).any():
            raise UnsafeSQLError(
                f"column {col!r} contains shopper ids; return aggregates (counts, sums) instead"
            )
