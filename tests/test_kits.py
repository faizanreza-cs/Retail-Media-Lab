import ast
import importlib
import json
import re
from pathlib import Path

import pandas as pd

from rml import powerbi
from rml.api import app

ROOT = Path(__file__).resolve().parents[1]


def test_powerbi_export_and_measures_agree(tmp_path):
    paths = powerbi.export(tmp_path)
    cols = {p.stem: set(pd.read_csv(p, nrows=1).columns) for p in paths}
    assert {"dim_date", "dim_channel", "fact_media_daily", "fact_budget_plan"} <= cols.keys()
    dax = (ROOT / "powerbi" / "measures.dax").read_text()
    refs = re.findall(r"\b(dim_\w+|fact_\w+)\[(\w+)\]", dax)
    assert refs
    for table, column in refs:  # every measure points at a real exported column
        assert column in cols[table], f"{table}[{column}]"


def test_powerbi_theme_is_valid():
    theme = json.loads((ROOT / "powerbi" / "theme.json").read_text())
    assert len(theme["dataColors"]) == 8
    assert all(re.fullmatch(r"#[0-9A-F]{6}", c) for c in theme["dataColors"])


def test_copilot_connector_matches_the_api():
    spec = json.loads((ROOT / "copilot" / "ads-copilot.swagger.json").read_text())
    assert spec["swagger"] == "2.0"
    routes = {(r.path, m) for r in app.routes for m in getattr(r, "methods", set())}
    for path, ops in spec["paths"].items():
        for method in ops:
            assert (path, method.upper()) in routes, path
    api_fields = {
        "AskRequest": set(app.openapi()["components"]["schemas"]["AskRequest"]["properties"]),
        "OptimizeRequest": set(
            app.openapi()["components"]["schemas"]["OptimizeRequest"]["properties"]
        ),
    }
    for name, fields in api_fields.items():
        assert set(spec["definitions"][name]["properties"]) == fields, name


def test_dataiku_recipes_import_real_functions():
    for recipe in (ROOT / "dataiku" / "recipes").glob("*.py"):
        tree = ast.parse(recipe.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("rml"):
                mod = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(mod, alias.name), f"{recipe.name}: {node.module}.{alias.name}"


def test_snowflake_setup_has_a_read_only_role():
    sql = (ROOT / "snowflake" / "setup.sql").read_text()
    assert "RML_ANALYST" in sql and "GRANT SELECT" in sql
    analyst = [line for line in sql.splitlines() if "TO ROLE RML_ANALYST" in line]
    assert analyst and not any(re.search(r"CREATE|INSERT|UPDATE|DELETE", g) for g in analyst)
