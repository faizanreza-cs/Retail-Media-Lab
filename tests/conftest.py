"""Build the data and analyses once per test session, with a smaller model search."""

from __future__ import annotations

import os

import pytest

from rml import pipeline, warehouse


@pytest.fixture(scope="session", autouse=True)
def built(tmp_path_factory: pytest.TempPathFactory) -> dict:
    db = tmp_path_factory.mktemp("data") / "rml.duckdb"
    old = {k: os.environ.get(k) for k in ("RML_DB", "RML_PROVIDER", "RML_BACKEND")}
    os.environ.update({"RML_DB": str(db), "RML_PROVIDER": "mock", "RML_BACKEND": "duckdb"})
    results = pipeline.build(n_search=250)
    yield results
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    warehouse.reset_cache()


@pytest.fixture
def wh() -> warehouse.Warehouse:
    return warehouse.get_warehouse()
