import pandas as pd
import pytest

from rml.agent.sql_guard import UnsafeSQLError, check_result, validate
from rml.queries import MIN_USERS, run_saved, saved_queries
from rml.warehouse import SnowflakeWarehouse, to_duckdb


@pytest.mark.parametrize("name", sorted(saved_queries()))
def test_saved_queries_run_and_respect_the_threshold(wh, name):
    df = run_saved(name, wh)
    assert len(df) > 0
    assert (df["users"] >= MIN_USERS).all()


def test_threshold_is_enforced_even_if_a_query_forgets():
    class Fake:
        name = "fake"

        def query(self, sql):
            return pd.DataFrame({"users": [150, 99], "x": [1, 2]})

    assert run_saved("ad_type_overlap", Fake())["users"].tolist() == [150]


def test_snowflake_functions_translate_to_duckdb():
    sql = to_duckdb("SELECT IFF(a > 1, 1, 0), DATEADD(day, 7, d), COUNT_IF(a > 1) FROM t")
    assert "CASE WHEN" in sql and "INTERVAL" in sql


def test_engine_blocks_file_access(wh, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("token\nTOPSECRET\n")
    with pytest.raises(Exception) as err:
        wh.query(f"SELECT * FROM read_csv('{secret.as_posix()}')")
    assert "TOPSECRET" not in str(err.value)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE geos",
        "SELECT * FROM read_csv('/etc/passwd')",
        "SELECT * FROM brands",
        "SELECT 1; SELECT 2",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM geos, LATERAL FLATTEN(input => 1)",
        "SELECT user_id, SUM(product_sales) FROM conversions GROUP BY 1",
        "SELECT c.user_id AS uid FROM conversions c",
        "SELECT 'x' || user_id AS k FROM conversions",
    ],
)
def test_guard_blocks(sql):
    with pytest.raises(UnsafeSQLError):
        validate(sql)


def test_guard_caps_rows_and_keeps_order(wh):
    out = validate("SELECT geo_id FROM geos ORDER BY geo_id DESC LIMIT 1000", row_limit=5)
    assert out.endswith("LIMIT 5")
    assert wh.query(out)["geo_id"].tolist() == ["G20", "G19", "G18", "G17", "G16"]
    pct = validate("SELECT * FROM daily_geo_sales LIMIT 50 PERCENT", row_limit=5)
    assert len(wh.query(pct)) == 5


def test_aggregates_allowed_but_shopper_ids_never_returned(wh):
    ok = validate(
        "WITH b AS (SELECT DISTINCT user_id FROM conversions) SELECT COUNT(*) AS n FROM b"
    )
    check_result(wh.query(ok))
    for sql in ("SELECT * FROM conversions",
                "WITH x AS (SELECT user_id AS u FROM conversions) SELECT u FROM x"):  # fmt: skip
        with pytest.raises(UnsafeSQLError):
            check_result(wh.query(validate(sql)))


def test_snowflake_backend_sends_sql_unchanged_and_lowercases_columns():
    seen = []

    class Cursor:
        def execute(self, sql):
            seen.append(sql)

        def fetch_pandas_all(self):
            return pd.DataFrame({"CHANNEL": ["SP"], "SPEND": [1.0]})

        def close(self):
            pass

    class Conn:
        def cursor(self):
            return Cursor()

    sql = "SELECT channel, IFF(spend > 0, 1, 0) AS spend FROM daily_geo_media"
    df = SnowflakeWarehouse(connection=Conn()).query(sql)
    assert seen == [sql] and list(df.columns) == ["channel", "spend"]
