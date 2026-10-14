import pandas as pd
import pytest

from rml import warehouse
from rml.agent.sql_guard import UnsafeSQLError, check_result, scrub_error, validate
from rml.agent.tools import execute
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


# ways to get shopper ids out: wrapped, aliased, through CTEs, unions, stars, filters, errors
LEAKS = [
    "SELECT UPPER(MAX(user_id)) AS id FROM conversions GROUP BY user_id",
    "SELECT REVERSE(MIN(user_id)) FROM conversions GROUP BY user_id",
    "SELECT LISTAGG(user_id, ',') FROM conversions",
    "WITH t AS (SELECT user_id AS u FROM conversions) SELECT UPPER(u) FROM t",
    "WITH t AS (SELECT user_id FROM conversions) SELECT * FROM t",
    "SELECT * FROM conversions",
    "SELECT * EXCLUDE (purchases) FROM conversions",
    "SELECT x FROM (SELECT user_id FROM conversions) AS s(x)",
    "SELECT u FROM (SELECT 'x' AS u UNION ALL SELECT user_id FROM conversions)",
    "SELECT SUBSTR(user_id, 2) FROM conversions",
    "SELECT COUNT(*) FROM conversions WHERE CAST(REVERSE(user_id) AS INT) = 1",
    "SELECT COUNT(*) FROM conversions WHERE user_id = 'u0119912'",
    "SELECT COUNT(*) FROM conversions WHERE user_id LIKE 'u011%'",
    "WITH g AS (SELECT 'u0000001' AS user_id) "
    "SELECT COUNT(*) FROM g JOIN conversions c ON c.user_id = g.user_id",
    "SELECT c FROM conversions c",
    "SELECT TO_JSON(OBJECT_CONSTRUCT(*)) FROM conversions",
    "SELECT REVERSE(COLUMNS('user_id')) FROM conversions",
    "SELECT * FROM (SELECT user_id, purchases FROM conversions) "
    "PIVOT (SUM(purchases) FOR user_id IN ('u0000001'))",
    "SELECT (SELECT MAX(user_id) FROM conversions) AS x",
    "SELECT CASE WHEN purchases > 1 THEN user_id END FROM conversions",
    "SELECT DISTINCT user_id FROM conversions",
    "SELECT HASH(user_id) FROM conversions",
]
# what analysts do with shopper ids: count, group, join and pass them through to an aggregate
ALLOWED = [
    "SELECT COUNT(DISTINCT user_id) AS purchasers FROM conversions",
    "SELECT COUNT(DISTINCT c.user_id) AS n FROM conversions c JOIN dsp_impressions d "
    "ON c.user_id = d.user_id AND d.impression_dt_utc < c.conversion_event_dt_utc",
    "WITH per_user AS (SELECT user_id, COUNT(*) AS n FROM conversions GROUP BY user_id) "
    "SELECT n, COUNT(*) AS shoppers FROM per_user GROUP BY n",
    "SELECT COUNT(*) AS n FROM dsp_impressions d "
    "WHERE d.user_id IN (SELECT user_id FROM conversions)",
    "SELECT COUNT(*) AS n FROM dsp_impressions d "
    "WHERE EXISTS (SELECT 1 FROM conversions c WHERE c.user_id = d.user_id)",
    "WITH touches AS (SELECT user_id, 'DSP' AS ch FROM dsp_impressions UNION ALL "
    "SELECT user_id, ad_product_type FROM sponsored_ads_traffic) "
    "SELECT ch, COUNT(DISTINCT user_id) AS reach FROM touches GROUP BY ch",
    "WITH f AS (SELECT user_id, ad_product_type, ROW_NUMBER() OVER "
    "(PARTITION BY user_id ORDER BY event_dt_utc) AS rn FROM sponsored_ads_traffic) "
    "SELECT ad_product_type, COUNT(*) AS shoppers FROM f WHERE rn = 1 GROUP BY 1",
    "SELECT COUNT(*) AS n FROM conversions JOIN dsp_impressions USING (user_id)",
    "SELECT campaign, APPROX_COUNT_DISTINCT(user_id) AS reach FROM dsp_impressions GROUP BY 1",
]


@pytest.mark.parametrize("sql", LEAKS)
def test_shopper_ids_cant_leave(sql):
    with pytest.raises(UnsafeSQLError):
        validate(sql)


@pytest.mark.parametrize("sql", ALLOWED)
def test_shopper_ids_can_be_counted_grouped_and_joined(wh, sql):
    df = wh.query(validate(sql))
    check_result(df)
    assert len(df) > 0


def test_result_check_catches_ids_in_any_case():
    for value in ("u0119912", "U0119912", "ids: U0000001,U0000002"):
        with pytest.raises(UnsafeSQLError):
            check_result(pd.DataFrame({"x": [value]}))
    check_result(pd.DataFrame({"x": ["SP", "DSP"], "n": [1, 2]}))


def test_engine_errors_dont_echo_values(wh):
    assert scrub_error("Could not convert string 'u0119912' to INT32") == (
        "Could not convert string '...' to INT32"
    )
    out, is_error = execute("run_sql", {"sql": "SELECT CAST(campaign AS INT) FROM dsp_impressions"})
    assert is_error and "convert" in out.lower() and "Brand" not in out


def test_long_queries_are_stopped(wh):
    cross = (
        "SELECT COUNT(*) FROM sponsored_ads_traffic a JOIN sponsored_ads_traffic b "
        "ON a.user_id <> b.user_id"
    )
    with pytest.raises(TimeoutError):
        wh.query(cross, timeout=0.5)
    assert len(wh.query("SELECT 1 AS x")) == 1  # the connection still works


def test_snowflake_connects_once(monkeypatch):
    made = []

    class Fake:
        name = "snowflake"

        def __init__(self):
            made.append(self)

    monkeypatch.setenv("RML_BACKEND", "snowflake")
    monkeypatch.setattr(warehouse, "SnowflakeWarehouse", Fake)
    warehouse.reset_cache()
    try:
        assert warehouse.get_warehouse() is warehouse.get_warehouse() and len(made) == 1
    finally:
        warehouse.reset_cache()


def test_snowflake_backend_sends_sql_unchanged_and_lowercases_columns():
    seen = []

    class Cursor:
        def execute(self, sql, timeout=None):
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
