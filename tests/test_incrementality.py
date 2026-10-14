import pandas as pd
import pytest

from rml import incrementality as inc_mod


def test_both_methods_recover_the_true_holdout_effect(built):
    inc = built["incrementality"]
    truth = inc["truth"]
    for r in inc["results"]:
        assert r["iroas"] == pytest.approx(truth["iroas"], rel=0.15), r["method"]
        lo, hi = r["iroas_ci"]
        assert lo <= truth["iroas"] <= hi, r["method"]
    assert inc["results"][0]["spend_saved"] == pytest.approx(truth["spend_saved"], rel=0.03)


def test_placebo_test_finds_nothing(wh):
    sales = wh.query("SELECT * FROM daily_geo_sales")
    weekly = inc_mod.weekly_sales(sales)
    fake_geos = ["G01", "G02", "G04", "G05", "G06"]  # control geos: DSP never went dark
    start = pd.Timestamp("2026-04-06")
    early = weekly[weekly["week"] < pd.Timestamp("2026-06-01")]
    did = inc_mod.diff_in_diff(early, fake_geos, start)
    assert abs(did["lift_pct"]) < 0.02
    assert did["lift_pct_ci"][0] <= 0 <= did["lift_pct_ci"][1]
