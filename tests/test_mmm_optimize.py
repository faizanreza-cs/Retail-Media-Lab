import numpy as np
import pandas as pd
import pytest

from rml import mmm, optimize, pipeline
from rml.synth import CHANNELS


def _rows(built):
    return {r["channel"]: r for r in built["mmm"]["channels"]}


def test_model_fits_and_pins_down_dsp(built):
    assert built["mmm"]["fit"]["r2"] > 0.8
    rows = _rows(built)
    dsp = rows["DSP"]
    assert dsp["roas"] == pytest.approx(dsp["true_roas"], rel=0.15)
    assert dsp["mroas_at_bau"] == pytest.approx(dsp["true_mroas"], rel=0.15)
    assert dsp["mroas_at_bau"] > rows["SP"]["mroas_at_bau"]  # the next dollar: DSP over SP
    # always-on, saturated SP is the hardest channel to pin down: its interval is the widest
    width = {c: r["roas_high"] - r["roas_low"] for c, r in rows.items()}
    assert width["SP"] > width["DSP"]


def test_reported_roas_matches_incremental_sales(built):
    for r in built["mmm"]["channels"]:  # one estimate, not a median next to a mean
        assert r["roas"] == pytest.approx(r["incremental_sales"] / r["spend"], rel=1e-3)
        assert r["roas_low"] <= r["roas"] <= r["roas_high"]


def test_current_spend_window_ends_before_the_test():
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=200)})
    window = mmm.days_before(df, pd.Timestamp("2025-06-01"))
    days = df["date"].iloc[window]
    assert len(days) == 91 and days.max() == pd.Timestamp("2025-05-31")
    with pytest.raises(ValueError):
        mmm.days_before(df, pd.Timestamp("2025-02-01"))


def test_curves_round_trip(built):
    c = pipeline.curves(built)
    again = mmm.Curves.from_dict(c.to_dict())
    xs = np.array([0.0, 1000.0, 5000.0])
    assert np.allclose(c.response("SB", xs), again.response("SB", xs))
    assert c.response("SB", 0.0) == pytest.approx(0.0)


def test_plan_respects_budget_and_limits(built):
    c = pipeline.curves(built)
    current = {r["channel"]: r["bau_daily_spend"] for r in built["mmm"]["channels"]}
    plan = optimize.optimize(c, current, max_change=0.2)
    rec = {r["channel"]: r["recommended_daily_spend"] for r in plan["channels"]}
    assert sum(rec.values()) == pytest.approx(sum(current.values()), rel=1e-6)
    for ch in CHANNELS:
        assert 0.8 * current[ch] - 1e-6 <= rec[ch] <= 1.2 * current[ch] + 1e-6
    assert plan["recommended_daily_incremental_sales"] >= plan["current_daily_incremental_sales"]
    bigger = optimize.optimize(c, current, total=sum(current.values()) * 1.5)
    assert bigger["total_daily_budget"] == pytest.approx(sum(current.values()) * 1.5)
    with pytest.raises(ValueError):
        optimize.optimize(c, current, min_roas=50)
    with pytest.raises(ValueError):
        optimize.optimize(c, current, max_change=1.5)


def test_free_channels_end_at_equal_marginal_roas(built):
    c = pipeline.curves(built)
    current = {r["channel"]: r["bau_daily_spend"] for r in built["mmm"]["channels"]}
    plan = optimize.optimize(c, current, max_change=0.3)
    free = [r["mroas_recommended"] for r in plan["channels"] if abs(r["change_pct"]) < 0.299]
    assert len(free) >= 2 and max(free) == pytest.approx(min(free), rel=0.01)


def test_recommended_plan_really_helps(built):
    score = built["budget"]["truth"]
    assert score["uplift_pct"] > 0
    assert score["share_of_best_captured"] > 0.4


def test_true_curves_match_the_simulator(built):
    t = built["truth"]
    tc = optimize.TrueCurves(t["media"], t["geo_weights"])
    for ch in CHANNELS:
        x = t["media"][ch]["bau_daily_spend"]
        assert tc.marginal_roas(ch, x) == pytest.approx(t["media"][ch]["mroas_at_bau"], rel=0.01)
