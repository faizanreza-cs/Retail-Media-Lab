import pandas as pd
import pytest

from rml import attribution as attr


def _j(rows):
    return pd.DataFrame(rows, columns=["path", "converted"])


def test_position_and_linear_rules_by_hand():
    j = _j([(("DSP", "SP"), True), (("SB",), True), (("DSP",), False)])
    assert attr.last_touch(j)[["SP", "SB", "DSP"]].tolist() == [1, 1, 0]
    assert attr.first_touch(j)[["SP", "SB", "DSP"]].tolist() == [0, 1, 1]
    assert attr.linear(j)[["SP", "SB", "DSP"]].tolist() == [0.5, 1.0, 0.5]


def test_markov_removal_effect_by_hand():
    # start -> SP -> conv (1), start -> DSP -> null (1). Removing SP kills every conversion,
    # removing DSP changes nothing, so SP gets all the credit.
    j = _j([(("SP",), True), (("DSP",), False)])
    credit = attr.markov(j)
    assert credit["SP"] == pytest.approx(1.0) and credit["DSP"] == pytest.approx(0.0)


def test_shapley_by_hand():
    # purchase rates: {SP}: 0.2, {DSP}: 0.1, {SP, DSP}: 0.4. For the two-channel set:
    # phi_SP = 0.5 * (0.2 - 0) + 0.5 * (0.4 - 0.1) = 0.25; phi_DSP = 0.15; shares 62.5 / 37.5.
    rows = ([(("SP",), True)] * 2 + [(("SP",), False)] * 8
            + [(("DSP",), True)] + [(("DSP",), False)] * 9
            + [(("DSP", "SP"), True)] * 4 + [(("DSP", "SP"), False)] * 6)  # fmt: skip
    credit = attr.shapley(_j(rows))
    assert credit["SP"] == pytest.approx(2 + 4 * 0.625)
    assert credit["DSP"] == pytest.approx(1 + 4 * 0.375)


def test_journeys_stop_at_first_purchase_and_collapse_repeats():
    touches = pd.DataFrame(
        {
            "user_id": ["a"] * 4,
            "ts": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-09"]),
            "channel": ["DSP", "DSP", "SP", "SB"],
        }
    )
    conv = pd.DataFrame({"user_id": ["a"], "conversion_event_dt_utc": ["2026-01-05"]})
    j = attr.journeys(touches, conv)
    assert j["path"].iloc[0] == ("DSP", "SP") and bool(j["converted"].iloc[0])


def test_on_simulated_data_last_touch_overcredits_sponsored_products(built):
    t = pd.DataFrame(built["attribution"]["table"])
    share = t.pivot(index="channel", columns="method", values="share")
    assert share.loc["SP", "last_touch"] > share.loc["SP", "truth"] + 0.08
    assert share.loc["DSP", "last_touch"] < share.loc["DSP", "truth"] - 0.05
    err = built["attribution"]["share_error"]
    assert err["shapley"] < err["last_touch"] and err["linear"] < err["last_touch"]
    # every method credits ads with far more purchases than they caused
    assert (
        built["attribution"]["exposed_buyers"]
        > 1.5 * built["attribution"]["true_incremental_purchasers"]
    )
