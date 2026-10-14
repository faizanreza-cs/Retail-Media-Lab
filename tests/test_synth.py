import numpy as np
import pandas as pd
import pytest

from rml import synth
from rml.synth import CHANNELS, SynthConfig, adstock, hill

SMALL = SynthConfig(n_shoppers=20_000)


def test_adstock_and_hill_by_hand():
    x = np.array([[10.0], [0.0], [0.0]])
    # a[0] = 0.5 * 10, a[1] = 0.5 * 5, a[2] = 0.5 * 2.5
    assert adstock(x, 0.5).ravel().tolist() == [5.0, 2.5, 1.25]
    steady = adstock(np.full((200, 1), 7.0), 0.8)[-1, 0]
    assert steady == pytest.approx(7.0)  # constant spend settles at the spend level
    assert float(hill(3.0, 3.0)) == 0.5 and float(hill(0.0, 1.0)) == 0.0


def test_generation_is_deterministic():
    a, b = synth.generate(SMALL), synth.generate(SMALL)
    assert a.truth == b.truth
    for name in a.tables:
        pd.testing.assert_frame_equal(a.tables[name], b.tables[name])


def test_truth_is_consistent():
    d = synth.generate(SMALL)
    shares = d.truth["journeys"]["incremental_share"]
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-3)
    for ch in CHANNELS:  # realised ROAS matches the configured truth
        assert d.truth["media"][ch]["roas"] == pytest.approx(synth.MEDIA[ch].roas, rel=1e-3)
    g = d.truth["geo_test"]
    assert g["lost_sales"] > 0
    assert g["iroas"] == pytest.approx(g["lost_sales"] / g["spend_saved"], rel=1e-3)


def test_holdout_switches_dsp_off_only_in_test_geos():
    d = synth.generate(SMALL)
    m = d.tables["daily_geo_media"]
    g = d.truth["geo_test"]
    dsp = m[(m["channel"] == "DSP") & (m["date"] >= pd.Timestamp(g["start"]))]
    in_test = dsp["geo_id"].isin(g["test_geos"])
    assert dsp.loc[in_test, "spend"].sum() == 0
    assert (dsp.loc[~in_test].groupby("geo_id")["spend"].sum() > 0).all()


def test_purchases_follow_the_ads_before_them():
    d = synth.generate(SMALL)
    conv = d.tables["conversions"]
    first = conv.groupby("user_id")["conversion_event_dt_utc"].min()
    ads = pd.concat(
        [
            d.tables["dsp_impressions"].set_index("user_id")["impression_dt_utc"],
            d.tables["sponsored_ads_traffic"].set_index("user_id")["event_dt_utc"],
        ]
    )
    last_ad = ads.groupby(level=0).max()
    both = first.index.intersection(last_ad.index)
    assert (first[both] > last_ad[both]).all()
