"""Geo holdout analysis: what did DSP really add?

DSP was switched off in a few test geos for the final weeks, while the control geos kept it.
Two standard ways to read the test:

* **Difference-in-differences**: log weekly sales with geo and week fixed effects; the test
  effect is the change in test geos beyond what control geos did in the same weeks. Standard
  errors are clustered by geo.
* **Synthetic control**: build a weighted mix of control geos that tracks the test geos before
  the test, and use it as the counterfactual during the test. Uncertainty comes from placebo
  tests: the same method applied to random groups of control geos, where the true effect is 0.

Both report the sales the holdout cost, the DSP spend it saved, and incremental ROAS (iROAS).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import nnls

PRE_WEEKS = 52


def weekly_sales(daily_geo_sales: pd.DataFrame) -> pd.DataFrame:
    df = daily_geo_sales.assign(week=pd.to_datetime(daily_geo_sales["date"]).dt.to_period("W"))
    out = df.groupby(["geo_id", "week"], as_index=False)["sales"].sum()
    out["week"] = out["week"].dt.start_time
    return out


def spend_saved(
    daily_geo_media: pd.DataFrame, test_geos: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> float:
    """DSP spend the test geos would have had: control-geo spend scaled by pre-test shares."""
    d = daily_geo_media[daily_geo_media["channel"] == "DSP"].assign(
        date=lambda x: pd.to_datetime(x["date"])
    )
    pre = d[(d["date"] < start) & (d["date"] >= start - pd.Timedelta(weeks=13))]
    in_test = pre["geo_id"].isin(test_geos)
    ratio = pre.loc[in_test, "spend"].sum() / pre.loc[~in_test, "spend"].sum()
    during = d[(d["date"] >= start) & (d["date"] <= end) & ~d["geo_id"].isin(test_geos)]
    return float(during["spend"].sum() * ratio)


def _window(weekly: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    return weekly[weekly["week"] >= start - pd.Timedelta(weeks=PRE_WEEKS)]


def diff_in_diff(weekly: pd.DataFrame, test_geos: list[str], start: pd.Timestamp) -> dict[str, Any]:
    df = _window(weekly, start).copy()
    df["treated"] = (df["geo_id"].isin(test_geos) & (df["week"] >= start)).astype(int)
    df["log_sales"] = np.log(df["sales"])
    fit = smf.ols("log_sales ~ C(geo_id) + C(week) + treated", data=df).fit(
        cov_type="cluster",
        cov_kwds={"groups": pd.factorize(df["geo_id"])[0]},
        use_t=True,  # t with (geos - 1) degrees of freedom: only 20 clusters
    )
    delta = float(fit.params["treated"])
    lo, hi = (float(x) for x in fit.conf_int().loc["treated"])
    actual = float(df.loc[df["treated"] == 1, "sales"].sum())

    def lost(d: float) -> float:  # counterfactual minus actual, with actual = cf * exp(d)
        return actual * (np.exp(-d) - 1.0)

    return {
        "method": "difference_in_differences",
        "lift_pct": round(float(np.exp(delta) - 1.0), 4),
        "lift_pct_ci": [round(float(np.exp(lo) - 1.0), 4), round(float(np.exp(hi) - 1.0), 4)],
        "lost_sales": round(lost(delta), 2),
        "lost_sales_ci": [round(lost(hi), 2), round(lost(lo), 2)],
        "p_value": round(float(fit.pvalues["treated"]), 6),
    }


def _sc_weights(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    """Non-negative donor weights summing to 1 that best match the target (least squares)."""
    big = 100.0  # an extra heavily weighted row makes the weights sum to 1
    A = np.vstack([donors, big * np.ones(donors.shape[1])])
    b = np.append(target, big)
    w, _ = nnls(A, b)
    return np.asarray(w / w.sum())


def _sc_effect(Y: pd.DataFrame, treated: list[str], donors: list[str], pre: np.ndarray) -> dict:
    t = Y[treated].sum(axis=1).to_numpy()
    d = Y[donors].to_numpy()
    t_scale, d_scale = t[pre].mean(), d[pre].mean(axis=0)
    w = _sc_weights(t[pre] / t_scale, (d / d_scale)[pre])
    cf = (d / d_scale) @ w * t_scale
    post = ~pre
    return {
        "weights": dict(zip(donors, w.round(4), strict=True)),
        "pct": float((t[post] - cf[post]).sum() / cf[post].sum()),
        "cf_post": float(cf[post].sum()),
        "actual_post": float(t[post].sum()),
        "pre_fit_rmspe": float(np.sqrt(np.mean((t[pre] / cf[pre] - 1.0) ** 2))),
        "series": pd.DataFrame({"week": Y.index, "actual": t, "synthetic": cf}),
    }


def synthetic_control(
    weekly: pd.DataFrame, test_geos: list[str], start: pd.Timestamp, placebos: int = 200,
    seed: int = 0,
) -> dict[str, Any]:  # fmt: skip
    Y = _window(weekly, start).pivot(index="week", columns="geo_id", values="sales")
    pre = np.asarray(Y.index < start)
    controls = [g for g in Y.columns if g not in test_geos]
    est = _sc_effect(Y, test_geos, controls, pre)

    # Placebo tests: random control groups of the same size, where the true effect is zero
    rng = np.random.default_rng(seed)
    fake = []
    for _ in range(placebos):
        group = list(rng.choice(controls, len(test_geos), replace=False))
        pool = [g for g in controls if g not in group]
        fake.append(_sc_effect(Y, group, pool, pre)["pct"])
    band = float(np.quantile(np.abs(fake), 0.95))
    pct = est["pct"]
    cf = est["cf_post"]
    return {
        "method": "synthetic_control",
        "lift_pct": round(pct, 4),
        "lift_pct_ci": [round(pct - band, 4), round(pct + band, 4)],
        "lost_sales": round(-pct * cf, 2),
        "lost_sales_ci": [round(-(pct + band) * cf, 2), round(-(pct - band) * cf, 2)],
        "p_value": round(float(np.mean(np.abs(fake) >= abs(pct))), 4),
        "pre_fit_rmspe": round(est["pre_fit_rmspe"], 4),
        "weights": {k: float(v) for k, v in est["weights"].items() if v > 0.001},
        "series": est["series"],
    }


def analyse(
    daily_geo_sales: pd.DataFrame, daily_geo_media: pd.DataFrame, geo_test: pd.DataFrame,
    truth: dict | None = None,
) -> dict[str, Any]:  # fmt: skip
    """Read the DSP geo holdout with both methods. Returns results and the weekly series."""
    test_geos = sorted(geo_test.loc[geo_test["test_group"] == "holdout", "geo_id"])
    start = pd.Timestamp(geo_test["test_start"].iloc[0])
    end = pd.Timestamp(geo_test["test_end"].iloc[0])
    weekly = weekly_sales(daily_geo_sales)
    saved = spend_saved(daily_geo_media, test_geos, start, end)
    results = [diff_in_diff(weekly, test_geos, start), synthetic_control(weekly, test_geos, start)]
    series = results[1].pop("series")
    for r in results:
        r["spend_saved"] = round(saved, 2)
        r["iroas"] = round(r["lost_sales"] / saved, 4)
        r["iroas_ci"] = [round(x / saved, 4) for x in r["lost_sales_ci"]]
    out: dict[str, Any] = {
        "channel": "DSP",
        "test_geos": test_geos,
        "start": str(start.date()),
        "end": str(end.date()),
        "results": results,
        "series": series,
    }
    if truth:
        out["truth"] = {k: truth[k] for k in ("lost_sales", "spend_saved", "iroas")}
    return out
