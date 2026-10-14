"""Multi-touch attribution on shopper journeys, scored against the true causal split.

Methods, from simplest to most data-driven:

* **last touch / first touch**: all credit to the last (first) ad before the purchase.
* **linear**: equal credit to every step of the path.
* **Markov removal effect**: model journeys as a chain of ad types; a channel's credit is how
  much the overall purchase probability drops when it is removed from every path.
* **Shapley**: for shoppers who saw a set of ad types, split each purchase by how much every
  ad type adds to the purchase *rate* across all sub-combinations (Shapley values).

All of them only see who was exposed and who bought, so they can't separate "the ad caused the
purchase" from "the ad was shown to someone who was going to buy anyway". The simulator knows
the true split, which is what ``compare`` reports next to each method.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

from rml.synth import AD_PRODUCT_TYPES, CHANNELS

METHODS = ("last_touch", "first_touch", "linear", "markov", "shapley")
_TYPE_TO_CHANNEL = {v: k for k, v in AD_PRODUCT_TYPES.items()}


def touches(dsp_impressions: pd.DataFrame, sponsored_ads_traffic: pd.DataFrame) -> pd.DataFrame:
    """One row per ad event: user_id, ts, channel."""
    dsp = pd.DataFrame(
        {
            "user_id": dsp_impressions["user_id"],
            "ts": pd.to_datetime(dsp_impressions["impression_dt_utc"]),
            "channel": "DSP",
        }
    )
    sp = pd.DataFrame(
        {
            "user_id": sponsored_ads_traffic["user_id"],
            "ts": pd.to_datetime(sponsored_ads_traffic["event_dt_utc"]),
            "channel": sponsored_ads_traffic["ad_product_type"].map(_TYPE_TO_CHANNEL),
        }
    )
    if sp["channel"].isna().any():
        raise ValueError("unknown ad_product_type in sponsored_ads_traffic")
    return pd.concat([dsp, sp], ignore_index=True)


def journeys(touch_df: pd.DataFrame, conversions: pd.DataFrame) -> pd.DataFrame:
    """One row per exposed shopper: the ordered path of ad types and whether they purchased.

    Buyers' paths stop at their first purchase. Repeated touches of one ad type in a row
    collapse into a single step. Shoppers who bought without seeing an ad are not attributable
    and are left out (``unexposed_buyers`` counts them).
    """
    first_buy = (
        conversions.assign(ts=pd.to_datetime(conversions["conversion_event_dt_utc"]))
        .groupby("user_id")["ts"]
        .min()
        .rename("purchase_ts")
    )
    t = touch_df.join(first_buy, on="user_id")
    t = t[t["purchase_ts"].isna() | (t["ts"] < t["purchase_ts"])]
    t = t.sort_values(["user_id", "ts"], kind="stable")
    prev = t.groupby("user_id")["channel"].shift()
    t = t[prev.isna() | (prev != t["channel"])]
    paths = t.groupby("user_id")["channel"].agg(tuple).rename("path").to_frame()
    paths["converted"] = paths.index.isin(first_buy.index)
    return paths.reset_index()


def unexposed_buyers(journey_df: pd.DataFrame, conversions: pd.DataFrame) -> int:
    return int(conversions.loc[~conversions["user_id"].isin(journey_df["user_id"]), "user_id"]
               .nunique())  # fmt: skip


# --------------------------------------------------------------------------- methods


def _position(j: pd.DataFrame, idx: int) -> pd.Series:
    conv = j[j["converted"]]
    counts = conv["path"].map(lambda p: p[idx]).value_counts()
    return counts.reindex(CHANNELS, fill_value=0).astype(float)


def last_touch(j: pd.DataFrame) -> pd.Series:
    return _position(j, -1)


def first_touch(j: pd.DataFrame) -> pd.Series:
    return _position(j, 0)


def linear(j: pd.DataFrame) -> pd.Series:
    credit = dict.fromkeys(CHANNELS, 0.0)
    for path, n in j.loc[j["converted"], "path"].value_counts().items():
        for ch in path:
            credit[ch] += n / len(path)
    return pd.Series(credit)


def _absorption(P: np.ndarray, n_transient: int, conv: int) -> float:
    """Probability of ending in ``conv`` from state 0 for an absorbing chain."""
    Q = P[:n_transient, :n_transient]
    R = P[:n_transient, n_transient:]
    B = np.linalg.solve(np.eye(n_transient) - Q, R)
    return float(B[0, conv - n_transient])


def markov(j: pd.DataFrame) -> pd.Series:
    """First-order Markov chain, credit proportional to each channel's removal effect."""
    states = ["start", *CHANNELS, "conv", "null"]
    ix = {s: i for i, s in enumerate(states)}
    counts = np.zeros((len(states), len(states)))
    grouped = j.groupby(["path", "converted"]).size()
    for (path, converted), n in grouped.items():
        seq = ["start", *path, "conv" if converted else "null"]
        for a, b in zip(seq, seq[1:], strict=False):
            counts[ix[a], ix[b]] += n
    n_t = 1 + len(CHANNELS)
    P = np.zeros_like(counts)
    rows = counts[:n_t].sum(axis=1, keepdims=True)
    P[:n_t] = np.divide(counts[:n_t], rows, out=np.zeros_like(counts[:n_t]), where=rows > 0)
    base = _absorption(P, n_t, ix["conv"])
    effects = {}
    for ch in CHANNELS:
        Pr = P.copy()
        Pr[:n_t, ix["null"]] += Pr[:n_t, ix[ch]]  # paths that would reach ch now end
        Pr[:n_t, ix[ch]] = 0.0
        effects[ch] = max(0.0, 1.0 - _absorption(Pr, n_t, ix["conv"]) / base)
    total = sum(effects.values())
    buyers = float(j["converted"].sum())
    return pd.Series({ch: buyers * effects[ch] / total for ch in CHANNELS})


def shapley(j: pd.DataFrame) -> pd.Series:
    """Shapley split of each purchase over the ad types the buyer saw, using purchase rates.

    v(S) is the purchase rate of shoppers who saw exactly the ad types in S (0 for no ads or an
    unobserved combination). For buyers who saw set S, each ad type's share is its Shapley value
    in the game restricted to S, divided by v(S).
    """
    sets = j.assign(s=j["path"].map(frozenset)).groupby("s")["converted"].agg(["sum", "count"])
    rate = {s: row["sum"] / row["count"] for s, row in sets.iterrows()}
    credit = dict.fromkeys(CHANNELS, 0.0)
    for s, row in sets.iterrows():
        if row["sum"] == 0:
            continue
        members = sorted(s)
        k = len(members)
        phi = {}
        for ch in members:
            others = [m for m in members if m != ch]
            val = 0.0
            for r in range(k):
                for sub in combinations(others, r):
                    w = factorial(r) * factorial(k - r - 1) / factorial(k)
                    with_ch = rate.get(frozenset((*sub, ch)), 0.0)
                    without = rate.get(frozenset(sub), 0.0)
                    val += w * (with_ch - without)
            phi[ch] = val
        total = sum(phi.values())
        for ch in members:
            share = phi[ch] / total if total > 0 else 1.0 / k
            credit[ch] += row["sum"] * share
    return pd.Series(credit)


_FUNCS = {
    "last_touch": last_touch,
    "first_touch": first_touch,
    "linear": linear,
    "markov": markov,
    "shapley": shapley,
}


def compare(j: pd.DataFrame, truth: dict | None = None) -> pd.DataFrame:
    """Credited purchasers and share per channel for every method (+ the truth, if known)."""
    out = pd.DataFrame({m: _FUNCS[m](j) for m in METHODS}).reindex(list(CHANNELS))
    shares = out / out.sum()
    rows = []
    for ch in CHANNELS:
        for m in METHODS:
            rows.append(
                {
                    "channel": ch,
                    "method": m,
                    "credited_purchasers": round(float(out.loc[ch, m]), 1),
                    "share": round(float(shares.loc[ch, m]), 4),
                }
            )
        if truth:
            rows.append(
                {
                    "channel": ch,
                    "method": "truth",
                    "credited_purchasers": truth["incremental_purchasers"][ch],
                    "share": truth["incremental_share"][ch],
                }
            )
    return pd.DataFrame(rows)


def share_error(table: pd.DataFrame) -> pd.Series:
    """Total absolute share error of each method against the truth (0 = perfect)."""
    wide = table.pivot(index="channel", columns="method", values="share")
    return (wide.drop(columns="truth").sub(wide["truth"], axis=0).abs().sum() / 2).round(4)
