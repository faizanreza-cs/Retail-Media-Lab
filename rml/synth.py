"""Synthetic retail media data for a fictional supplements brand on Amazon, with ground truth.

Two layers, both seeded and reproducible:

* **Shopper journeys** (60 days): one row per ad event and per purchase, in tables shaped like
  Amazon Marketing Cloud's (``dsp_impressions``, ``sponsored_ads_traffic``, ``conversions``).
  Every ad touch raises the shopper's purchase probability by a known amount, so the true
  number of purchases each channel *caused* is known exactly.
* **Geo x day aggregates** (two years): spend, impressions and sales for 20 geos, generated with
  adstock (carry-over), saturation, seasonality, retail events and a DSP geo holdout at the end.
  The true ROAS of every channel and the true sales lost in the holdout are known exactly.

Every method in this repo is scored against that ground truth. The brand, products, campaigns
and numbers are made up, and the table layouts are simplified versions of AMC's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BRAND = "Fernway Nutrition"
CHANNELS: tuple[str, ...] = ("SP", "SB", "SD", "DSP")
CHANNEL_NAMES = {
    "SP": "Sponsored Products",
    "SB": "Sponsored Brands",
    "SD": "Sponsored Display",
    "DSP": "Amazon DSP",
}
FUNNEL_STAGE = {
    "DSP": "Awareness",
    "SD": "Consideration",
    "SB": "Consideration",
    "SP": "Conversion",
}
AD_PRODUCT_TYPES = {"SP": "sponsored_products", "SB": "sponsored_brands", "SD": "sponsored_display"}

ASINS = pd.DataFrame(
    [
        ("B0FWN00001", "Fernway Collagen Peptides 20 oz", "Collagen", 29.99, 0.24),
        ("B0FWN00002", "Fernway Whey Protein Vanilla 2 lb", "Protein", 39.99, 0.20),
        ("B0FWN00003", "Fernway Daily Multivitamin 120 ct", "Multivitamins", 19.99, 0.18),
        ("B0FWN00004", "Fernway Vitamin D3 5000 IU 240 ct", "Vitamins", 14.99, 0.14),
        ("B0FWN00005", "Fernway Magnesium Glycinate 180 ct", "Minerals", 22.99, 0.12),
        ("B0FWN00006", "Fernway Probiotic 50B CFU 60 ct", "Probiotics", 34.99, 0.12),
    ],
    columns=["tracked_asin", "product_name", "category", "list_price", "sales_mix"],
)

REGIONS = ("Northeast", "Southeast", "Midwest", "South Central", "West")

# Retail events: (first day, length in days, name, baseline sales lift, ad spend multiplier)
EVENTS: tuple[tuple[str, int, str, float, float], ...] = (
    ("2024-10-08", 2, "Prime Big Deal Days", 0.45, 1.7),
    ("2024-11-29", 4, "Black Friday - Cyber Monday", 0.40, 1.6),
    ("2025-07-08", 4, "Prime Day", 0.55, 1.8),
    ("2025-10-07", 2, "Prime Big Deal Days", 0.45, 1.7),
    ("2025-11-28", 4, "Black Friday - Cyber Monday", 0.40, 1.6),
    ("2026-07-14", 2, "Prime Day", 0.55, 1.8),
)


@dataclass(frozen=True)
class MediaTruth:
    """How a channel really drives sales in the geo x day data."""

    daily_spend: float  # typical national daily spend, USD
    decay: float  # adstock carry-over per day (normalised geometric adstock)
    half_sat: float  # spend level at half of maximum effect, as a multiple of mean daily spend
    roas: float  # true average return on ad spend over the two years


MEDIA: dict[str, MediaTruth] = {
    # SP: biggest budget, best average ROAS, but close to saturation
    "SP": MediaTruth(daily_spend=9000, decay=0.15, half_sat=0.4, roas=3.0),
    "SB": MediaTruth(daily_spend=3000, decay=0.35, half_sat=1.0, roas=2.4),
    "SD": MediaTruth(daily_spend=2200, decay=0.30, half_sat=1.2, roas=1.6),
    # DSP: upper funnel, long carry-over, far from saturation
    "DSP": MediaTruth(daily_spend=4200, decay=0.65, half_sat=2.5, roas=2.0),
}


@dataclass(frozen=True)
class JourneyTruth:
    """How a channel reaches shoppers and moves purchase probability in the journey data."""

    reach_in_market: float  # share of in-market shoppers reached
    reach_other: float  # share of other shoppers reached
    extra_touches: float  # reached shoppers see 1 + Poisson(extra_touches) ads
    lift_per_touch: float  # each touch converts this share of the remaining non-buyers
    window: tuple[float, float]  # when touches happen, in days after the journey starts


JOURNEY: dict[str, JourneyTruth] = {
    # Sponsored ads target shoppers already searching the category: high reach among in-market
    # shoppers, who would often buy anyway. That selection is what fools click-based attribution.
    "SP": JourneyTruth(0.75, 0.06, 3.0, 0.014, (12, 29)),
    "SB": JourneyTruth(0.40, 0.05, 1.5, 0.012, (8, 27)),
    "SD": JourneyTruth(0.32, 0.04, 2.0, 0.010, (4, 24)),
    # DSP prospecting reaches a broad audience early in the journey
    "DSP": JourneyTruth(0.42, 0.40, 4.0, 0.0045, (0, 20)),
}


@dataclass(frozen=True)
class SynthConfig:
    seed: int = 7
    start: str = "2024-10-07"  # a Monday
    weeks: int = 104
    n_geos: int = 20
    n_test_geos: int = 5
    holdout_weeks: int = 8  # DSP switched off in the test geos for the final weeks
    n_shoppers: int = 120_000
    journey_days: int = 60  # journey window ends the day before the geo test starts
    base_daily_sales: float = 52_000.0  # national organic sales, USD per day


@dataclass
class SynthData:
    tables: dict[str, pd.DataFrame]
    truth: dict[str, Any]


# --------------------------------------------------------------------------- transforms


def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Normalised geometric adstock along axis 0: a[t] = decay * a[t-1] + (1 - decay) * x[t].

    A constant spend x settles at x, so the response curve can be read in spend units.
    """
    out = np.empty_like(x, dtype=float)
    acc = np.zeros(x.shape[1:], dtype=float)
    for t in range(x.shape[0]):
        acc = decay * acc + (1.0 - decay) * x[t]
        out[t] = acc
    return out


def hill(a: np.ndarray | float, half_sat: np.ndarray | float) -> np.ndarray:
    """Saturation curve with slope 1: 0 at no spend, 0.5 at ``half_sat``, approaching 1."""
    a = np.asarray(a, dtype=float)
    return a / (a + half_sat)


# --------------------------------------------------------------------------- geo x day layer


def _calendar(cfg: SynthConfig) -> pd.DataFrame:
    days = pd.date_range(cfg.start, periods=cfg.weeks * 7, freq="D")
    cal = pd.DataFrame({"date": days})
    cal["event"] = ""
    cal["event_lift"] = 0.0
    cal["event_spend"] = 1.0
    for first, length, name, lift, spend in EVENTS:
        mask = (cal["date"] >= pd.Timestamp(first)) & (
            cal["date"] < pd.Timestamp(first) + pd.Timedelta(days=length)
        )
        cal.loc[mask, ["event", "event_lift", "event_spend"]] = [name, lift, spend]
    # New-year resolution season: a lift that fades over the first three weeks of January
    jan_day = np.where(cal["date"].dt.month == 1, cal["date"].dt.day - 1, 99)
    cal["new_year"] = np.where(jan_day < 21, np.exp(-jan_day / 7.0), 0.0)
    return cal


def _spend_plans(cfg: SynthConfig, cal: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """National daily spend per channel (days x channels) with realistic, independent variation."""
    n = len(cal)
    plan = np.zeros((n, len(CHANNELS)))
    season = 1.0 + 0.25 * cal["new_year"].to_numpy() + 0.10 * (cal["date"].dt.month >= 11)
    event = cal["event_spend"].to_numpy()
    holdout_start = n - cfg.holdout_weeks * 7
    for j, ch in enumerate(CHANNELS):
        # AR(1) noise in log spend: day-to-day pacing and bid changes
        eps = np.zeros(n)
        for t in range(1, n):
            eps[t] = 0.75 * eps[t - 1] + rng.normal(0, 0.06)
        # budget changes every ~8 weeks
        steps = np.repeat(rng.normal(0, 0.12, n // 56 + 1), 56)[:n]
        on = np.ones(n)
        if ch == "SP":
            # two 10-day budget tests at half spend
            for s in rng.choice(np.arange(60, holdout_start - 60), 2, replace=False):
                on[s : s + 10] = 0.5
        elif ch in {"SB", "SD"}:
            for s in rng.choice(np.arange(30, holdout_start - 30), 3, replace=False):
                on[s : s + int(rng.integers(7, 15))] = 0.0
        else:  # DSP runs in flights, and stays on through the geo test
            t = 0
            while t < holdout_start - 28:
                t += int(rng.integers(21, 43))
                gap = int(rng.integers(7, 22))
                on[t : min(t + gap, holdout_start - 28)] = 0.0
                t += gap
        plan[:, j] = MEDIA[ch].daily_spend * season * event * np.exp(eps + steps) * on
    return plan


def _geo_sizes(cfg: SynthConfig, rng: np.random.Generator) -> np.ndarray:
    sizes = rng.lognormal(0, 0.6, cfg.n_geos)
    return np.asarray(sizes / sizes.sum())


def _baseline(cfg: SynthConfig, cal: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """National organic sales per day: trend, seasonality, weekday, events, coupons."""
    t = np.arange(len(cal))
    doy = cal["date"].dt.dayofyear.to_numpy()
    weekday = np.array([0.04, 0.01, -0.01, -0.02, -0.03, -0.01, 0.02])[cal["date"].dt.weekday]
    coupon = np.zeros(len(cal))
    for s in rng.choice(np.arange(14, len(cal) - 14, 7), 8, replace=False):
        coupon[s : s + 7] = 0.12
    cal["coupon"] = coupon  # the brand's own coupon calendar, known to the analyst
    rel = (
        0.10 * t / 365.0
        + 0.08 * np.sin(2 * np.pi * doy / 365.25)
        + 0.04 * np.cos(2 * np.pi * doy / 365.25)
        + 0.30 * cal["new_year"].to_numpy()
        + cal["event_lift"].to_numpy()
        + weekday
        + coupon
    )
    return cfg.base_daily_sales * (1.0 + rel)


def _geo_layer(cfg: SynthConfig, rng: np.random.Generator) -> tuple[dict[str, pd.DataFrame], dict]:
    cal = _calendar(cfg)
    n, g = len(cal), cfg.n_geos
    plan = _spend_plans(cfg, cal, rng)
    w = _geo_sizes(cfg, rng)
    geo_factor = rng.lognormal(0, 0.05, g)

    # Split national spend across geos with targeting noise, keeping the national total
    noise = rng.lognormal(0, 0.08, (n, len(CHANNELS), g))
    split = w[None, None, :] * noise
    split /= split.sum(axis=2, keepdims=True)
    spend = plan[:, :, None] * split  # days x channels x geos

    # DSP geo holdout: test geos go dark for the final weeks; the budget is not moved elsewhere
    test_geos = np.sort(rng.choice(np.arange(g), cfg.n_test_geos, replace=False))
    hold = np.arange(n - cfg.holdout_weeks * 7, n)
    dsp = CHANNELS.index("DSP")
    planned_dsp = spend[:, dsp, :].copy()
    spend[np.ix_(hold, [dsp], test_geos)] = 0.0

    base = _baseline(cfg, cal, rng)
    contrib = np.zeros_like(spend)
    truth_media: dict[str, Any] = {}
    betas = np.zeros(len(CHANNELS))
    for j, ch in enumerate(CHANNELS):
        m = MEDIA[ch]
        k_nat = m.half_sat * plan[:, j].mean()
        a = adstock(spend[:, j, :], m.decay)
        shape = w[None, :] * hill(a / w[None, :], k_nat)
        betas[j] = m.roas * spend[:, j, :].sum() / shape.sum()
        contrib[:, j, :] = betas[j] * shape
        truth_media[ch] = {
            "decay": m.decay,
            "half_saturation_spend": round(k_nat, 2),
            "beta": round(float(betas[j]), 2),
            "roas": round(float(contrib[:, j, :].sum() / spend[:, j, :].sum()), 4),
        }

    common = rng.normal(0, 0.03, n)
    idio = rng.normal(0, 0.05, (n, g))
    expected = base[:, None] * w[None, :] * geo_factor[None, :] + contrib.sum(axis=1)
    sales = expected * np.exp(common[:, None] + idio)

    # The geo test's ground truth: sales the test geos would have had with DSP left on
    k_dsp = MEDIA["DSP"].half_sat * plan[:, dsp].mean()
    a_cf = adstock(planned_dsp[:, test_geos], MEDIA["DSP"].decay)
    cf = betas[dsp] * w[test_geos] * hill(a_cf / w[test_geos], k_dsp)
    lost = float((cf[hold] - contrib[hold][:, dsp, test_geos]).sum())
    saved = float(planned_dsp[np.ix_(hold, test_geos)].sum())

    # Business-as-usual plan: average daily spend in the 13 weeks before the test
    bau = slice(n - cfg.holdout_weeks * 7 - 91, n - cfg.holdout_weeks * 7)
    for j, ch in enumerate(CHANNELS):
        share = spend[bau, j, :].sum(axis=0) / spend[bau, j, :].sum()
        x = float(plan[bau, j].mean())
        truth_media[ch]["bau_daily_spend"] = round(x, 2)
        truth_media[ch]["bau_geo_share"] = [round(float(s), 6) for s in share]
        gain = true_response(ch, truth_media, x * 1.01, w) - true_response(ch, truth_media, x, w)
        truth_media[ch]["mroas_at_bau"] = round(gain / (x * 0.01), 4)

    dates = cal["date"]
    geo_ids = [f"G{i + 1:02d}" for i in range(g)]
    cpm = {"SP": 9.0, "SB": 11.0, "SD": 6.0, "DSP": 7.5}
    media_rows = []
    for j, ch in enumerate(CHANNELS):
        s = spend[:, j, :]
        imps = s / cpm[ch] * 1000 * rng.lognormal(0, 0.05, s.shape)
        media_rows.append(
            pd.DataFrame(
                {
                    "date": np.repeat(dates.to_numpy(), g),
                    "geo_id": np.tile(geo_ids, n),
                    "channel": ch,
                    "spend": s.ravel().round(2),
                    "impressions": imps.ravel().round().astype(np.int64),
                }
            )
        )
    daily_geo_media = pd.concat(media_rows, ignore_index=True)
    aov = 31.0 * rng.lognormal(0, 0.02, sales.shape)
    daily_geo_sales = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), g),
            "geo_id": np.tile(geo_ids, n),
            "sales": sales.ravel().round(2),
            "orders": (sales / aov).ravel().round().astype(np.int64),
        }
    )
    geos = pd.DataFrame(
        {
            "geo_id": geo_ids,
            "geo_name": [f"Geo {i + 1:02d}" for i in range(g)],
            "region": [REGIONS[i % len(REGIONS)] for i in range(g)],
            "population_share": w.round(4),
        }
    )
    start, end = dates.iloc[hold[0]].date(), dates.iloc[hold[-1]].date()
    geo_test = pd.DataFrame(
        {
            "geo_id": geo_ids,
            "test_group": ["holdout" if i in set(test_geos) else "control" for i in range(g)],
            "channel": "DSP",
            "test_start": start,
            "test_end": end,
        }
    )
    calendar = cal[["date", "event", "new_year", "coupon"]].copy()
    truth = {
        "media": truth_media,
        "geo_test": {
            "channel": "DSP",
            "test_geos": [geo_ids[i] for i in test_geos],
            "start": str(start),
            "end": str(end),
            "lost_sales": round(lost, 2),
            "spend_saved": round(saved, 2),
            "iroas": round(lost / saved, 4),
        },
        "geo_weights": [round(float(x), 6) for x in w],
    }
    tables = {
        "geos": geos,
        "geo_test": geo_test,
        "calendar": calendar,
        "daily_geo_media": daily_geo_media,
        "daily_geo_sales": daily_geo_sales,
    }
    return tables, truth


def true_response(channel: str, media: dict[str, Any], daily_spend: float, w: np.ndarray) -> float:
    """True steady-state daily incremental sales at a national daily spend (geo-summed)."""
    t = media[channel]
    share = np.asarray(t.get("bau_geo_share", w))
    return float((t["beta"] * w * hill(daily_spend * share / w, t["half_saturation_spend"])).sum())


# --------------------------------------------------------------------------- journey layer


def _shapley_truth(touches: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Exact per-shopper Shapley split of the *expected* purchases ads caused.

    v(S) = 1 - (1 - base) * prod_{c in S} (1 - lift_c)^touches_c is the purchase probability if
    only the channels in S had run. The Shapley value shares v(all) - v(none) fairly between
    channels. Returns shoppers x channels.
    """
    k = len(CHANNELS)
    log_keep = touches * np.log1p(-np.array([JOURNEY[c].lift_per_touch for c in CHANNELS]))
    v = {}
    for mask in range(1 << k):
        cols = [j for j in range(k) if mask >> j & 1]
        v[mask] = 1.0 - (1.0 - base) * np.exp(log_keep[:, cols].sum(axis=1))
    fact = [1, 1, 2, 6, 24, 120]
    phi = np.zeros((len(base), k))
    for j in range(k):
        for mask in range(1 << k):
            if mask >> j & 1:
                continue
            s = bin(mask).count("1")
            weight = fact[s] * fact[k - s - 1] / fact[k]
            phi[:, j] += weight * (v[mask | (1 << j)] - v[mask])
    return phi


def _journey_layer(
    cfg: SynthConfig, start: pd.Timestamp, rng: np.random.Generator
) -> tuple[dict[str, pd.DataFrame], dict]:
    n = cfg.n_shoppers
    in_market = rng.random(n) < 0.25
    prior_buyer = rng.random(n) < np.where(in_market, 0.12, 0.03)
    base = np.where(in_market, 0.075, 0.010) * rng.lognormal(0, 0.25, n)
    base = np.clip(base, 0, 0.5)

    touches = np.zeros((n, len(CHANNELS)), dtype=np.int64)
    for j, ch in enumerate(CHANNELS):
        jt = JOURNEY[ch]
        p = np.where(in_market, jt.reach_in_market, jt.reach_other)
        if ch == "DSP":
            p = np.where(prior_buyer, p * 0.6, p)  # prospecting excludes many past buyers
        reached = rng.random(n) < p
        touches[:, j] = reached * (1 + rng.poisson(jt.extra_touches, n))

    lift = np.array([JOURNEY[c].lift_per_touch for c in CHANNELS])
    p_buy = 1.0 - (1.0 - base) * np.prod((1.0 - lift) ** touches, axis=1)
    buyer = rng.random(n) < p_buy
    phi = _shapley_truth(touches, base)

    user_ids = np.array([f"u{i:07d}" for i in range(n)])
    journey_start = rng.uniform(0, cfg.journey_days - 30, n)

    # ad events
    ev_user, ev_channel, ev_day = [], [], []
    for j, ch in enumerate(CHANNELS):
        k = touches[:, j]
        users = np.repeat(np.arange(n), k)
        lo, hi = JOURNEY[ch].window
        day = journey_start[users] + rng.uniform(lo, hi, len(users))
        ev_user.append(users)
        ev_channel.append(np.full(len(users), j))
        ev_day.append(day)
    eu, ec, ed = np.concatenate(ev_user), np.concatenate(ev_channel), np.concatenate(ev_day)

    # purchases come after the shopper's last ad; un-touched buyers buy any time in the journey
    last_touch = np.full(n, -1.0)
    np.maximum.at(last_touch, eu, ed)
    buy_day = np.where(
        last_touch >= 0,
        last_touch + rng.uniform(0.05, 3.0, n),
        journey_start + rng.uniform(0, 30, n),
    )
    buy_day = np.minimum(buy_day, cfg.journey_days - 0.01)

    def ts(days: np.ndarray) -> np.ndarray:
        secs = np.floor(days * 86400).astype("int64")
        return (start + pd.to_timedelta(secs, unit="s")).to_numpy()

    campaigns = {
        "DSP": ["DSP | Prospecting | Collagen", "DSP | Prospecting | Protein",
                "DSP | Lifestyle | Wellness"],
        "SP": ["SP | Auto | All products", "SP | Exact | Collagen", "SP | Exact | Protein",
               "SP | Category | Vitamins"],
        "SB": ["SB | Store spotlight", "SB | Video | Protein"],
        "SD": ["SD | Views remarketing", "SD | Purchases remarketing"],
    }  # fmt: skip
    camp_ids = {name: f"cmp_{i + 1:03d}" for i, name in enumerate(sum(campaigns.values(), []))}

    dsp_j = CHANNELS.index("DSP")
    m = ec == dsp_j
    dsp_camp = rng.choice(campaigns["DSP"], m.sum(), p=[0.4, 0.35, 0.25])
    dsp_impressions = pd.DataFrame(
        {
            "user_id": user_ids[eu[m]],
            "impression_dt_utc": ts(ed[m]),
            "campaign_id": [camp_ids[c] for c in dsp_camp],
            "campaign": dsp_camp,
            "impressions": 1,
            "total_cost": (rng.lognormal(np.log(7.5), 0.15, m.sum()) / 1000).round(5),
        }
    ).sort_values("impression_dt_utc", ignore_index=True)

    sp_rows = []
    ctr = {"SP": 0.30, "SB": 0.18, "SD": 0.06}
    cpc = {"SP": 1.10, "SB": 1.35, "SD": 0.85}
    for ch in ("SP", "SB", "SD"):
        m = ec == CHANNELS.index(ch)
        names = rng.choice(campaigns[ch], m.sum())
        clicks = (rng.random(m.sum()) < ctr[ch]).astype(np.int64)
        sp_rows.append(
            pd.DataFrame(
                {
                    "user_id": user_ids[eu[m]],
                    "event_dt_utc": ts(ed[m]),
                    "ad_product_type": AD_PRODUCT_TYPES[ch],
                    "campaign_id": [camp_ids[c] for c in names],
                    "campaign": names,
                    "impressions": 1,
                    "clicks": clicks,
                    "spend": (clicks * rng.lognormal(np.log(cpc[ch]), 0.2, m.sum())).round(2),
                }
            )
        )
    sponsored_ads_traffic = pd.concat(sp_rows, ignore_index=True).sort_values(
        "event_dt_utc", ignore_index=True
    )

    b = np.flatnonzero(buyer)
    repeat = b[rng.random(len(b)) < 0.12]
    repeat_day = buy_day[repeat] + rng.uniform(7, 20, len(repeat))
    keep = repeat_day < cfg.journey_days
    conv_user = np.concatenate([b, repeat[keep]])
    conv_day = np.concatenate([buy_day[b], repeat_day[keep]])
    first = np.concatenate([np.ones(len(b), bool), np.zeros(keep.sum(), bool)])
    asin = rng.choice(len(ASINS), len(conv_user), p=ASINS["sales_mix"].to_numpy())
    qty = 1 + (rng.random(len(conv_user)) < 0.15)
    conversions = pd.DataFrame(
        {
            "user_id": user_ids[conv_user],
            "conversion_event_dt_utc": ts(conv_day),
            "tracked_asin": ASINS["tracked_asin"].to_numpy()[asin],
            "purchases": qty,
            "product_sales": (qty * ASINS["list_price"].to_numpy()[asin]).round(2),
            "new_to_brand": first & ~prior_buyer[conv_user],
        }
    ).sort_values("conversion_event_dt_utc", ignore_index=True)

    incremental = phi.sum(axis=0)
    truth = {
        "window_start": str(start.date()),
        "window_end": str((start + pd.Timedelta(days=cfg.journey_days - 1)).date()),
        "shoppers": n,
        "expected_purchasers": round(float(p_buy.sum()), 1),
        "expected_baseline_purchasers": round(float(base.sum()), 1),
        "incremental_purchasers": {
            c: round(float(incremental[j]), 1) for j, c in enumerate(CHANNELS)
        },
        "incremental_share": {
            c: round(float(incremental[j] / incremental.sum()), 4) for j, c in enumerate(CHANNELS)
        },
    }
    tables = {
        "dsp_impressions": dsp_impressions,
        "sponsored_ads_traffic": sponsored_ads_traffic,
        "conversions": conversions,
    }
    return tables, truth


# --------------------------------------------------------------------------- entry points


def generate(cfg: SynthConfig | None = None) -> SynthData:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    geo_tables, geo_truth = _geo_layer(cfg, rng)
    holdout_start = pd.Timestamp(geo_truth["geo_test"]["start"])
    journey_start = holdout_start - pd.Timedelta(days=cfg.journey_days)
    journey_tables, journey_truth = _journey_layer(cfg, journey_start, rng)
    tables = {"asins": ASINS.drop(columns="sales_mix"), **geo_tables, **journey_tables}
    truth = {
        "brand": f"{BRAND} (fictional)",
        "seed": cfg.seed,
        **geo_truth,
        "journeys": journey_truth,
    }
    return SynthData(tables=tables, truth=truth)


def write(data: SynthData, db_path: str | Path) -> Path:
    """Write all tables to a fresh DuckDB file and the ground truth next to it."""
    import duckdb

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp.duckdb")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        for name, df in data.tables.items():
            con.register("_df", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
            con.unregister("_df")
    finally:
        con.close()
    tmp.replace(db_path)  # atomic: readers never see a half-written database
    truth_path(db_path).write_text(json.dumps(data.truth, indent=2))
    return db_path


def truth_path(db_path: str | Path) -> Path:
    return Path(db_path).with_name("truth.json")
