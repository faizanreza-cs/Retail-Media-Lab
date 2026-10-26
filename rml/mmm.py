"""Marketing mix model: daily sales explained by baseline, calendar and ad spend.

sales[t] = intercept + trend + seasonality + weekday + retail events + coupons
           + sum over channels of  beta_c * hill(adstock(spend_c, decay_c), half_sat_c)

* **Adstock** carries part of today's spend into the next days (normalised geometric decay).
* **Hill saturation** makes each extra dollar worth less as spend grows.
* Coefficients come from a ridge least-squares fit with ad effects constrained to be >= 0
  (``scipy.optimize.lsq_linear``). Decay and saturation per channel are chosen by random search,
  scored with time-series cross-validation (``sklearn.model_selection.TimeSeriesSplit``).
* Many decay/saturation settings fit the data almost equally well, and they disagree most on
  always-on channels whose spend barely varies. So the model keeps the ``n_models`` best
  settings (an ensemble), and the intervals combine that model uncertainty with a moving-block
  bootstrap of the residuals. A wide interval means "the data can't tell": run an experiment.

Outputs per channel: average ROAS, marginal ROAS at the current spend, and a response curve
(daily spend -> daily incremental sales, averaged over the ensemble) for the budget optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from rml.synth import CHANNELS, adstock, hill

DECAY_RANGE = (0.0, 0.9)
HALF_SAT_RANGE = (0.2, 4.0)  # multiples of the channel's mean daily spend
RIDGE = 1.0


def national_daily(
    daily_geo_sales: pd.DataFrame, daily_geo_media: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """One row per day: sales, spend per channel and the known calendar effects."""
    sales = daily_geo_sales.groupby("date", as_index=False)["sales"].sum()
    spend = daily_geo_media.pivot_table(
        index="date", columns="channel", values="spend", aggfunc="sum"
    ).reset_index()
    df = sales.merge(spend, on="date").merge(calendar, on="date")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date", ignore_index=True)


def _controls(df: pd.DataFrame) -> pd.DataFrame:
    t = np.arange(len(df)) / 365.0
    doy = df["date"].dt.day_of_year.to_numpy()
    c = pd.DataFrame({"trend": t}, index=df.index)
    for k in (1, 2):
        c[f"sin{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        c[f"cos{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    for d in range(1, 7):
        c[f"weekday_{d}"] = (df["date"].dt.day_of_week == d).astype(float)
    for name in sorted(e for e in df["event"].unique() if e):
        c[f"event: {name}"] = (df["event"] == name).astype(float)
    c["new_year"] = df["new_year"]
    c["coupon"] = (df["coupon"] > 0).astype(float)
    return c


@dataclass
class Params:
    decay: dict[str, float]
    half_sat: dict[str, float]  # absolute daily spend at half saturation


def media_features(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    return pd.DataFrame(
        {ch: hill(adstock(df[ch].to_numpy(), p.decay[ch]), p.half_sat[ch]) for ch in CHANNELS},
        index=df.index,
    )


def _solve(X: np.ndarray, y: np.ndarray, n_media: int) -> np.ndarray:
    """Ridge least squares on standardised columns; media (last n_media) coefficients >= 0."""
    scale = X.std(axis=0)
    scale[0] = 1.0  # intercept
    scale[scale == 0] = 1.0
    Xs = X / scale
    k = X.shape[1]
    pen = np.sqrt(RIDGE) * np.eye(k)[1:]  # don't penalise the intercept
    A = np.vstack([Xs, pen])
    b = np.concatenate([y, np.zeros(k - 1)])
    lower = np.full(k, -np.inf)
    lower[k - n_media :] = 0.0
    res = lsq_linear(A, b, bounds=(lower, np.full(k, np.inf)), method="bvls")
    return np.asarray(res.x / scale)


def _design(df: pd.DataFrame, p: Params) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), _controls(df), media_features(df, p)])


def _cv_mape(df: pd.DataFrame, p: Params, splits: TimeSeriesSplit) -> float:
    X, y = _design(df, p), df["sales"].to_numpy()
    errs = []
    for tr, te in splits.split(X):
        coef = _solve(X[tr], y[tr], len(CHANNELS))
        errs.append(mean_absolute_percentage_error(y[te], X[te] @ coef))
    return float(np.mean(errs))


@dataclass
class Member:
    params: Params
    beta: dict[str, float]  # media coefficients
    cv_mape: float


@dataclass
class Curves:
    """Response curves of the ensemble: what a constant daily spend returns per day."""

    members: list[Member]

    def response(self, channel: str, daily_spend: np.ndarray | float) -> np.ndarray:
        """Steady-state daily incremental sales at a constant daily spend (ensemble mean)."""
        x = np.asarray(daily_spend, dtype=float)
        return np.mean(
            [m.beta[channel] * hill(x, m.params.half_sat[channel]) for m in self.members], axis=0
        )

    def marginal_roas(self, channel: str, daily_spend: float) -> float:
        """Extra daily sales per extra dollar of daily spend (derivative of the response)."""
        return float(
            np.mean(
                [
                    m.beta[channel]
                    * m.params.half_sat[channel]
                    / (daily_spend + m.params.half_sat[channel]) ** 2
                    for m in self.members
                ]
            )
        )

    def to_dict(self) -> list[dict]:
        return [
            {"decay": m.params.decay, "half_sat": m.params.half_sat, "beta": m.beta,
             "cv_mape": m.cv_mape}
            for m in self.members
        ]  # fmt: skip

    @classmethod
    def from_dict(cls, rows: list[dict]) -> Curves:
        return cls(
            [Member(Params(r["decay"], r["half_sat"]), r["beta"], r["cv_mape"]) for r in rows]
        )


def days_before(df: pd.DataFrame, start: pd.Timestamp, days: int = 91) -> slice:
    """Rows of the ``days`` days before ``start``: "current" spend, ahead of the geo test."""
    before = df.index[df["date"] < pd.Timestamp(start)]
    if len(before) < days:
        raise ValueError(f"need {days} days of data before {start:%Y-%m-%d}")
    return slice(int(before[-days]), int(before[-1]) + 1)


@dataclass
class MMMResult:
    curves: Curves
    fit: dict[str, float]
    channels: pd.DataFrame  # per-channel ROAS, mROAS and intervals
    decomposition: pd.DataFrame  # date, sales, fitted, baseline, one column per channel


def fit(
    df: pd.DataFrame,
    n_search: int = 500,
    n_models: int = 20,
    n_boot: int = 10,
    seed: int = 0,
    bau_days: slice | None = None,
) -> MMMResult:
    """Fit the ensemble. ``bau_days`` selects the days whose average spend counts as current."""
    rng = np.random.default_rng(seed)
    mean_spend = {ch: float(df[ch].mean()) for ch in CHANNELS}
    splits = TimeSeriesSplit(n_splits=4, test_size=56)
    candidates = []
    for _ in range(n_search):
        p = Params(
            decay={ch: float(rng.uniform(*DECAY_RANGE)) for ch in CHANNELS},
            half_sat={ch: float(rng.uniform(*HALF_SAT_RANGE)) * mean_spend[ch] for ch in CHANNELS},
        )
        candidates.append((_cv_mape(df, p, splits), p))
    candidates.sort(key=lambda c: c[0])

    y = df["sales"].to_numpy()
    n, k = len(y), len(CHANNELS)
    bau = bau_days or slice(n - 91, n)
    x_bau = {ch: float(df[ch].iloc[bau].mean()) for ch in CHANNELS}
    spend = {ch: float(df[ch].sum()) for ch in CHANNELS}
    xb = np.array([x_bau[ch] for ch in CHANNELS])
    tot = np.array([spend[ch] for ch in CHANNELS])
    members, fitted_all, contrib_all = [], [], []
    point: dict[str, list[np.ndarray]] = {"roas": [], "mroas": []}
    draws: dict[str, list[np.ndarray]] = {"roas": [], "mroas": []}
    block = 14
    for score, p in candidates[:n_models]:
        X = _design(df, p)
        coef = _solve(X, y, k)
        media = media_features(df, p).to_numpy()
        fitted = X @ coef
        fitted_all.append(fitted)
        contrib_all.append(media * coef[-k:])
        members.append(Member(p, dict(zip(CHANNELS, (float(b) for b in coef[-k:]), strict=True)),
                              score))  # fmt: skip
        half = np.array([p.half_sat[ch] for ch in CHANNELS])
        betas = [coef[-k:]]
        resid = y - fitted
        for _ in range(n_boot):  # same settings, refit on resampled residual blocks
            starts = rng.integers(0, n - block, n // block + 1)
            e = np.concatenate([resid[s : s + block] for s in starts])[:n]
            betas.append(_solve(X, fitted + e, k)[-k:])
        for i, beta in enumerate(betas):
            roas = (media * beta).sum(axis=0) / tot
            mroas = beta * half / (xb + half) ** 2
            target = point if i == 0 else draws
            target["roas"].append(roas)
            target["mroas"].append(mroas)

    # ensemble means: the same numbers as incremental_sales / spend and as Curves (also means)
    roas_pt, mroas_pt = np.mean(point["roas"], axis=0), np.mean(point["mroas"], axis=0)
    pool = {s: np.vstack(point[s] + draws[s]) for s in point}
    contrib = np.mean(contrib_all, axis=0)
    rows = []
    for j, ch in enumerate(CHANNELS):
        lo, hi = np.quantile(pool["roas"][:, j], [0.05, 0.95])
        mlo, mhi = np.quantile(pool["mroas"][:, j], [0.05, 0.95])
        rows.append(
            {
                "channel": ch,
                "spend": round(spend[ch], 2),
                "incremental_sales": round(float(contrib[:, j].sum()), 2),
                "roas": round(float(roas_pt[j]), 4),
                "roas_low": round(float(lo), 4),
                "roas_high": round(float(hi), 4),
                "bau_daily_spend": round(x_bau[ch], 2),
                "mroas_at_bau": round(float(mroas_pt[j]), 4),
                "mroas_low": round(float(mlo), 4),
                "mroas_high": round(float(mhi), 4),
            }
        )
    fitted = np.mean(fitted_all, axis=0)
    decomposition = pd.DataFrame(
        {"date": df["date"], "sales": y, "fitted": fitted, "baseline": fitted - contrib.sum(axis=1)}
    )
    for j, ch in enumerate(CHANNELS):
        decomposition[ch] = contrib[:, j]
    return MMMResult(
        curves=Curves(members),
        fit={
            "r2": round(float(r2_score(y, fitted)), 4),
            "mape": round(float(mean_absolute_percentage_error(y, fitted)), 4),
            "cv_mape": round(float(np.mean([m.cv_mape for m in members])), 4),
            "models": len(members),
            "searched": n_search,
        },
        channels=pd.DataFrame(rows),
        decomposition=decomposition,
    )
