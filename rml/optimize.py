"""Budget allocation: the daily spend per channel that maximises incremental sales.

Uses the MMM's response curves. The default keeps the total budget and lets each channel move
at most ``max_change`` (30%) from its current share of it, a guardrail marketing teams use so
one model run can't swing a plan overnight. An optional minimum ROAS is supported. Solved with
SLSQP (``scipy.optimize.minimize``). At the optimum, channels that aren't at a limit end up with
the same marginal ROAS: the next dollar earns the same everywhere.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from scipy.optimize import minimize

from rml.synth import CHANNELS


class ResponseModel(Protocol):
    def response(self, channel: str, daily_spend: np.ndarray | float) -> np.ndarray: ...

    def marginal_roas(self, channel: str, daily_spend: float) -> float: ...


def optimize(
    model: ResponseModel,
    current: dict[str, float],
    total: float | None = None,
    max_change: float = 0.3,
    min_roas: float | None = None,
) -> dict[str, Any]:
    if not 0 <= max_change <= 1:
        raise ValueError("max_change must be between 0 and 1")
    x0 = np.array([float(current[c]) for c in CHANNELS])
    total = float(total) if total is not None else float(x0.sum())
    if total <= 0:
        raise ValueError("total budget must be positive")
    base = x0 * total / x0.sum()  # current mix, scaled to the new total
    lo, hi = base * (1 - max_change), base * (1 + max_change)

    def sales(x: np.ndarray) -> float:
        return float(sum(model.response(c, x[i]) for i, c in enumerate(CHANNELS)))

    # Solve for u = spend / current spend (around 1) rather than dollars: SLSQP's
    # finite-difference steps are absolute, too small to move sales at dollar scale.
    scale = max(sales(base), 1.0)
    cons: list[dict[str, Any]] = [{"type": "eq", "fun": lambda u: (u @ base - total) / total}]
    if min_roas is not None:
        cons.append(
            {"type": "ineq", "fun": lambda u: (sales(u * base) - min_roas * (u @ base)) / scale}
        )
    res = minimize(
        lambda u: -sales(u * base) / scale,
        np.ones_like(base),
        method="SLSQP",
        bounds=[(1 - max_change, 1 + max_change)] * len(base),
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    x = np.clip(res.x * base, lo, hi)
    feasible = abs(x.sum() - total) <= 1e-6 * total and (
        min_roas is None or sales(x) >= min_roas * x.sum() * (1 - 1e-6)
    )
    if not (res.success and feasible):
        raise ValueError(f"no plan meets the constraints ({res.message})")

    rows = []
    for i, c in enumerate(CHANNELS):
        rows.append(
            {
                "channel": c,
                "current_daily_spend": round(float(x0[i]), 2),
                "recommended_daily_spend": round(float(x[i]), 2),
                "change_pct": round(float(x[i] / x0[i] - 1), 4),
                "current_incremental_sales": round(float(model.response(c, x0[i])), 2),
                "recommended_incremental_sales": round(float(model.response(c, x[i])), 2),
                "mroas_current": round(model.marginal_roas(c, float(x0[i])), 4),
                "mroas_recommended": round(model.marginal_roas(c, float(x[i])), 4),
            }
        )
    current_sales, new_sales = sales(x0), sales(x)
    return {
        "total_daily_budget": round(total, 2),
        "max_change": max_change,
        "min_roas": min_roas,
        "channels": rows,
        "current_daily_incremental_sales": round(current_sales, 2),
        "recommended_daily_incremental_sales": round(new_sales, 2),
        "uplift_pct": round(new_sales / current_sales - 1, 4),
        "recommended_roas": round(new_sales / total, 4),
    }


class TrueCurves:
    """The simulator's true response curves (only exists for synthetic data)."""

    def __init__(self, truth_media: dict, geo_weights: list[float]) -> None:
        self.media = truth_media
        self.w = np.asarray(geo_weights)

    def response(self, channel: str, daily_spend: np.ndarray | float) -> np.ndarray:
        from rml.synth import true_response

        x = np.atleast_1d(np.asarray(daily_spend, dtype=float))
        out = np.array([true_response(channel, self.media, float(v), self.w) for v in x])
        return out if np.ndim(daily_spend) else out[0]

    def marginal_roas(self, channel: str, daily_spend: float) -> float:
        h = max(daily_spend * 1e-4, 1e-3)
        up = float(self.response(channel, daily_spend + h))
        down = float(self.response(channel, max(daily_spend - h, 0.0)))
        return (up - down) / (daily_spend + h - max(daily_spend - h, 0.0))


def score_with_truth(plan: dict[str, Any], truth: TrueCurves) -> dict[str, Any]:
    """True incremental sales of the current and the recommended plan, and the best possible
    plan under the same constraints (what a perfect model would recommend)."""
    cur = {r["channel"]: r["current_daily_spend"] for r in plan["channels"]}
    rec = {r["channel"]: r["recommended_daily_spend"] for r in plan["channels"]}
    true_cur = float(sum(truth.response(c, x) for c, x in cur.items()))
    true_rec = float(sum(truth.response(c, x) for c, x in rec.items()))
    oracle = optimize(truth, cur, plan["total_daily_budget"], plan["max_change"], plan["min_roas"])
    best = oracle["recommended_daily_incremental_sales"]
    gain = true_rec - true_cur
    return {
        "current_daily_incremental_sales": round(true_cur, 2),
        "recommended_daily_incremental_sales": round(true_rec, 2),
        "uplift_pct": round(gain / true_cur, 4),
        "best_possible_uplift_pct": round(best / true_cur - 1, 4),
        "share_of_best_captured": round(gain / (best - true_cur), 4) if best > true_cur else None,
        "best_plan": {r["channel"]: r["recommended_daily_spend"] for r in oracle["channels"]},
    }
