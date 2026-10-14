"""README charts, rendered from ``results.json`` with matplotlib.

Palette: a validated categorical order (adjacent pairs pass colour-blind separation); slots 3-4
are below 3:1 contrast on the light surface, so every chart carries a legend and direct labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from rml import pipeline  # noqa: E402
from rml.synth import CHANNEL_NAMES, CHANNELS  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
)  # fmt: skip
CHANNEL_COLOR = dict(zip(CHANNELS, SLOTS, strict=False))


def _axes(title: str, subtitle: str, size: tuple[float, float] = (8.0, 4.4)) -> tuple:
    fig, ax = plt.subplots(figsize=size, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8.5, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(0.012, 0.965, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=9, color=INK2, va="top")
    return fig, ax


def _save(fig: Any, path: Path) -> Path:
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def attribution_chart(res: dict, path: Path) -> Path:
    t = pd.DataFrame(res["attribution"]["table"])
    wide = t.pivot(index="channel", columns="method", values="share").reindex(list(CHANNELS))
    methods = [("last_touch", "Last touch"), ("linear", "Linear"), ("markov", "Markov"),
               ("shapley", "Shapley")]  # fmt: skip
    fig, ax = _axes(
        "Last touch over-credits Sponsored Products",
        "Share of credited purchases by attribution method, against the true causal share",
    )
    x = np.arange(len(CHANNELS))
    width = 0.19
    for i, (m, label) in enumerate(methods):
        ax.bar(x + (i - 1.5) * width, wide[m], width, color=SLOTS[i], label=label,
               edgecolor=SURFACE, linewidth=1.2)  # fmt: skip
    for xi, ch in zip(x, CHANNELS, strict=True):
        ax.hlines(wide.loc[ch, "truth"], xi - 2.2 * width, xi + 2.2 * width, color=INK, lw=2)
    sp = list(CHANNELS).index("SP")
    ax.annotate(f"{wide.loc['SP', 'last_touch']:.0%}", (x[sp] - 1.5 * width,
                wide.loc["SP", "last_touch"]), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK2)  # fmt: skip
    ax.annotate(f"truth {wide.loc['SP', 'truth']:.0%}", (x[sp] + 2.2 * width,
                wide.loc["SP", "truth"]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=INK)  # fmt: skip
    ax.set_xticks(x, [CHANNEL_NAMES[c] for c in CHANNELS])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK, lw=2))
    labels.append("True causal share")
    ax.legend(handles, labels, ncol=5, frameon=False, fontsize=8.5, labelcolor=INK2,
              loc="upper center", bbox_to_anchor=(0.5, -0.09))  # fmt: skip
    ax.set_ylim(0, max(wide.max()) * 1.12)
    return _save(fig, path)


def geo_test_chart(res: dict, path: Path) -> Path:
    inc = res["incrementality"]
    w = pd.DataFrame(inc["weekly"])
    w["week"] = pd.to_datetime(w["week"])
    sc = next(r for r in inc["results"] if r["method"] == "synthetic_control")
    fig, ax = _axes(
        f"Switching DSP off cut test-geo sales by {abs(sc['lift_pct']):.1%}",
        f"Weekly sales in the {len(inc['test_geos'])} holdout geos vs a synthetic control built "
        "from the control geos",
    )
    start = pd.Timestamp(inc["start"])
    ax.axvspan(start - pd.Timedelta(days=3.5), w["week"].max() + pd.Timedelta(days=3.5),
               color="#f0efec", lw=0)  # fmt: skip
    for col, color, label in (("synthetic", SLOTS[1], "Synthetic control"),
                              ("actual", SLOTS[0], "Holdout geos (actual)")):  # fmt: skip
        ax.plot(w["week"], w[col] / 1e3, color=color, lw=1.8, label=label)
        ax.annotate(label, (w["week"].iloc[-1], w[col].iloc[-1] / 1e3), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)  # fmt: skip
    ax.set_ylabel("Sales, $ thousands", color=INK2, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    ax.text(start, ax.get_ylim()[1], "  DSP off", va="top", fontsize=8.5, color=INK2)
    return _save(fig, path)


def response_chart(res: dict, path: Path) -> Path:
    curves = pipeline.curves(res)
    plan = {r["channel"]: r for r in res["budget"]["channels"]}
    fig, ax = _axes(
        "The next dollar earns most in DSP",
        "Daily incremental sales vs daily spend (MMM ensemble); the slope is the marginal ROAS. "
        "Open circle: current, filled: recommended.",
    )
    top = max(r["current_daily_spend"] for r in plan.values()) * 1.6
    xs = np.linspace(0, top, 200)
    for ch in CHANNELS:
        c = CHANNEL_COLOR[ch]
        y = curves.response(ch, xs) / 1e3
        ax.plot(xs / 1e3, y, color=c, lw=1.8, label=CHANNEL_NAMES[ch])
        cur, rec = plan[ch]["current_daily_spend"], plan[ch]["recommended_daily_spend"]
        ax.plot(cur / 1e3, curves.response(ch, cur) / 1e3, "o", ms=7, mfc=SURFACE, mec=c, mew=1.8)
        ax.plot(rec / 1e3, curves.response(ch, rec) / 1e3, "o", ms=7, color=c, mec=SURFACE,
                mew=1.5)  # fmt: skip
        ax.annotate(CHANNEL_NAMES[ch], (xs[-1] / 1e3, y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)  # fmt: skip
    ax.set_xlabel("Daily spend, $ thousands", color=INK2, fontsize=8.5)
    ax.set_ylabel("Daily incremental sales, $ thousands", color=INK2, fontsize=8.5)
    ax.set_xlim(0, top / 1e3)
    ax.set_ylim(0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    return _save(fig, path)


def roas_chart(res: dict, path: Path) -> Path:
    rows = {r["channel"]: r for r in res["mmm"]["channels"]}
    did = next(r for r in res["incrementality"]["results"] if r["method"].startswith("diff"))
    fig, ax = _axes(
        "The model pins down DSP, not Sponsored Products",
        "Average ROAS over two years: MMM median with 90% interval, the DSP geo test, the truth",
    )
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    order = list(reversed(CHANNELS))
    for i, ch in enumerate(order):
        r = rows[ch]
        ax.hlines(i, r["roas_low"], r["roas_high"], color=SLOTS[0], lw=2)
        ax.plot(r["roas"], i, "o", color=SLOTS[0], ms=8, mec=SURFACE, mew=1.5)
        if "true_roas" in r:
            ax.plot(r["true_roas"], i, "D", color=INK, ms=6.5, mec=SURFACE, mew=1.2)
    k = order.index("DSP")
    lo, hi = did["iroas_ci"]
    ax.hlines(k - 0.22, lo, hi, color=SLOTS[1], lw=2)
    ax.plot(did["iroas"], k - 0.22, "s", color=SLOTS[1], ms=7, mec=SURFACE, mew=1.2)
    ax.set_yticks(range(len(order)), [CHANNEL_NAMES[c] for c in order])
    ax.set_xlabel(r"ROAS (\$ incremental sales per \$1 of spend)", color=INK2, fontsize=8.5)
    ax.set_xlim(0, max(r["roas_high"] for r in rows.values()) * 1.1)
    ax.set_ylim(-0.6, len(order) - 0.4)
    handles = [
        Line2D([], [], color=SLOTS[0], lw=2, marker="o", ms=7, label="MMM (90% interval)"),
        Line2D([], [], color=SLOTS[1], lw=2, marker="s", ms=6, label="DSP geo test iROAS (95%)"),
        Line2D([], [], color=INK, lw=0, marker="D", ms=6, label="True ROAS"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    return _save(fig, path)


def render(out: str | Path = "docs/img") -> list[Path]:
    res = pipeline.load_results()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return [
        attribution_chart(res, out / "attribution_vs_truth.png"),
        geo_test_chart(res, out / "geo_test.png"),
        roas_chart(res, out / "roas_vs_truth.png"),
        response_chart(res, out / "response_curves.png"),
    ]
