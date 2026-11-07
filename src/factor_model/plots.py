"""Research charts (quintile bars, IC, correlation heatmap) in the HCCT Group style."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantcore.report import PALETTE  # noqa: E402

FOOT = "HCCT Group · Equity Factor Model"


def _finish(fig, path, title):
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.text(0.99, 0.005, FOOT, ha="right", fontsize=8, color="#7A7A7A")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def quintile_bars(qret: dict[str, pd.DataFrame], path, periods: int = 12):
    """Grouped bars: annualised mean return of quintiles Q1..Q5 for each factor."""
    names = list(qret)
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=110)
    n = 5
    width = 0.8 / n
    blues = plt.cm.RdYlGn(np.linspace(0.12, 0.88, n))
    x = np.arange(len(names))
    for q in range(1, n + 1):
        vals = [qret[k][q].mean() * periods for k in names]
        bars = ax.bar(x + (q - 3) * width, vals, width, color=blues[q - 1], edgecolor="white", label=f"Q{q}" + (" (low)" if q == 1 else " (high)" if q == n else ""))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.002 if v >= 0 else -0.002), f"{v*100:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    ax.set_xticks(x, names)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Annualised mean return (gross)")
    ax.axhline(0, color="k", lw=0.6)
    ax.grid(axis="y", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.set_title("Equal-weight quintile portfolios, formed monthly — a good factor rises from Q1 to Q5", loc="left", fontsize=10)
    return _finish(fig, path, "Quintile monotonicity")


def ic_chart(monthly_ic: dict[str, pd.Series], decay: dict[str, pd.DataFrame], path):
    """Left: rolling 12-month mean of monthly rank IC. Right: IC decay across horizons."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=110, gridspec_kw={"width_ratios": [2.2, 1]})
    for i, (k, s) in enumerate(monthly_ic.items()):
        a1.plot(s.rolling(12).mean(), color=PALETTE[i % len(PALETTE)], lw=2.0 if k == "Composite" else 1.3, label=f"{k} (mean {s.mean():+.3f})")
    a1.axhline(0, color="k", lw=0.6)
    a1.set_title("Rolling 12-month mean of monthly rank IC (21d forward)", loc="left", fontweight="bold", fontsize=10)
    a1.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=4)
    hz = list(next(iter(decay.values())).index)
    x = np.arange(len(hz)); w = 0.8 / len(decay)
    for i, (k, d) in enumerate(decay.items()):
        a2.bar(x + (i - (len(decay) - 1) / 2) * w, d["IC Mean"].values, w, color=PALETTE[i % len(PALETTE)], label=k)
    a2.set_xticks(x, [f"{h}d" for h in hz]); a2.axhline(0, color="k", lw=0.6)
    a2.set_title("IC decay: mean daily rank IC by horizon", loc="left", fontweight="bold", fontsize=10)
    a2.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
    for a in (a1, a2):
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    return _finish(fig, path, "Information coefficient")


def correlation_heatmaps(score_corr: pd.DataFrame, ret_corr: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=110)
    for ax, m, t in zip(axes, (score_corr, ret_corr), ("Avg cross-sectional rank corr. of scores", "Corr. of daily long/short returns")):
        im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1)
        labels = [c.replace(" ", "\n", 1) for c in m.columns]
        ax.set_xticks(range(len(m)), labels, fontsize=8); ax.set_yticks(range(len(m)), labels, fontsize=8)
        for (i, j), v in np.ndenumerate(m.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if abs(v) > 0.6 else "black")
        ax.set_title(t, loc="left", fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8)
    return _finish(fig, path, "Factor correlation")
