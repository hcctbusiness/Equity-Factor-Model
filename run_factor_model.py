"""End-to-end factor research on real data (Yahoo Finance + Ken French library).

Run:  python run_factor_model.py
Writes docs/*.png and docs/results.md. Every number in the README comes from here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import quantcore as qc
from factor_model import analytics as an
from factor_model import attribution as at
from factor_model import factors as fc
from factor_model import plots
from factor_model import portfolio as pf
from factor_model.universe import TICKERS

DOCS = Path(__file__).resolve().parent / "docs"
START = "2010-01-01"           # research window start (2009 is warm-up for 12-1 momentum)
COSTS = qc.CostModel(commission_bps=1, slippage_bps=2, borrow_bps_annual=50)


def pct(x, d=2):
    return "—" if pd.isna(x) else f"{x*100:.{d}f}%"


def num(x, d=2):
    return "—" if pd.isna(x) else f"{x:.{d}f}"


def md_table(df: pd.DataFrame, fmt) -> str:
    head = "| " + " | ".join([df.index.name or ""] + list(map(str, df.columns))) + " |"
    sep = "|---|" + "---:|" * len(df.columns)
    rows = ["| " + " | ".join([str(i)] + [fmt(c, v) for c, v in r.items()]) + " |" for i, r in df.iterrows()]
    return "\n".join([head, sep, *rows])


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    px = qc.data.prices(TICKERS, start="2009-01-01").ffill(limit=5)
    px = px[[t for t in TICKERS if t in px.columns]]
    print(f"Universe: {px.shape[1]} names, {px.index[0].date()} -> {px.index[-1].date()}")

    scores = fc.build_factors(px)
    names = list(scores)
    scores = {k: v.loc[START:] for k, v in scores.items()}
    pxr = px.loc[START:]

    # ---------------- IC analysis ----------------
    me = an.month_end_dates(pxr.index)
    px_m = pxr.reindex(me)
    fwd_m = px_m.shift(-1) / px_m - 1
    monthly_ic = {k: an.rank_ic(scores[k].reindex(me), fwd_m) for k in names}
    ic_tab = pd.DataFrame({k: an.ic_summary(monthly_ic[k], 1, periods_per_year=12) for k in names}).T
    decay = {k: an.ic_decay(scores[k], pxr, (1, 5, 21, 63)) for k in names}
    decay_tab = pd.DataFrame({k: d["IC Mean"] for k, d in decay.items()}).T
    decay_tab.columns = [f"{h}d" for h in decay_tab.columns]
    decay_t = pd.DataFrame({k: d["t-stat (NW)"] for k, d in decay.items()}).T
    decay_t.columns = [f"{h}d" for h in decay_t.columns]

    # ---------------- Quintiles ----------------
    qret = {k: an.quantile_returns(scores[k], pxr) for k in names}
    q_tab = pd.DataFrame({k: q.mean() * 12 for k, q in qret.items()}).T
    q_tab.columns = [f"Q{int(c)}" for c in q_tab.columns]
    q_tab["Q5−Q1"] = q_tab["Q5"] - q_tab["Q1"]
    q_tab["Monotonicity ρ"] = [pd.Series(q_tab.loc[k, [f"Q{i}" for i in range(1, 6)]].values).corr(pd.Series(range(5)), method="spearman") for k in names]

    # ---------------- Portfolios ----------------
    ls = {k: pf.run(pf.long_short_targets(scores[k]), pxr, COSTS, f"L/S {k}") for k in names}
    lo = {k: pf.run(pf.long_only_targets(scores[k]), pxr, COSTS, f"LO {k}") for k in names}
    ew = pf.run(pf.equal_weight_targets(pxr), pxr, COSTS, "EW Universe")
    # execution-delay sensitivity: trade the composite one day later than the signal
    delayed = {k: v.shift(1) for k, v in scores.items()}
    ls_lag = pf.run(pf.long_short_targets(delayed["Composite"]), pxr, COSTS, "L/S Composite (+1d delay)")

    common = max(r.returns.index[0] for r in [*ls.values(), *lo.values(), ew, ls_lag])
    trim = lambda r: qc.BacktestResult(r.returns[common:], r.gross_returns[common:], r.weights[common:], r.turnover[common:], r.costs[common:], name=r.name)
    ls = {k: trim(v) for k, v in ls.items()}; lo = {k: trim(v) for k, v in lo.items()}
    ew, ls_lag = trim(ew), trim(ls_lag)

    rows = ["CAGR", "Ann. Vol", "Sharpe", "Max Drawdown", "Calmar", "Hit Rate", "Avg Daily Turnover", "Annual Cost Drag"]
    ls_stats = qc.compare([ls[k] for k in names] + [ls_lag]).loc[rows]
    ls_stats.loc["Sharpe (gross)"] = [qc.metrics.sharpe(r.gross_returns) for r in [ls[k] for k in names] + [ls_lag]]
    lo_stats = qc.compare([lo[k] for k in names] + [ew]).loc[rows]
    lo_ex = {r.name: r.returns - ew.returns for r in [lo[k] for k in names]}
    lo_stats.loc["Excess vs EW (ann.)"] = [lo_ex[c].mean() * 252 if c in lo_ex else np.nan for c in lo_stats.columns]
    lo_stats.loc["Tracking Error"] = [lo_ex[c].std() * np.sqrt(252) if c in lo_ex else np.nan for c in lo_stats.columns]
    lo_stats.loc["Information Ratio"] = [lo_ex[c].mean() / lo_ex[c].std() * np.sqrt(252) if c in lo_ex else np.nan for c in lo_stats.columns]
    qc.report._PCT.update({"Excess vs EW (ann.)", "Tracking Error"})

    # ---------------- Attribution ----------------
    ff = at.fama_french_factors()
    attr_md = "Ken French data could not be downloaded; attribution skipped."
    attr_summary = {}
    if ff is not None:
        regs = {
            "L/S Composite": at.attribute(ls["Composite"].returns, ff),
            "L/S Momentum 12-1": at.attribute(ls["Momentum 12-1"].returns, ff),
            "L/S Reversal 1M": at.attribute(ls["Reversal 1M"].returns, ff),
            "L/S Low Vol 63d": at.attribute(ls["Low Vol 63d"].returns, ff),
            "LO Composite": at.attribute(lo["Composite"].returns, ff, excess=True),
        }
        tab = pd.DataFrame({k: [f"{r.loc[i,'Coef']*(100 if i=='Alpha' else 1):.2f}{'%' if i=='Alpha' else ''} ({r.loc[i,'t-stat']:.2f})" for i in r.index] for k, r in regs.items()},
                           index=next(iter(regs.values())).index)
        tab.loc["R²"] = [f"{r.attrs['R2']:.2f}" for r in regs.values()]
        tab.index.name = "Coefficient (t-stat)"
        r0 = next(iter(regs.values()))
        attr_md = (f"Daily returns, {r0.attrs['start'].date()} → {r0.attrs['end'].date()} (N={r0.attrs['N']}), Newey-West (5 lags) t-stats in brackets. "
                   "Alpha is annualised. Long/short books are self-financing so are regressed directly; the long-only book is regressed in excess of RF.\n\n"
                   + md_table(tab, lambda c, v: v))
        attr_summary = {k: (r.loc["Alpha", "Coef"], r.loc["Alpha", "t-stat"]) for k, r in regs.items()}
        print(tab.to_string())

    # ---------------- Charts ----------------
    qc.report.tearsheet({r.name: r.returns for r in [ls["Composite"], ls["Momentum 12-1"], ls["Reversal 1M"], ls["Low Vol 63d"]]},
                        DOCS / "tearsheet.png", title="HCCT Group – Quintile long/short factor portfolios (net of costs)")
    qc.report.tearsheet({r.name: r.returns for r in [lo["Composite"], lo["Low Vol 63d"], lo["Momentum 12-1"], ew]},
                        DOCS / "tearsheet_long_only.png", title="HCCT Group – Long-only top quintile vs equal-weight universe")
    plots.quintile_bars(qret, DOCS / "quintiles.png")
    plots.ic_chart(monthly_ic, decay, DOCS / "ic.png")
    score_corr = an.factor_correlation(scores, dates=me)
    ret_corr = pd.DataFrame({k: ls[k].returns for k in names}).corr()
    plots.correlation_heatmaps(score_corr, ret_corr, DOCS / "correlation.png")

    # ---------------- Markdown ----------------
    ic_tab.index.name = "Factor"
    q_tab.index.name = "Factor"
    decay_tab.index.name = "Factor"
    decay_t.index.name = "Factor"
    score_corr.index.name = "Score corr."
    ret_corr.index.name = "L/S return corr."
    period = f"{common.date()} → {ls['Composite'].returns.index[-1].date()}"
    icfmt = lambda c, v: num(v, 0) if c == "Obs" else pct(v, 0) if c == "IC > 0" else num(v, 3) if c in ("IC Mean", "IC Std") else num(v)
    md = f"""# Results — HCCT Group Equity Factor Model

Generated by `run_factor_model.py`. Universe: {px.shape[1]} large-cap US stocks (today's S&P 500 survivors, see README limitations).
Signals from {START}; portfolio period {period}. Costs: 1 bp commission + 2 bp slippage on traded notional, 50 bp/yr borrow on shorts.
Long/short books are dollar-neutral: $1 long the top quintile, $1 short the bottom quintile (equal weight), rebalanced at each month end and left to drift in between.

## 1. Rank IC (monthly, month-end score vs next-month return)

{md_table(ic_tab, icfmt)}

## 2. IC decay (mean daily rank IC by forward horizon)

{md_table(decay_tab, lambda c, v: num(v, 4))}

Newey-West t-stats (lag = horizon):

{md_table(decay_t, lambda c, v: num(v))}

## 3. Quintile returns (annualised mean of monthly equal-weight returns, gross)

{md_table(q_tab, lambda c, v: num(v) if c.startswith("Mono") else pct(v))}

## 4. Long/short quintile portfolios (net of costs)

{qc.report.to_markdown(ls_stats)}

## 5. Long-only top quintile vs equal-weight universe (net of costs)

{qc.report.to_markdown(lo_stats)}

## 6. Fama-French 5 + momentum attribution

{attr_md}

## 7. Factor correlation

{md_table(score_corr, lambda c, v: num(v))}

{md_table(ret_corr, lambda c, v: num(v))}
"""
    (DOCS / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
