"""Factor diagnostics: rank IC, IC decay, quintile returns, factor correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Simple return from close t to close t+h (aligned at t)."""
    return prices.shift(-horizon) / prices - 1.0


def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, min_names: int = 10) -> pd.Series:
    """Row-wise Spearman rank correlation between factor and forward returns."""
    fwd = fwd.reindex_like(factor)
    mask = factor.notna() & fwd.notna()
    fr = factor.where(mask).rank(axis=1)
    rr = fwd.where(mask).rank(axis=1)
    fr = fr.sub(fr.mean(axis=1), axis=0)
    rr = rr.sub(rr.mean(axis=1), axis=0)
    num = (fr * rr).sum(axis=1)
    den = np.sqrt((fr**2).sum(axis=1) * (rr**2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    ic[mask.sum(axis=1) < min_names] = np.nan
    return ic.dropna()


def newey_west_tstat(x: pd.Series, lags: int) -> float:
    """t-statistic of the mean with a Newey-West (Bartlett) HAC standard error."""
    x = pd.Series(x).dropna().to_numpy(float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    lr = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        lr += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    return float(x.mean() / np.sqrt(lr / n)) if lr > 0 else np.nan


def ic_summary(ic: pd.Series, horizon: int = 1, periods_per_year: int | None = None) -> pd.Series:
    """IC mean, std, information ratio (mean/std), HAC t-stat and hit rate.

    ``horizon`` sets the Newey-West lag so overlapping forward windows are handled.
    """
    ic = ic.dropna()
    sd = ic.std(ddof=1)
    out = {
        "IC Mean": ic.mean(),
        "IC Std": sd,
        "IC IR": ic.mean() / sd if sd > 0 else np.nan,
        "t-stat (NW)": newey_west_tstat(ic, lags=max(horizon, 1)),
        "IC > 0": (ic > 0).mean(),
        "Obs": float(len(ic)),
    }
    if periods_per_year:
        out["IC IR (ann.)"] = out["IC IR"] * np.sqrt(periods_per_year)
    return pd.Series(out)


def ic_decay(factor: pd.DataFrame, prices: pd.DataFrame, horizons=(1, 5, 21, 63)) -> pd.DataFrame:
    """Mean daily rank IC and HAC t-stat for several forward horizons."""
    rows = {}
    for h in horizons:
        ic = rank_ic(factor, forward_returns(prices, h))
        rows[h] = {"IC Mean": ic.mean(), "t-stat (NW)": newey_west_tstat(ic, lags=h)}
    return pd.DataFrame(rows).T.rename_axis("Horizon (days)")


def assign_quantiles(row: pd.Series, n: int = 5) -> pd.Series:
    """Map one cross-section of scores to buckets 1..n (n = highest score). NaN stays NaN."""
    valid = row.dropna()
    out = pd.Series(np.nan, index=row.index)
    if len(valid) < n:
        return out
    pct = valid.rank(method="first") / len(valid)
    out[valid.index] = np.ceil(pct * n).clip(1, n)
    return out


def quantile_frame(scores: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return scores.apply(assign_quantiles, axis=1, n=n)


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(pd.Grouper(freq="ME")).last().dropna().values)


def quantile_returns(scores: pd.DataFrame, prices: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Equal-weight next-month return of each quantile, formed at each month end.

    Returns a DataFrame (formation date x quantile) of simple monthly returns
    (gross of costs) — the classic monotonicity test.
    """
    dates = month_end_dates(scores.index)
    px_m = prices.reindex(dates)
    fwd = px_m.shift(-1) / px_m - 1.0
    q = quantile_frame(scores.reindex(dates), n)
    out = {}
    for d in dates[:-1]:
        f, b = fwd.loc[d], q.loc[d]
        ok = f.notna() & b.notna()
        if ok.sum() >= n:
            out[d] = f[ok].groupby(b[ok]).mean()
    return pd.DataFrame(out).T.sort_index(axis=1)


def factor_correlation(scores: dict[str, pd.DataFrame], dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Time-average of the cross-sectional Spearman correlation between factor scores."""
    names = list(scores)
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, fb = scores[a], scores[b]
            if dates is not None:
                fa, fb = fa.reindex(dates), fb.reindex(dates)
            c = rank_ic(fa, fb).mean()
            m.loc[a, b] = m.loc[b, a] = c
    return m
