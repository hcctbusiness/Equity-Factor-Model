"""Quintile portfolio construction with genuine monthly (buy-and-hold) rebalancing."""
from __future__ import annotations

import numpy as np
import pandas as pd

import quantcore as qc

from .analytics import quantile_frame


def long_short_targets(scores: pd.DataFrame, n: int = 5, leg_gross: float = 1.0) -> pd.DataFrame:
    """Dollar-neutral targets: +leg_gross equally across the top quantile,
    -leg_gross equally across the bottom quantile (sum = 0, gross = 2*leg_gross)."""
    q = quantile_frame(scores, n)
    top = (q == n).astype(float)
    bot = (q == 1).astype(float)
    w = top.div(top.sum(axis=1).replace(0, np.nan), axis=0) - bot.div(bot.sum(axis=1).replace(0, np.nan), axis=0)
    return (w * leg_gross).fillna(0.0)


def long_only_targets(scores: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Equal weight across the top quantile (fully invested)."""
    q = quantile_frame(scores, n)
    top = (q == n).astype(float)
    return top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def equal_weight_targets(prices: pd.DataFrame) -> pd.DataFrame:
    avail = prices.notna().astype(float)
    return avail.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def drift_between_rebalances(targets: pd.DataFrame, prices: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Turn month-end targets into a weight path that QuantCore trades only at rebalances.

    ``qc.sizing.rebalance`` holds the *target* constant between rebalance dates, which
    the engine (correctly) interprets as trading daily back to target. To model a
    portfolio that is traded once a month and then left to drift, this function returns
    the drifted weights so that, with ``lag=1``, held weights on day t equal the
    month-end target executed at the rebalance close and drifted with realised returns.
    Turnover (and cost) is therefore incurred only on rebalance days.
    """
    stepped = qc.sizing.rebalance(targets, freq)  # month-end targets, forward-filled
    marks = set(targets.groupby(pd.Grouper(freq=freq)).tail(1).index)
    rets = prices[targets.columns].pct_change(fill_method=None).fillna(0.0).to_numpy()
    tgt = stepped.to_numpy()
    idx = targets.index
    held = np.zeros_like(tgt)
    for t in range(1, len(idx)):
        if idx[t - 1] in marks:
            held[t] = tgt[t - 1]
        else:
            h, r = held[t - 1], rets[t - 1]
            held[t] = h * (1 + r) / (1 + h @ r)
    # backtest(lag=1) uses weights.shift(1) as held weights -> weights[t] = held[t+1]
    w = pd.DataFrame(held, index=idx, columns=targets.columns).shift(-1)
    w.iloc[-1] = w.iloc[-2] if len(w) > 1 else 0.0
    return w.fillna(0.0)


def run(targets: pd.DataFrame, prices: pd.DataFrame, costs: qc.CostModel, name: str, start: str | None = None) -> qc.BacktestResult:
    """Monthly-rebalanced backtest via QuantCore, trimmed to start after the first trade.

    Execution-delay sensitivity is tested by lagging the *scores* (see run_factor_model.py),
    which keeps the drift logic exact.
    """
    w = drift_between_rebalances(targets, prices)
    res = qc.backtest(w, prices, costs, lag=1, name=name)
    first = res.weights.abs().sum(axis=1).gt(0).idxmax()
    begin = max(first, pd.Timestamp(start)) if start else first
    sl = slice(begin, None)
    return qc.BacktestResult(res.returns[sl], res.gross_returns[sl], res.weights[sl], res.turnover[sl], res.costs[sl], name=name)
