"""Price-based equity factors and cross-sectional standardisation.

Every factor value at date t uses only prices up to and including the close of t.
Sign convention: a HIGHER score means a MORE attractive stock (expected to outperform).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """12-month return skipping the most recent month (Jegadeesh-Titman / Carhart)."""
    return prices.shift(21) / prices.shift(252) - 1.0


def reversal_1m(prices: pd.DataFrame) -> pd.DataFrame:
    """Short-term reversal: minus the trailing 1-month (21d) return."""
    return -(prices / prices.shift(21) - 1.0)


def low_volatility(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Low-volatility anomaly: minus trailing realised daily volatility."""
    rets = prices.pct_change(fill_method=None)
    vol = rets.rolling(window, min_periods=int(window * 0.8)).std()
    return -vol


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each row (date) to its own [lower, upper] cross-sectional quantiles."""
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore(df: pd.DataFrame, min_names: int = 10) -> pd.DataFrame:
    """Cross-sectional z-score per row; rows with fewer than ``min_names`` valid values -> NaN."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1, ddof=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    z[df.notna().sum(axis=1) < min_names] = np.nan
    return z


def standardize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Winsorize then z-score cross-sectionally."""
    return zscore(winsorize(df, lower, upper))


def composite(scores: dict[str, pd.DataFrame], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Weighted average of standardized factor scores, re-standardized.

    A stock needs a valid score on every factor to receive a composite score.
    """
    weights = weights or {k: 1.0 for k in scores}
    total = sum(weights.values())
    out = sum(scores[k] * (w / total) for k, w in weights.items())
    return zscore(out)


def build_factors(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All standardized factor scores plus the equal-weight composite."""
    raw = {
        "Momentum 12-1": momentum_12_1(prices),
        "Reversal 1M": reversal_1m(prices),
        "Low Vol 63d": low_volatility(prices),
    }
    scores = {k: standardize(v) for k, v in raw.items()}
    scores["Composite"] = composite({k: scores[k] for k in raw})
    return scores
