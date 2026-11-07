"""Offline unit tests on synthetic data (no network)."""
import numpy as np
import pandas as pd
import pytest

import quantcore as qc
from factor_model import analytics as an
from factor_model import attribution as at
from factor_model import factors as fc
from factor_model import portfolio as pf


@pytest.fixture
def prices():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2015-01-01", periods=600)
    r = rng.normal(0.0003, 0.015, size=(len(idx), 40))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), idx, [f"S{i:02d}" for i in range(40)])


def test_winsorize_clips_to_quantiles():
    row = pd.DataFrame([np.r_[np.arange(99.0), 1e6]])
    w = fc.winsorize(row, 0.01, 0.99)
    assert w.values.max() < 1e6 and w.values.max() == pytest.approx(row.quantile(0.99, axis=1).iloc[0])
    assert w.values.min() == pytest.approx(row.quantile(0.01, axis=1).iloc[0])


def test_zscore_rows_have_zero_mean_unit_std():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(5, 3, size=(50, 30)))
    df.iloc[0, :5] = np.nan
    z = fc.zscore(df)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-12)
    assert np.allclose(z.std(axis=1, ddof=1), 1)
    assert z.iloc[0, :5].isna().all()


def test_zscore_requires_min_names():
    df = pd.DataFrame([[1.0, 2.0, 3.0] + [np.nan] * 20])
    assert fc.zscore(df, min_names=10).isna().all().all()


def test_factors_use_no_future_data(prices):
    base = fc.build_factors(prices)
    shocked = prices.copy()
    shocked.iloc[400:] *= 3.0  # change the future only
    alt = fc.build_factors(shocked)
    for k in base:
        pd.testing.assert_frame_equal(base[k].iloc[:400], alt[k].iloc[:400])


def test_rank_ic_on_known_correlated_data():
    rng = np.random.default_rng(7)
    n_t, n_s = 300, 80
    f = pd.DataFrame(rng.normal(size=(n_t, n_s)))
    perfect = an.rank_ic(f, f * 2 + 1)
    assert np.allclose(perfect, 1.0)
    assert np.allclose(an.rank_ic(f, -f), -1.0)
    rho = 0.3
    y = rho * f + np.sqrt(1 - rho**2) * pd.DataFrame(rng.normal(size=(n_t, n_s)))
    ic = an.rank_ic(f, y)
    assert abs(ic.mean() - 0.29) < 0.03  # Spearman ≈ (6/π)·asin(ρ/2) ≈ 0.287
    s = an.ic_summary(ic)
    assert s["t-stat (NW)"] > 20 and s["IC > 0"] > 0.95
    noise = an.rank_ic(f, pd.DataFrame(rng.normal(size=(n_t, n_s))))
    assert abs(noise.mean()) < 0.02


def test_newey_west_matches_plain_t_without_lags():
    x = pd.Series(np.random.default_rng(3).normal(0.1, 1, 500))
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert an.newey_west_tstat(x, lags=0) == pytest.approx(plain)


def test_quintile_assignment():
    row = pd.Series(np.arange(1.0, 21.0), index=[f"S{i}" for i in range(20)])
    row.iloc[3] = np.nan
    q = an.assign_quantiles(row, 5)
    assert np.isnan(q.iloc[3])
    counts = q.value_counts().sort_index()
    assert list(counts.index) == [1, 2, 3, 4, 5]
    assert counts.max() - counts.min() <= 1
    assert q.loc["S19"] == 5 and q.loc["S0"] == 1
    # monotone: higher score never in a lower bucket
    v = row.dropna().sort_values()
    assert q[v.index].is_monotonic_increasing


def test_quantile_returns_monotone_with_perfect_signal(prices):
    fut = prices.shift(-25) / prices - 1  # cheating signal: future return
    qr = an.quantile_returns(fut, prices)
    m = qr.mean()
    assert m.is_monotonic_increasing and m.iloc[-1] > m.iloc[0]


def test_long_short_is_dollar_neutral(prices):
    scores = fc.build_factors(prices)["Composite"]
    w = pf.long_short_targets(scores)
    live = w.abs().sum(axis=1) > 0
    assert live.sum() > 100
    assert np.allclose(w[live].sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(w[live].abs().sum(axis=1), 2.0)
    assert np.allclose(w[live].clip(lower=0).sum(axis=1), 1.0)


def test_long_only_fully_invested(prices):
    w = pf.long_only_targets(fc.build_factors(prices)["Momentum 12-1"])
    live = w.sum(axis=1) > 0
    assert np.allclose(w[live].sum(axis=1), 1.0) and (w >= 0).all().all()


def test_monthly_rebalance_trades_once_per_month(prices):
    scores = fc.build_factors(prices)["Reversal 1M"]
    res = pf.run(pf.long_short_targets(scores), prices, qc.CostModel(1, 2, 50), "x")
    traded = res.turnover[res.turnover > 1e-9]
    per_month = traded.groupby(traded.index.to_period("M")).size()
    assert per_month.max() == 1
    # held book stays (close to) dollar-neutral; shorts pay borrow every day
    assert res.weights.sum(axis=1).abs().max() < 0.25
    assert (res.costs > 0).all()


def test_ols_hac_recovers_known_betas():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2018-01-01", periods=3000)
    X = pd.DataFrame(rng.normal(0, 0.01, size=(3000, 2)), idx, ["Mkt-RF", "SMB"])
    y = 0.0002 + 0.8 * X["Mkt-RF"] - 0.3 * X["SMB"] + rng.normal(0, 0.002, 3000)
    out = at.ols_hac(pd.Series(y, idx), X)
    assert out.loc["Mkt-RF", "Coef"] == pytest.approx(0.8, abs=0.02)
    assert out.loc["SMB", "Coef"] == pytest.approx(-0.3, abs=0.02)
    assert out.loc["Alpha", "Coef"] == pytest.approx(0.0002 * 252, abs=0.02)


def test_parse_french_csv():
    txt = "Header text\n\n,Mkt-RF,SMB,RF\n20200102,  1.00,  -0.50,  0.01\n20200103, -2.00, 0.25, 0.01\n\n Annual Factors\n,Mkt-RF,SMB,RF\n2020, 10, 1, 0.5\n"
    df = at.parse_french_csv(txt)
    assert list(df.columns) == ["Mkt-RF", "SMB", "RF"] and len(df) == 2
    assert df.iloc[0, 0] == pytest.approx(0.01) and df.index[1] == pd.Timestamp("2020-01-03")
