"""Fama-French 5-factor + momentum attribution (Ken French data library)."""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
}
CACHE = Path(os.environ.get("FACTOR_MODEL_CACHE", Path.home() / ".quantcore_cache"))


def parse_french_csv(text: str) -> pd.DataFrame:
    """Parse the first (daily) table of a Ken French CSV: header row starts with ','."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(","))
    rows = []
    for l in lines[start + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) < 2 or not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts)
    cols = [c.strip() for c in lines[start].split(",")[1:]]
    df = pd.DataFrame([r[1:] for r in rows], index=pd.to_datetime([r[0] for r in rows], format="%Y%m%d"), columns=cols)
    return df.astype(float) / 100.0


def _load(key: str, refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"french_{key}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    req = Request(BASE + FILES[key], headers={"User-Agent": "Mozilla/5.0 (HCCT Group factor research)"})
    raw = urlopen(req, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    df = parse_french_csv(z.read(z.namelist()[0]).decode("latin1"))
    df.to_csv(path)
    return df


def fama_french_factors(refresh: bool = False) -> pd.DataFrame | None:
    """Daily Mkt-RF, SMB, HML, RMW, CMA, MOM, RF (decimals). Returns None if unavailable."""
    try:
        ff = _load("ff5", refresh)
        mom = _load("mom", refresh)
        mom.columns = ["MOM"]
        return ff.join(mom, how="inner")
    except Exception as exc:  # network / format failure -> degrade gracefully
        print(f"[attribution] Ken French data unavailable ({exc!r}); skipping attribution.")
        return None


def ols_hac(y: pd.Series, X: pd.DataFrame, lags: int = 5, periods: int = 252) -> pd.DataFrame:
    """OLS with intercept and Newey-West standard errors.

    Returns coefficient, t-stat table; the intercept row ('Alpha') is annualised.
    """
    df = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    Y = df["y"].to_numpy()
    Xm = np.column_stack([np.ones(len(df)), df[X.columns].to_numpy()])
    n, k = Xm.shape
    XtX_inv = np.linalg.inv(Xm.T @ Xm)
    beta = XtX_inv @ Xm.T @ Y
    u = Y - Xm @ beta
    Xu = Xm * u[:, None]
    S = Xu.T @ Xu
    for L in range(1, lags + 1):
        G = Xu[L:].T @ Xu[:-L]
        S += (1 - L / (lags + 1)) * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    names = ["Alpha"] + list(X.columns)
    out = pd.DataFrame({"Coef": beta, "t-stat": beta / se}, index=names)
    out.loc["Alpha", "Coef"] *= periods
    r2 = 1 - (u @ u) / ((Y - Y.mean()) @ (Y - Y.mean()))
    out.attrs.update({"R2": float(r2), "N": int(n), "start": df.index[0], "end": df.index[-1]})
    return out


def attribute(returns: pd.Series, ff: pd.DataFrame, excess: bool = False, lags: int = 5) -> pd.DataFrame:
    """Regress strategy returns on FF5 + MOM. Set ``excess=True`` for funded (long-only)
    portfolios so RF is subtracted; dollar-neutral books are already excess returns."""
    y = returns.copy()
    if excess:
        y = y - ff["RF"].reindex(y.index)
    X = ff[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]]
    return ols_hac(y, X, lags=lags)
