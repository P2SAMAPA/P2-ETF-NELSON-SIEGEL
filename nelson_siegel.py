"""
Nelson-Siegel Yield Curve Engine
=================================
Fits the Nelson-Siegel (1987) parametric yield curve model to the
US Treasury term structure and extracts three latent factors:

    y(tau) = beta0
           + beta1 * [(1 - exp(-lambda*tau)) / (lambda*tau)]
           + beta2 * [(1 - exp(-lambda*tau)) / (lambda*tau) - exp(-lambda*tau)]

where:
    beta0  = long-run level  (slope of yield curve as tau -> inf)
    beta1  = short-term component (loads heavily on short maturities)
    beta2  = medium-term hump (curvature)
    lambda = decay speed (controls where the hump peaks)

ETF scoring logic (Diebold-Li 2006 approach):
---------------------------------------------
Each ETF in the universe is assigned a score based on its sensitivity
(OLS rolling beta) to the three NS factors. The score is a composite of:

  - Level (beta0) momentum:   rising level = rising rates = bad for duration
  - Slope (beta1) change:     steepening = risk-on = good for equity-sensitive FI
  - Curvature (beta2) signal: humping curve = medium-term rate pressure

Specifically:
  score_i = -w_level  * beta_level_i  * delta_level
            + w_slope  * beta_slope_i  * delta_slope
            + w_curve  * beta_curve_i  * delta_curve

Negative level loading because rising rates hurt bond ETFs.
Positive slope loading because steepening favours credit/equity risk.
All betas and deltas are z-scored cross-sectionally before combining.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import zscore


# ── Nelson-Siegel factor loadings ────────────────────────────────────────────

def ns_loadings(maturities: np.ndarray, lam: float) -> np.ndarray:
    """
    Compute the three Nelson-Siegel factor loading vectors for a given
    array of maturities (in years) and decay parameter lambda.

    Returns matrix of shape (len(maturities), 3):
        col 0 = level loading   (always 1.0)
        col 1 = slope loading   [(1 - exp(-lam*tau)) / (lam*tau)]
        col 2 = curvature loading [slope_loading - exp(-lam*tau)]
    """
    lam_tau = lam * maturities
    # Guard against division by zero for very short maturities
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(
            lam_tau < 1e-10,
            1.0,
            (1.0 - np.exp(-lam_tau)) / lam_tau
        )
    curvature = slope - np.exp(-lam_tau)
    level     = np.ones_like(maturities)
    return np.column_stack([level, slope, curvature])


def fit_ns(yields: np.ndarray, maturities: np.ndarray,
           lambda_init: float = 0.7) -> dict:
    """
    Fit Nelson-Siegel model to a single cross-section of yields.

    Parameters
    ----------
    yields     : 1-D array of observed yields (decimal, e.g. 0.045 for 4.5%)
                 Must align with maturities.
    maturities : 1-D array of maturities in years.
    lambda_init: starting value for lambda optimisation.

    Returns
    -------
    dict with keys: beta0, beta1, beta2, lam, rmse, success
    """
    mask = np.isfinite(yields) & np.isfinite(maturities) & (maturities > 0)
    y    = yields[mask]
    tau  = maturities[mask]

    if len(y) < 4:
        return dict(beta0=np.nan, beta1=np.nan, beta2=np.nan,
                    lam=lambda_init, rmse=np.nan, success=False)

    def objective(params):
        lam = params[0]
        if lam <= 0:
            return 1e10
        L = ns_loadings(tau, lam)          # (n_maturities, 3)
        # OLS for betas given lam
        try:
            betas, _, _, _ = np.linalg.lstsq(L, y, rcond=None)
        except np.linalg.LinAlgError:
            return 1e10
        y_hat = L @ betas
        return np.mean((y - y_hat) ** 2)

    res = minimize(objective, x0=[lambda_init],
                   bounds=[(0.01, 5.0)],
                   method="L-BFGS-B",
                   options={"ftol": 1e-10, "maxiter": 200})

    lam_opt = res.x[0]
    L       = ns_loadings(tau, lam_opt)
    betas, _, _, _ = np.linalg.lstsq(L, y, rcond=None)
    y_hat   = L @ betas
    rmse    = np.sqrt(np.mean((y - y_hat) ** 2))

    return dict(beta0=float(betas[0]),
                beta1=float(betas[1]),
                beta2=float(betas[2]),
                lam=float(lam_opt),
                rmse=float(rmse),
                success=bool(res.success))


def extract_ns_factors(df: pd.DataFrame,
                       treasury_cols: dict,
                       window: int,
                       lambda_init: float = 0.7) -> pd.DataFrame:
    """
    Roll a Nelson-Siegel fit over a DataFrame, producing daily time series
    of (beta0, beta1, beta2, lambda, rmse).

    Parameters
    ----------
    df           : master_data DataFrame with treasury yield columns.
    treasury_cols: dict mapping column name → maturity in years.
    window       : rolling window in days.
    lambda_init  : initial lambda for optimisation.

    Returns
    -------
    DataFrame indexed by date with columns [beta0, beta1, beta2, lam, rmse].
    Only rows where sufficient treasury data is available are included.
    """
    available = {col: mat for col, mat in treasury_cols.items()
                 if col in df.columns}
    if len(available) < 4:
        return pd.DataFrame()

    cols     = list(available.keys())
    mats     = np.array([available[c] for c in cols])
    yield_df = df[cols].copy() / 100.0   # convert percent → decimal

    records = []
    dates   = yield_df.index

    for i in range(window - 1, len(dates)):
        slice_yields = yield_df.iloc[i - window + 1: i + 1]
        # Use the last available cross-section in the window (today's curve)
        today_yields = slice_yields.iloc[-1].values.astype(float)
        result       = fit_ns(today_yields, mats, lambda_init)
        result["date"] = dates[i]
        records.append(result)

    if not records:
        return pd.DataFrame()

    factors = pd.DataFrame(records).set_index("date")
    return factors[["beta0", "beta1", "beta2", "lam", "rmse"]]


# ── ETF sensitivity to NS factors ────────────────────────────────────────────

def compute_etf_ns_betas(returns: pd.DataFrame,
                         factors: pd.DataFrame,
                         window: int) -> pd.DataFrame:
    """
    Compute rolling OLS betas of each ETF's daily return on the daily
    changes of NS factors (beta0, beta1, beta2).

    Parameters
    ----------
    returns : DataFrame of ETF log returns, indexed by date.
    factors : DataFrame of NS factors (beta0, beta1, beta2), indexed by date.
    window  : lookback window for OLS regression.

    Returns
    -------
    DataFrame of shape (n_dates, n_etfs * 3) — one row per date, columns
    named <TICKER>_b0, <TICKER>_b1, <TICKER>_b2.
    """
    # Use factor changes as regressors (stationary)
    factor_changes = factors[["beta0", "beta1", "beta2"]].diff().dropna()

    # Align to common dates
    common = returns.index.intersection(factor_changes.index)
    ret    = returns.loc[common]
    fc     = factor_changes.loc[common]

    results = {}
    dates   = common

    for i in range(window - 1, len(dates)):
        d      = dates[i]
        r_win  = ret.iloc[i - window + 1: i + 1]
        fc_win = fc.iloc[i - window + 1: i + 1]

        # Drop rows with any NaN
        combined = pd.concat([r_win, fc_win], axis=1).dropna()
        if len(combined) < window // 2:
            continue

        X = combined[["beta0", "beta1", "beta2"]].values
        X = np.column_stack([np.ones(len(X)), X])   # add intercept

        row = {}
        for ticker in ret.columns:
            y = combined[ticker].values if ticker in combined.columns else None
            if y is None or np.all(np.isnan(y)):
                row[f"{ticker}_b0"] = np.nan
                row[f"{ticker}_b1"] = np.nan
                row[f"{ticker}_b2"] = np.nan
                continue
            try:
                betas, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                # betas[0] = intercept, betas[1]=b0, betas[2]=b1, betas[3]=b2
                row[f"{ticker}_b0"] = float(betas[1])
                row[f"{ticker}_b1"] = float(betas[2])
                row[f"{ticker}_b2"] = float(betas[3])
            except Exception:
                row[f"{ticker}_b0"] = np.nan
                row[f"{ticker}_b1"] = np.nan
                row[f"{ticker}_b2"] = np.nan

        results[d] = row

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).T


# ── Score computation ─────────────────────────────────────────────────────────

def compute_ns_scores(returns: pd.DataFrame,
                      factors: pd.DataFrame,
                      ns_betas: pd.DataFrame,
                      score_lookback: int = 21,
                      w_level: float = 0.40,
                      w_slope: float = 0.35,
                      w_curve: float = 0.25) -> pd.Series:
    """
    Compute today's NS-based score for each ETF.

    Score = -w_level * zscore(b0_i * delta_level)
            + w_slope * zscore(b1_i * delta_slope)
            + w_curve * zscore(b2_i * delta_curve)

    where delta_* = recent change (score_lookback-day) in NS factor.

    Returns pd.Series {ticker: score}.
    """
    if ns_betas.empty or factors.empty:
        return pd.Series(dtype=float)

    # Recent factor changes (score_lookback days)
    common_dates = factors.index.intersection(ns_betas.index)
    if len(common_dates) < score_lookback + 1:
        return pd.Series(dtype=float)

    recent_factors = factors.loc[common_dates].iloc[-(score_lookback + 1):]
    delta_level = float(recent_factors["beta0"].iloc[-1] - recent_factors["beta0"].iloc[0])
    delta_slope = float(recent_factors["beta1"].iloc[-1] - recent_factors["beta1"].iloc[0])
    delta_curve = float(recent_factors["beta2"].iloc[-1] - recent_factors["beta2"].iloc[0])

    # Latest ETF betas
    latest_betas = ns_betas.iloc[-1]
    tickers      = returns.columns.tolist()

    raw_scores = {}
    for ticker in tickers:
        b0 = latest_betas.get(f"{ticker}_b0", np.nan)
        b1 = latest_betas.get(f"{ticker}_b1", np.nan)
        b2 = latest_betas.get(f"{ticker}_b2", np.nan)
        if any(np.isnan(v) for v in [b0, b1, b2]):
            raw_scores[ticker] = np.nan
        else:
            raw_scores[ticker] = (
                -w_level * b0 * delta_level
                + w_slope * b1 * delta_slope
                + w_curve * b2 * delta_curve
            )

    s = pd.Series(raw_scores).dropna()
    if len(s) < 2:
        return s

    # Cross-sectional z-score
    s = (s - s.mean()) / (s.std() + 1e-10)
    return s
