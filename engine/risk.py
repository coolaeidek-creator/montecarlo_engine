"""
Risk metrics module for Monte Carlo simulation.

Provides:
- Value at Risk (VaR) — parametric, historical, and Monte Carlo
- Conditional VaR (CVaR / Expected Shortfall)
- Maximum Drawdown
- Sharpe Ratio from simulated returns
- Portfolio risk aggregation
"""

import numpy as np
from .models import MarketEnvironment
from .paths import simulate_paths


def compute_var(
    market: MarketEnvironment,
    confidence: float = 0.95,
    horizon_days: int = 10,
    n_simulations: int = 50000,
    method: str = "monte_carlo",
) -> dict:
    """
    Compute Value at Risk.

    VaR answers: "What is the maximum loss at X% confidence over N days?"

    Methods:
    - 'parametric':   assumes normal returns (fast, closed-form)
    - 'monte_carlo':  simulates full GBM paths (most accurate)

    Parameters
    ----------
    confidence : float
        Confidence level (e.g. 0.95 = 95%)
    horizon_days : int
        Risk horizon in trading days
    """
    S = market.spot
    sigma = market.volatility
    r = market.rate

    if method == "parametric":
        # Parametric VaR (normal assumption)
        from scipy.stats import norm
        dt = horizon_days / 252.0
        z = norm.ppf(1 - confidence)
        mu = (r - 0.5 * sigma ** 2) * dt
        vol = sigma * np.sqrt(dt)

        # VaR as dollar loss
        var_pct = -(mu + z * vol)
        var_dollar = S * var_pct

        # CVaR (Expected Shortfall)
        es_z = norm.pdf(norm.ppf(1 - confidence)) / (1 - confidence)
        cvar_pct = -(mu - vol * es_z)
        cvar_dollar = S * cvar_pct

        return {
            "var_pct": var_pct,
            "var_dollar": var_dollar,
            "cvar_pct": cvar_pct,
            "cvar_dollar": cvar_dollar,
            "confidence": confidence,
            "horizon_days": horizon_days,
            "method": "parametric",
        }

    else:  # monte_carlo
        # Simulate paths over the horizon
        horizon_market = MarketEnvironment(
            spot=S, rate=r, volatility=sigma,
            maturity=horizon_days / 252.0,
        )
        paths = simulate_paths(
            horizon_market, n_simulations, n_steps=horizon_days, antithetic=True
        )
        terminal = paths[:, -1]

        # P&L distribution
        pnl = terminal - S
        pnl_sorted = np.sort(pnl)

        # VaR: loss at the (1-confidence) percentile
        var_idx = int((1 - confidence) * len(pnl_sorted))
        var_dollar = -pnl_sorted[var_idx]
        var_pct = var_dollar / S

        # CVaR: average of losses beyond VaR
        tail_losses = pnl_sorted[:var_idx]
        cvar_dollar = -np.mean(tail_losses) if len(tail_losses) > 0 else var_dollar
        cvar_pct = cvar_dollar / S

        # Max drawdown from paths
        running_max = np.maximum.accumulate(paths, axis=1)
        drawdowns = (running_max - paths) / running_max
        max_dd = np.mean(np.max(drawdowns, axis=1))

        return {
            "var_pct": var_pct,
            "var_dollar": var_dollar,
            "cvar_pct": cvar_pct,
            "cvar_dollar": cvar_dollar,
            "max_drawdown": max_dd,
            "confidence": confidence,
            "horizon_days": horizon_days,
            "method": "monte_carlo",
            "pnl_mean": np.mean(pnl),
            "pnl_std": np.std(pnl),
            "pnl_skew": float(_skewness(pnl)),
            "pnl_kurtosis": float(_kurtosis(pnl)),
            "worst_case": -pnl_sorted[0],
            "best_case": pnl_sorted[-1],
            "n_simulations": n_simulations,
        }


def compute_portfolio_var(
    markets: list,
    weights: np.ndarray,
    confidence: float = 0.95,
    horizon_days: int = 10,
    n_simulations: int = 50000,
    correlation_matrix: np.ndarray = None,
) -> dict:
    """
    Portfolio-level VaR using correlated GBM simulation.

    Parameters
    ----------
    markets : list of MarketEnvironment
        One per asset
    weights : np.ndarray
        Portfolio weights (sum to 1.0)
    correlation_matrix : np.ndarray or None
        If None, assumes zero correlation (independent)
    """
    n_assets = len(markets)
    dt = horizon_days / 252.0

    # Generate correlated shocks via Cholesky decomposition
    if correlation_matrix is not None:
        L = np.linalg.cholesky(correlation_matrix)
    else:
        L = np.eye(n_assets)

    z_indep = np.random.standard_normal((n_simulations, n_assets))
    z_corr = z_indep @ L.T  # correlated shocks

    # Simulate terminal values for each asset
    portfolio_values = np.zeros(n_simulations)
    initial_value = 0.0

    for i, mkt in enumerate(markets):
        drift = (mkt.rate - 0.5 * mkt.volatility ** 2) * dt
        diffusion = mkt.volatility * np.sqrt(dt) * z_corr[:, i]
        terminal = mkt.spot * np.exp(drift + diffusion)

        portfolio_values += weights[i] * terminal
        initial_value += weights[i] * mkt.spot

    pnl = portfolio_values - initial_value
    pnl_sorted = np.sort(pnl)

    var_idx = int((1 - confidence) * n_simulations)
    var_dollar = -pnl_sorted[var_idx]
    var_pct = var_dollar / initial_value

    tail = pnl_sorted[:var_idx]
    cvar_dollar = -np.mean(tail) if len(tail) > 0 else var_dollar
    cvar_pct = cvar_dollar / initial_value

    return {
        "var_pct": var_pct,
        "var_dollar": var_dollar,
        "cvar_pct": cvar_pct,
        "cvar_dollar": cvar_dollar,
        "confidence": confidence,
        "horizon_days": horizon_days,
        "portfolio_value": initial_value,
        "n_assets": n_assets,
        "expected_pnl": np.mean(pnl),
        "pnl_std": np.std(pnl),
        "n_simulations": n_simulations,
    }


def _skewness(x: np.ndarray) -> float:
    """Sample skewness."""
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    return (n / ((n - 1) * (n - 2))) * np.sum(((x - m) / s) ** 3)


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (0 for normal)."""
    n = len(x)
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s == 0:
        return 0.0
    k4 = np.mean(((x - m) / s) ** 4)
    return k4 - 3.0
