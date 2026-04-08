"""
Variance Swap Pricing.

A variance swap pays the difference between realized variance and
a fixed strike variance over a period. Fair strike = E[σ²_realized]
under the risk-neutral measure.

This module computes:
- Fair variance swap strike (MC simulation of realized variance)
- Variance swap P&L given a realized vol outcome
- Comparison with implied variance from BS
"""

import numpy as np
from .models import MarketEnvironment, OptionContract
from .paths import simulate_paths


def price_variance_swap(
    market: MarketEnvironment,
    n_simulations: int = 50000,
    n_steps: int = 252,
    notional: float = 1e6,
) -> dict:
    """
    Compute fair variance swap strike via MC simulation.

    The fair strike K_var = E[σ²_realized] where realized variance is
    computed from log returns of simulated paths.

    Parameters
    ----------
    notional : float
        Vega notional (variance notional = vega_notional / (2 * K_vol))
    """
    paths = simulate_paths(market, n_simulations, n_steps, antithetic=True)
    dt = market.maturity / n_steps

    # Compute realized variance for each path
    # RV = (252/T) * Σ (log(S_{i+1}/S_i))²
    log_returns = np.log(paths[:, 1:] / paths[:, :-1])
    realized_var = np.sum(log_returns ** 2, axis=1) / market.maturity

    # Fair strike (annualized variance)
    fair_var = np.mean(realized_var)
    fair_vol = np.sqrt(fair_var)
    se_var = np.std(realized_var) / np.sqrt(n_simulations)

    # Implied variance for comparison
    implied_var = market.volatility ** 2

    # Variance risk premium
    vrp = fair_var - implied_var

    # P&L scenarios
    # If realized vol = X%, P&L = notional * (X²/100² - K_var) for long position
    scenarios = []
    for vol_pct in [10, 15, 20, 25, 30, 35, 40, 50]:
        real_var = (vol_pct / 100) ** 2
        pnl = notional / (2 * fair_vol) * (real_var - fair_var) * 100
        scenarios.append({
            "realized_vol": vol_pct,
            "realized_var": real_var,
            "pnl": pnl,
        })

    return {
        "fair_variance": fair_var,
        "fair_volatility": fair_vol,
        "std_error": se_var,
        "implied_variance": implied_var,
        "implied_volatility": market.volatility,
        "variance_risk_premium": vrp,
        "convexity_adj": fair_var - implied_var,
        "scenarios": scenarios,
        "notional": notional,
        "realized_var_stats": {
            "mean": float(np.mean(realized_var)),
            "std": float(np.std(realized_var)),
            "skew": float(((realized_var - np.mean(realized_var)) ** 3).mean() / np.std(realized_var) ** 3) if np.std(realized_var) > 0 else 0,
            "p5": float(np.percentile(realized_var, 5)),
            "p95": float(np.percentile(realized_var, 95)),
        },
        "n_simulations": n_simulations,
        "n_steps": n_steps,
        "model": "variance-swap",
    }
