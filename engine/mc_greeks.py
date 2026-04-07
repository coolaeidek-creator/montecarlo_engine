"""
Monte Carlo Greeks Estimation.

Computes option sensitivities numerically via finite differences
applied to MC simulation. Unlike analytical BS Greeks, these work
for any payoff structure (exotic, path-dependent, etc.).

Methods:
- Bump-and-revalue (finite difference)
- Pathwise derivative (where applicable)
"""

import numpy as np
from .models import MarketEnvironment, OptionContract
from .pricer import price_option


def mc_greeks(
    market: MarketEnvironment,
    contract: OptionContract,
    n_simulations: int = 50000,
    method: str = "antithetic",
) -> dict:
    """
    Compute Greeks via finite difference on MC prices.

    Uses centered differences for better accuracy:
    delta ≈ (V(S+h) - V(S-h)) / (2h)
    """
    base = price_option(market, contract, n_simulations, method)

    # Delta: bump spot
    h_spot = market.spot * 0.01  # 1% bump
    m_up = MarketEnvironment(
        spot=market.spot + h_spot, rate=market.rate,
        volatility=market.volatility, maturity=market.maturity,
    )
    m_dn = MarketEnvironment(
        spot=market.spot - h_spot, rate=market.rate,
        volatility=market.volatility, maturity=market.maturity,
    )
    np.random.seed(42)
    v_up = price_option(m_up, contract, n_simulations, method)["price"]
    np.random.seed(42)
    v_dn = price_option(m_dn, contract, n_simulations, method)["price"]
    delta = (v_up - v_dn) / (2 * h_spot)

    # Gamma: second derivative
    gamma = (v_up - 2 * base["price"] + v_dn) / (h_spot ** 2)

    # Vega: bump vol
    h_vol = 0.01  # 1% bump
    m_vup = MarketEnvironment(
        spot=market.spot, rate=market.rate,
        volatility=market.volatility + h_vol, maturity=market.maturity,
    )
    m_vdn = MarketEnvironment(
        spot=market.spot, rate=market.rate,
        volatility=max(0.01, market.volatility - h_vol), maturity=market.maturity,
    )
    np.random.seed(42)
    v_vup = price_option(m_vup, contract, n_simulations, method)["price"]
    np.random.seed(42)
    v_vdn = price_option(m_vdn, contract, n_simulations, method)["price"]
    vega = (v_vup - v_vdn) / (2 * h_vol * 100)  # per 1% vol

    # Theta: bump time
    h_t = 1 / 252  # 1 day
    if market.maturity > h_t:
        m_tdn = MarketEnvironment(
            spot=market.spot, rate=market.rate,
            volatility=market.volatility, maturity=market.maturity - h_t,
        )
        np.random.seed(42)
        v_tdn = price_option(m_tdn, contract, n_simulations, method)["price"]
        theta = (v_tdn - base["price"]) / h_t / 252  # per day
    else:
        theta = 0.0

    # Rho: bump rate
    h_r = 0.001  # 10bp bump
    m_rup = MarketEnvironment(
        spot=market.spot, rate=market.rate + h_r,
        volatility=market.volatility, maturity=market.maturity,
    )
    np.random.seed(42)
    v_rup = price_option(m_rup, contract, n_simulations, method)["price"]
    rho = (v_rup - base["price"]) / (h_r * 100)  # per 1% rate

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "base_price": base["price"],
        "method": f"mc_finite_diff ({method})",
    }
