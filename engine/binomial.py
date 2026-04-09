"""
Binomial Tree Option Pricing (Cox-Ross-Rubinstein).

The CRR binomial tree provides a discrete-time approximation to
the continuous GBM process. It's exact in the limit of infinite
steps and gives a useful visualization of the option pricing process.

Key advantage over MC: exact for American options (no regression needed).
"""

import numpy as np
from .models import MarketEnvironment, OptionContract


def _binomial_price_core(S, K, r, sigma, T, n_steps, option_type, american):
    """Core pricing without Greeks computation (no recursion)."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    ST = np.array([S * u ** (n_steps - 2 * j) for j in range(n_steps + 1)])

    if option_type == "call":
        V = np.maximum(ST - K, 0)
    else:
        V = np.maximum(K - ST, 0)

    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:i + 1] + (1 - p) * V[1:i + 2])
        if american:
            Si = np.array([S * u ** (i - 2 * j) for j in range(i + 1)])
            if option_type == "call":
                exercise = np.maximum(Si - K, 0)
            else:
                exercise = np.maximum(K - Si, 0)
            V = np.maximum(V, exercise)

    return float(V[0]), u, d, p


def price_binomial(
    market: MarketEnvironment,
    contract: OptionContract,
    n_steps: int = 200,
    american: bool = False,
) -> dict:
    """
    Price option using CRR binomial tree.
    """
    S = market.spot
    K = contract.strike
    r = market.rate
    sigma = market.volatility
    T = market.maturity

    price, u, d, p = _binomial_price_core(
        S, K, r, sigma, T, n_steps, contract.option_type, american,
    )

    # European price for comparison
    if american:
        euro_price, _, _, _ = _binomial_price_core(
            S, K, r, sigma, T, n_steps, contract.option_type, False,
        )
        early_premium = price - euro_price
    else:
        euro_price = price
        early_premium = 0.0

    # Greeks via finite difference on core pricer (no recursion)
    h = S * 0.01
    p_up, _, _, _ = _binomial_price_core(
        S + h, K, r, sigma, T, n_steps, contract.option_type, american,
    )
    p_dn, _, _, _ = _binomial_price_core(
        S - h, K, r, sigma, T, n_steps, contract.option_type, american,
    )
    delta = (p_up - p_dn) / (2 * h)
    gamma = (p_up - 2 * price + p_dn) / (h ** 2)

    return {
        "price": price,
        "european_price": euro_price,
        "early_exercise_premium": early_premium,
        "delta": delta,
        "gamma": gamma,
        "u": u,
        "d": d,
        "p": p,
        "n_steps": n_steps,
        "american": american,
        "option_type": contract.option_type,
        "method": "crr-binomial",
    }


def binomial_convergence(
    market: MarketEnvironment,
    contract: OptionContract,
    steps_list: list = None,
    american: bool = False,
) -> dict:
    """
    Show how binomial price converges as steps increase.
    """
    if steps_list is None:
        steps_list = [10, 20, 50, 100, 200, 500, 1000]

    results = []
    for n in steps_list:
        r = price_binomial(market, contract, n, american)
        results.append({
            "n_steps": n,
            "price": r["price"],
            "delta": r["delta"],
        })

    return {
        "results": results,
        "option_type": contract.option_type,
        "american": american,
        "model": "binomial-convergence",
    }
