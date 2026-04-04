"""
Black-Scholes Greeks — option sensitivities.

All Greeks computed analytically via closed-form BS formulas.
"""

import numpy as np
from scipy.stats import norm
from .models import MarketEnvironment, OptionContract


def compute_greeks(market: MarketEnvironment, contract: OptionContract) -> dict:
    """
    Compute all BS Greeks for a European option.

    Returns dict with: delta, gamma, theta, vega, rho
    """
    S, K = market.spot, contract.strike
    r, sigma, T = market.rate, market.volatility, market.maturity
    sqrt_T = np.sqrt(T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    phi_d1 = norm.pdf(d1)  # standard normal PDF
    df = np.exp(-r * T)

    is_call = contract.option_type == "call"

    # Delta: dC/dS
    delta = norm.cdf(d1) if is_call else norm.cdf(d1) - 1

    # Gamma: d²C/dS² (same for call and put)
    gamma = phi_d1 / (S * sigma * sqrt_T)

    # Theta: dC/dt (per calendar day)
    theta_term1 = -(S * phi_d1 * sigma) / (2 * sqrt_T)
    if is_call:
        theta = (theta_term1 - r * K * df * norm.cdf(d2)) / 365
    else:
        theta = (theta_term1 + r * K * df * norm.cdf(-d2)) / 365

    # Vega: dC/dσ (per 1% vol move)
    vega = S * phi_d1 * sqrt_T / 100

    # Rho: dC/dr (per 1% rate move)
    if is_call:
        rho = K * T * df * norm.cdf(d2) / 100
    else:
        rho = -K * T * df * norm.cdf(-d2) / 100

    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "d1": d1,
        "d2": d2,
    }


def compute_greeks_both(market: MarketEnvironment, strike: float) -> dict:
    """Compute Greeks for both call and put at given strike."""
    call_c = OptionContract(strike=strike, option_type="call")
    put_c = OptionContract(strike=strike, option_type="put")
    return {
        "call": compute_greeks(market, call_c),
        "put": compute_greeks(market, put_c),
    }
