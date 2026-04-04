"""
Black-Scholes analytical pricing engine.

Provides closed-form solutions for European option prices,
used as a benchmark to validate Monte Carlo results.
"""

import numpy as np
from scipy.stats import norm
from .models import MarketEnvironment, OptionContract


def bs_price(market: MarketEnvironment, contract: OptionContract) -> float:
    """
    Black-Scholes closed-form European option price.

    Call = S·N(d1) - K·e^(-rT)·N(d2)
    Put  = K·e^(-rT)·N(-d2) - S·N(-d1)
    """
    S, K = market.spot, contract.strike
    r, sigma, T = market.rate, market.volatility, market.maturity

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if contract.option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_price_both(market: MarketEnvironment, strike: float) -> dict:
    """Price both call and put at once."""
    call_c = OptionContract(strike=strike, option_type="call")
    put_c = OptionContract(strike=strike, option_type="put")
    return {
        "call": bs_price(market, call_c),
        "put": bs_price(market, put_c),
    }
