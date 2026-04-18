"""
Quanto Option Pricing.

A quanto (quantity-adjusted) option pays off in a currency
different from the underlying's natural currency, with a fixed
FX conversion rate. Used by international investors to get
exposure without FX risk.

Key adjustment: the drift of the underlying in the payoff
currency's risk-neutral measure is reduced by the FX-equity
correlation term.

Modified drift: μ' = r_f - q - ρ·σ_S·σ_FX

where:
    r_f     = foreign (payoff currency) risk-free rate
    q       = dividend yield of underlying
    ρ       = correlation between underlying and FX
    σ_S     = underlying volatility
    σ_FX    = FX volatility
"""

import numpy as np
from scipy.stats import norm
from .models import MarketEnvironment, OptionContract


def quanto_bs_price(
    spot: float,
    strike: float,
    rate_domestic: float,
    rate_foreign: float,
    volatility: float,
    fx_volatility: float,
    correlation: float,
    maturity: float,
    option_type: str = "call",
    dividend: float = 0.0,
) -> dict:
    """
    Price a quanto option using closed-form (modified Black-Scholes).

    Parameters
    ----------
    spot : float
        Spot price in foreign currency
    strike : float
        Strike in foreign currency
    rate_domestic : float
        Foreign (payoff) currency risk-free rate
    rate_foreign : float
        Domestic (underlying) currency risk-free rate
    volatility : float
        Underlying (equity) volatility
    fx_volatility : float
        FX rate volatility
    correlation : float
        Correlation between equity and FX returns
    maturity : float
        Time to expiry (years)
    option_type : str
        'call' or 'put'
    dividend : float
        Continuous dividend yield
    """
    # Quanto-adjusted drift
    q_adj = rate_foreign - rate_domestic + correlation * volatility * fx_volatility + dividend
    mu_q = rate_domestic - q_adj

    # Modified BS formula
    sq = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (mu_q + 0.5 * volatility ** 2) * maturity) / (volatility * sq)
    d2 = d1 - volatility * sq

    df_foreign = np.exp(-q_adj * maturity)
    df_domestic = np.exp(-rate_domestic * maturity)

    if option_type == "call":
        price = spot * df_foreign * norm.cdf(d1) - strike * df_domestic * norm.cdf(d2)
    else:
        price = strike * df_domestic * norm.cdf(-d2) - spot * df_foreign * norm.cdf(-d1)

    # Compare with vanilla (no quanto adjustment)
    vanilla_d1 = (np.log(spot / strike) + (rate_foreign + 0.5 * volatility ** 2) * maturity) / (volatility * sq)
    vanilla_d2 = vanilla_d1 - volatility * sq
    if option_type == "call":
        vanilla_price = spot * np.exp(-dividend * maturity) * norm.cdf(vanilla_d1) - strike * np.exp(-rate_foreign * maturity) * norm.cdf(vanilla_d2)
    else:
        vanilla_price = strike * np.exp(-rate_foreign * maturity) * norm.cdf(-vanilla_d2) - spot * np.exp(-dividend * maturity) * norm.cdf(-vanilla_d1)

    return {
        "price": float(price),
        "vanilla_price": float(vanilla_price),
        "quanto_adjustment": float(price - vanilla_price),
        "adjusted_drift": float(mu_q),
        "quanto_div_yield": float(q_adj),
        "correlation": correlation,
        "fx_volatility": fx_volatility,
        "option_type": option_type,
        "model": "quanto-bs",
    }


def quanto_mc(
    spot: float,
    strike: float,
    rate_domestic: float,
    rate_foreign: float,
    volatility: float,
    fx_volatility: float,
    correlation: float,
    maturity: float,
    option_type: str = "call",
    n_simulations: int = 50000,
) -> dict:
    """
    Price quanto option via Monte Carlo using correlated GBM
    for both the underlying and FX rate.
    """
    # Adjusted drift in foreign measure
    q_adj = rate_foreign - rate_domestic + correlation * volatility * fx_volatility
    mu_q = rate_domestic - q_adj

    # Simulate terminal prices
    z = np.random.standard_normal(n_simulations)
    ST = spot * np.exp(
        (mu_q - 0.5 * volatility ** 2) * maturity
        + volatility * np.sqrt(maturity) * z
    )

    # Payoff
    if option_type == "call":
        payoffs = np.maximum(ST - strike, 0)
    else:
        payoffs = np.maximum(strike - ST, 0)

    discount = np.exp(-rate_domestic * maturity)
    price = discount * np.mean(payoffs)
    se = discount * np.std(payoffs) / np.sqrt(n_simulations)

    return {
        "price": float(price),
        "std_error": float(se),
        "confidence_interval": (float(price - 1.96 * se), float(price + 1.96 * se)),
        "n_simulations": n_simulations,
        "option_type": option_type,
        "model": "quanto-mc",
    }
