"""
Dividend Adjustments for Option Pricing.

Supports two dividend models:
    1. Continuous dividend yield q (used in BS with q > 0)
    2. Discrete cash dividends with escrowed-dividend model
       (subtract PV of known dividends from spot)

Key formulas:
    Continuous: C = S·e^{-qT}·N(d1) - K·e^{-rT}·N(d2)
    Discrete:   S_adj = S - Σ D_i·e^{-r·t_i}, then price with S_adj
"""

import numpy as np
from scipy.stats import norm


def bs_with_continuous_dividend(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    maturity: float,
    option_type: str = "call",
) -> dict:
    """
    Black-Scholes-Merton price with continuous dividend yield q.
    """
    sq = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility ** 2) * maturity) / (volatility * sq)
    d2 = d1 - volatility * sq

    df_q = np.exp(-dividend_yield * maturity)
    df_r = np.exp(-rate * maturity)

    if option_type == "call":
        price = spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2)
        delta = df_q * norm.cdf(d1)
    else:
        price = strike * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1)
        delta = -df_q * norm.cdf(-d1)

    gamma = df_q * norm.pdf(d1) / (spot * volatility * sq)
    vega = spot * df_q * norm.pdf(d1) * sq / 100

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "dividend_yield": dividend_yield,
        "d1": float(d1),
        "d2": float(d2),
        "option_type": option_type,
        "model": "bs-continuous-dividend",
    }


def bs_with_discrete_dividends(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    dividends: list,  # [(time_in_years, cash_amount), ...]
    option_type: str = "call",
) -> dict:
    """
    Price using escrowed-dividend model: subtract PV of dividends
    from spot, then price with standard BS.

    Parameters
    ----------
    dividends : list of (t, D) tuples
        Each is (time to ex-div in years, cash dividend amount)
    """
    # PV of known dividends
    pv_divs = 0.0
    for t_div, d_amt in dividends:
        if 0 < t_div <= maturity:
            pv_divs += d_amt * np.exp(-rate * t_div)

    # Adjusted spot
    spot_adj = spot - pv_divs

    if spot_adj <= 0:
        return {
            "price": 0.0,
            "spot_adjusted": float(spot_adj),
            "pv_dividends": float(pv_divs),
            "warning": "Adjusted spot non-positive",
            "option_type": option_type,
            "model": "bs-discrete-dividend",
        }

    # Standard BS on adjusted spot
    sq = np.sqrt(maturity)
    d1 = (np.log(spot_adj / strike) + (rate + 0.5 * volatility ** 2) * maturity) / (volatility * sq)
    d2 = d1 - volatility * sq
    df_r = np.exp(-rate * maturity)

    if option_type == "call":
        price = spot_adj * norm.cdf(d1) - strike * df_r * norm.cdf(d2)
    else:
        price = strike * df_r * norm.cdf(-d2) - spot_adj * norm.cdf(-d1)

    # Vanilla (no dividend) for comparison
    sq_v = np.sqrt(maturity)
    d1_v = (np.log(spot / strike) + (rate + 0.5 * volatility ** 2) * maturity) / (volatility * sq_v)
    d2_v = d1_v - volatility * sq_v
    if option_type == "call":
        vanilla = spot * norm.cdf(d1_v) - strike * df_r * norm.cdf(d2_v)
    else:
        vanilla = strike * df_r * norm.cdf(-d2_v) - spot * norm.cdf(-d1_v)

    return {
        "price": float(price),
        "vanilla_price": float(vanilla),
        "dividend_impact": float(price - vanilla),
        "spot_adjusted": float(spot_adj),
        "pv_dividends": float(pv_divs),
        "n_dividends": len(dividends),
        "option_type": option_type,
        "model": "bs-discrete-dividend",
    }


def dividend_schedule(
    annual_yield: float,
    maturity: float,
    frequency: str = "quarterly",
    spot: float = 100.0,
) -> list:
    """
    Generate a schedule of discrete dividends from an annual yield.

    Parameters
    ----------
    frequency : 'monthly', 'quarterly', 'semiannual', 'annual'
    """
    freq_map = {"monthly": 12, "quarterly": 4, "semiannual": 2, "annual": 1}
    n_per_year = freq_map.get(frequency, 4)
    n_total = int(maturity * n_per_year)
    per_div = spot * annual_yield / n_per_year

    schedule = []
    for i in range(1, n_total + 1):
        t = i / n_per_year
        if t <= maturity:
            schedule.append((t, per_div))

    return schedule
