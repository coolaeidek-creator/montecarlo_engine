"""
Implied Volatility Solver.

Uses Newton-Raphson iteration with BS Vega as the derivative
to find the volatility that makes BS price = market price.

Also provides:
- Bisection fallback for robustness
- IV surface computation across strikes and maturities
"""

import numpy as np
from scipy.stats import norm
from .models import MarketEnvironment, OptionContract


def bs_price_for_iv(S, K, r, T, sigma, option_type):
    """Inline BS price for IV solver (avoids circular imports)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_vega(S, K, r, T, sigma):
    """BS Vega = S * sqrt(T) * N'(d1). Used as Newton-Raphson derivative."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * norm.pdf(d1)


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    option_type: str = "call",
    initial_guess: float = 0.25,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> dict:
    """
    Find implied volatility using Newton-Raphson.

    Given a market-observed option price, solve for σ such that:
        BS(S, K, r, T, σ) = market_price

    Falls back to bisection if Newton-Raphson diverges.

    Parameters
    ----------
    market_price : float
        Observed option price in the market
    spot, strike, rate, maturity : float
        Market parameters
    option_type : str
        'call' or 'put'

    Returns
    -------
    dict with iv, converged, iterations, method
    """
    sigma = initial_guess

    # Newton-Raphson
    for i in range(max_iter):
        price = bs_price_for_iv(spot, strike, rate, maturity, sigma, option_type)
        vega = bs_vega(spot, strike, rate, maturity, sigma)

        if vega < 1e-12:
            break  # vega too small, switch to bisection

        diff = price - market_price
        if abs(diff) < tol:
            return {
                "iv": sigma,
                "converged": True,
                "iterations": i + 1,
                "method": "newton-raphson",
                "price_error": abs(diff),
            }

        sigma = sigma - diff / vega

        # Guard against negative or extreme sigma
        if sigma <= 0.001 or sigma > 5.0:
            break

    # Fallback: bisection method
    return _bisection_iv(
        market_price, spot, strike, rate, maturity, option_type, tol, max_iter
    )


def _bisection_iv(market_price, spot, strike, rate, maturity, option_type, tol, max_iter):
    """Bisection fallback — always converges if IV exists in [0.01, 5.0]."""
    lo, hi = 0.01, 5.0

    for i in range(max_iter):
        mid = (lo + hi) / 2.0
        price = bs_price_for_iv(spot, strike, rate, maturity, mid, option_type)
        diff = price - market_price

        if abs(diff) < tol:
            return {
                "iv": mid,
                "converged": True,
                "iterations": i + 1,
                "method": "bisection",
                "price_error": abs(diff),
            }

        if diff > 0:
            hi = mid
        else:
            lo = mid

    return {
        "iv": (lo + hi) / 2.0,
        "converged": False,
        "iterations": max_iter,
        "method": "bisection",
        "price_error": abs(diff),
    }


def iv_surface(
    spot: float,
    rate: float,
    strikes: list,
    maturities: list,
    market_prices: np.ndarray,
    option_type: str = "call",
) -> np.ndarray:
    """
    Compute an implied volatility surface.

    Parameters
    ----------
    strikes : list of float
        Strike prices (columns)
    maturities : list of float
        Maturities in years (rows)
    market_prices : np.ndarray of shape (len(maturities), len(strikes))
        Observed option prices grid

    Returns
    -------
    np.ndarray of shape (len(maturities), len(strikes))
        IV surface (σ values)
    """
    n_mat = len(maturities)
    n_str = len(strikes)
    surface = np.zeros((n_mat, n_str))

    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            result = implied_volatility(
                market_price=market_prices[i, j],
                spot=spot, strike=K, rate=rate, maturity=T,
                option_type=option_type,
            )
            surface[i, j] = result["iv"] if result["converged"] else np.nan

    return surface


def compute_smile(
    spot: float,
    rate: float,
    maturity: float,
    strikes: list,
    market_prices: list,
    option_type: str = "call",
) -> list:
    """
    Compute volatility smile for a single maturity across strikes.

    Returns list of dicts with strike, iv, moneyness.
    """
    results = []
    for K, price in zip(strikes, market_prices):
        iv_result = implied_volatility(
            market_price=price, spot=spot, strike=K,
            rate=rate, maturity=maturity, option_type=option_type,
        )
        results.append({
            "strike": K,
            "iv": iv_result["iv"],
            "converged": iv_result["converged"],
            "moneyness": K / spot,
        })
    return results
