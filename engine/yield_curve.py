"""
Yield Curve / Term Structure Module.

Models the risk-free rate term structure for more realistic pricing.
Supports Nelson-Siegel parametrization and bootstrapped curves.

Nelson-Siegel: r(T) = β0 + β1·[(1-e^(-T/τ))/(T/τ)]
                     + β2·[(1-e^(-T/τ))/(T/τ) - e^(-T/τ)]
"""

import numpy as np
from .models import MarketEnvironment, OptionContract
from .analytical import bs_price


def nelson_siegel(T, beta0=0.05, beta1=-0.02, beta2=0.03, tau=1.5):
    """
    Nelson-Siegel yield curve model.

    Parameters
    ----------
    T : float or array
        Time to maturity
    beta0 : float
        Long-run rate level
    beta1 : float
        Short-term component (slope)
    beta2 : float
        Medium-term component (curvature)
    tau : float
        Decay parameter
    """
    T = np.asarray(T, dtype=float)
    T = np.maximum(T, 1e-6)  # avoid division by zero

    x = T / tau
    factor1 = (1 - np.exp(-x)) / x
    factor2 = factor1 - np.exp(-x)

    return beta0 + beta1 * factor1 + beta2 * factor2


def flat_curve(T, rate=0.05):
    """Constant rate across all maturities."""
    T = np.asarray(T, dtype=float)
    return np.full_like(T, rate)


def generate_yield_curve(
    model: str = "nelson-siegel",
    rate: float = 0.05,
    n_points: int = 50,
    max_maturity: float = 30.0,
    **kwargs,
) -> dict:
    """
    Generate a yield curve across maturities.

    Parameters
    ----------
    model : str
        "flat", "nelson-siegel", "inverted", "steep"
    """
    maturities = np.linspace(0.1, max_maturity, n_points)

    if model == "flat":
        rates = flat_curve(maturities, rate)
    elif model == "nelson-siegel":
        beta0 = kwargs.get("beta0", rate)
        beta1 = kwargs.get("beta1", -0.02)
        beta2 = kwargs.get("beta2", 0.03)
        tau = kwargs.get("tau", 1.5)
        rates = nelson_siegel(maturities, beta0, beta1, beta2, tau)
    elif model == "inverted":
        # Inverted curve: short rates > long rates
        rates = nelson_siegel(maturities, rate - 0.01, 0.03, -0.02, 1.0)
    elif model == "steep":
        # Steep normal curve
        rates = nelson_siegel(maturities, rate + 0.02, -0.04, 0.05, 2.0)
    else:
        rates = flat_curve(maturities, rate)

    # Discount factors
    discount_factors = np.exp(-rates * maturities)

    # Forward rates (instantaneous)
    forward_rates = np.zeros_like(rates)
    forward_rates[0] = rates[0]
    for i in range(1, len(maturities)):
        dt = maturities[i] - maturities[i - 1]
        forward_rates[i] = (
            rates[i] * maturities[i] - rates[i - 1] * maturities[i - 1]
        ) / dt

    return {
        "maturities": maturities.tolist(),
        "rates": rates.tolist(),
        "discount_factors": discount_factors.tolist(),
        "forward_rates": forward_rates.tolist(),
        "model": model,
    }


def price_with_term_structure(
    spot: float,
    strike: float,
    volatility: float,
    maturity: float,
    option_type: str = "call",
    curve_model: str = "nelson-siegel",
    **curve_params,
) -> dict:
    """
    Price option using term-structure rate instead of flat rate.

    Compares pricing with flat rate vs term-structure rate.
    """
    # Get rate from curve at the option's maturity
    if curve_model == "nelson-siegel":
        curve_rate = float(nelson_siegel(
            maturity,
            curve_params.get("beta0", 0.05),
            curve_params.get("beta1", -0.02),
            curve_params.get("beta2", 0.03),
            curve_params.get("tau", 1.5),
        ))
    elif curve_model == "inverted":
        curve_rate = float(nelson_siegel(maturity, 0.04, 0.03, -0.02, 1.0))
    elif curve_model == "steep":
        curve_rate = float(nelson_siegel(maturity, 0.07, -0.04, 0.05, 2.0))
    else:
        curve_rate = curve_params.get("rate", 0.05)

    # Price with curve rate
    market_curve = MarketEnvironment(
        spot=spot, rate=curve_rate,
        volatility=volatility, maturity=maturity,
    )
    contract = OptionContract(strike=strike, option_type=option_type)
    price_curve = bs_price(market_curve, contract)

    # Price with flat 5% for comparison
    market_flat = MarketEnvironment(
        spot=spot, rate=0.05,
        volatility=volatility, maturity=maturity,
    )
    price_flat = bs_price(market_flat, contract)

    return {
        "curve_rate": curve_rate,
        "curve_price": price_curve,
        "flat_rate": 0.05,
        "flat_price": price_flat,
        "rate_impact": price_curve - price_flat,
        "option_type": option_type,
        "maturity": maturity,
        "model": curve_model,
    }
