"""
Volatility Surface Generator.

Generates synthetic but realistic volatility surfaces using
industry-standard parameterizations:
- SVI (Stochastic Volatility Inspired) parametrization
- Simple skew model
- Term structure interpolation

Used when real market IV data isn't available — produces
realistic smile/skew shapes for demo and testing.
"""

import numpy as np


def svi_slice(
    k: np.ndarray,
    a: float = 0.04,
    b: float = 0.15,
    rho: float = -0.3,
    m: float = 0.0,
    sigma: float = 0.3,
) -> np.ndarray:
    """
    SVI (Stochastic Volatility Inspired) parametrization.

    Total implied variance w(k) = a + b * (ρ*(k-m) + √((k-m)² + σ²))

    where k = log(K/F) is log-moneyness.

    Parameters
    ----------
    k : np.ndarray
        Log-moneyness values
    a, b, rho, m, sigma : float
        SVI parameters

    Returns
    -------
    np.ndarray
        Total implied variance at each k
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def generate_vol_surface(
    spot: float,
    rate: float,
    base_vol: float = 0.25,
    strikes_pct: np.ndarray = None,
    maturities: np.ndarray = None,
    skew: float = -0.15,
    smile: float = 0.05,
    term_slope: float = -0.03,
) -> dict:
    """
    Generate a synthetic but realistic volatility surface.

    Uses a simplified model:
        σ(K, T) = base_vol + skew*(K/S - 1) + smile*(K/S - 1)² + term_slope*√T

    This produces:
    - Negative skew (puts more expensive — like equity markets)
    - Smile curvature at wings
    - Declining vol term structure (contango-like)

    Parameters
    ----------
    spot : float
        Current stock price
    base_vol : float
        ATM volatility level
    strikes_pct : np.ndarray
        Strike prices as % of spot [0.7, 0.8, ..., 1.3]
    maturities : np.ndarray
        Maturities in years [0.083, 0.25, 0.5, 1.0, 2.0]
    skew : float
        Skew parameter (negative for equity-like)
    smile : float
        Curvature parameter
    term_slope : float
        Term structure slope

    Returns
    -------
    dict with surface, strikes, maturities, spot
    """
    if strikes_pct is None:
        strikes_pct = np.arange(0.70, 1.35, 0.05)
    if maturities is None:
        maturities = np.array([1/12, 2/12, 3/12, 6/12, 1.0, 2.0])

    n_mat = len(maturities)
    n_str = len(strikes_pct)
    surface = np.zeros((n_mat, n_str))
    strikes = strikes_pct * spot

    for i, T in enumerate(maturities):
        for j, k_pct in enumerate(strikes_pct):
            moneyness = k_pct - 1.0  # deviation from ATM
            vol = (
                base_vol
                + skew * moneyness
                + smile * moneyness ** 2
                + term_slope * (np.sqrt(T) - np.sqrt(0.25))  # relative to 3M
            )
            surface[i, j] = max(vol, 0.01)  # floor at 1%

    return {
        "surface": surface,
        "strikes": strikes,
        "strikes_pct": strikes_pct,
        "maturities": maturities,
        "spot": spot,
        "base_vol": base_vol,
    }


def generate_svi_surface(
    spot: float,
    rate: float,
    maturities: np.ndarray = None,
    n_strikes: int = 21,
    strike_range: tuple = (0.7, 1.3),
) -> dict:
    """
    Generate a vol surface using SVI parametrization per slice.

    More realistic than the simple model — produces arbitrage-free
    (approximately) surfaces commonly used on trading desks.
    """
    if maturities is None:
        maturities = np.array([1/12, 2/12, 3/12, 6/12, 1.0, 2.0])

    strikes_pct = np.linspace(strike_range[0], strike_range[1], n_strikes)
    strikes = strikes_pct * spot
    forward = spot * np.exp(rate * maturities[-1])  # simplified

    n_mat = len(maturities)
    surface = np.zeros((n_mat, n_strikes))

    for i, T in enumerate(maturities):
        F = spot * np.exp(rate * T)
        k = np.log(strikes / F)  # log-moneyness

        # SVI params evolve with maturity
        a = 0.02 + 0.01 * np.sqrt(T)
        b = 0.20 - 0.03 * np.sqrt(T)
        rho = -0.30 + 0.05 * T
        m = -0.02
        sig = 0.20 + 0.10 * T

        total_var = svi_slice(k, a, b, rho, m, sig)
        total_var = np.maximum(total_var, 0.001)
        surface[i, :] = np.sqrt(total_var / T)

    return {
        "surface": surface,
        "strikes": strikes,
        "strikes_pct": strikes_pct,
        "maturities": maturities,
        "spot": spot,
        "model": "SVI",
    }


def print_surface(surface_data: dict, currency: str = "$"):
    """Pretty-print a vol surface to terminal."""
    surface = surface_data["surface"]
    strikes = surface_data["strikes"]
    mats = surface_data["maturities"]

    # Header
    header = f"{'Expiry':>8}"
    for K in strikes[::2]:  # every other strike for readability
        header += f"  {currency}{K:>8.0f}"
    print(header)
    print("─" * len(header))

    for i, T in enumerate(mats):
        if T < 1:
            label = f"{T*12:.0f}M"
        else:
            label = f"{T:.1f}Y"

        row = f"{label:>8}"
        for j in range(0, len(strikes), 2):
            row += f"  {surface[i, j]*100:>8.1f}%"
        print(row)
