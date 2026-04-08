"""
SABR Stochastic Alpha Beta Rho Model.

The SABR model is the industry-standard for interest rate and FX
options volatility modeling. It provides analytical implied vol
approximations via the Hagan et al. (2002) formula.

dF = σ · F^β · dW_1
dσ = α · σ · dW_2
Corr(dW_1, dW_2) = ρ

Parameters:
    α (alpha) — vol-of-vol
    β (beta)  — CEV exponent (0=normal, 1=lognormal)
    ρ (rho)   — forward-vol correlation
    σ_0       — initial volatility
"""

import numpy as np
from .models import MarketEnvironment, OptionContract
from .analytical import bs_price


def sabr_implied_vol(
    F: float,
    K: float,
    T: float,
    alpha: float = 0.3,
    beta: float = 0.7,
    rho: float = -0.25,
    sigma0: float = 0.25,
) -> float:
    """
    SABR implied volatility via Hagan et al. (2002) approximation.

    Parameters
    ----------
    F : float — forward price
    K : float — strike
    T : float — time to expiry
    alpha : float — vol of vol
    beta : float — CEV exponent [0,1]
    rho : float — forward-vol correlation [-1,1]
    sigma0 : float — initial stochastic vol
    """
    if F <= 0 or K <= 0 or T <= 0:
        return 0.0

    # ATM case
    if abs(F - K) < 1e-10:
        FK_mid = F
        logFK = 0.0
        # ATM formula
        term1 = sigma0 / (FK_mid ** (1 - beta))
        term2 = (
            ((1 - beta) ** 2 / 24) * sigma0 ** 2 / (FK_mid ** (2 * (1 - beta)))
            + 0.25 * rho * beta * alpha * sigma0 / (FK_mid ** (1 - beta))
            + (2 - 3 * rho ** 2) / 24 * alpha ** 2
        )
        return term1 * (1 + term2 * T)

    # General case
    FK_mid = (F * K) ** ((1 - beta) / 2)
    logFK = np.log(F / K)

    z = (alpha / sigma0) * FK_mid * logFK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    if abs(x_z) < 1e-10:
        x_z = 1.0
        z_over_x = 1.0
    else:
        z_over_x = z / x_z

    # Prefix
    FK_beta = (F * K) ** ((1 - beta) / 2)
    prefix = sigma0 / (
        FK_beta
        * (1 + (1 - beta) ** 2 / 24 * logFK ** 2
           + (1 - beta) ** 4 / 1920 * logFK ** 4)
    )

    # Correction term
    term2 = (
        ((1 - beta) ** 2 / 24) * sigma0 ** 2 / ((F * K) ** (1 - beta))
        + 0.25 * rho * beta * alpha * sigma0 / FK_beta
        + (2 - 3 * rho ** 2) / 24 * alpha ** 2
    )

    return prefix * z_over_x * (1 + term2 * T)


def sabr_smile(
    spot: float,
    rate: float,
    maturity: float,
    alpha: float = 0.3,
    beta: float = 0.7,
    rho: float = -0.25,
    sigma0: float = 0.25,
    n_strikes: int = 15,
) -> dict:
    """
    Generate volatility smile from SABR model.

    Returns strikes, moneyness, implied vols, and call prices.
    """
    F = spot * np.exp(rate * maturity)  # Forward price
    moneyness = np.linspace(0.75, 1.25, n_strikes)
    strikes = spot * moneyness

    ivs = []
    prices = []

    for K in strikes:
        iv = sabr_implied_vol(F, float(K), maturity, alpha, beta, rho, sigma0)
        iv = max(0.001, iv)  # floor
        ivs.append(iv)

        # BS price with SABR IV
        market = MarketEnvironment(
            spot=spot, rate=rate, volatility=iv, maturity=maturity,
        )
        contract = OptionContract(strike=float(K), option_type="call")
        p = bs_price(market, contract)
        prices.append(p)

    return {
        "strikes": strikes.tolist(),
        "moneyness": moneyness.tolist(),
        "implied_vols": ivs,
        "prices": prices,
        "forward": F,
        "params": {
            "alpha": alpha,
            "beta": beta,
            "rho": rho,
            "sigma0": sigma0,
        },
        "model": "sabr",
    }


def sabr_surface(
    spot: float,
    rate: float,
    alpha: float = 0.3,
    beta: float = 0.7,
    rho: float = -0.25,
    sigma0: float = 0.25,
    n_strikes: int = 11,
    maturities: list = None,
) -> dict:
    """
    Generate full SABR volatility surface across strikes and maturities.
    """
    if maturities is None:
        maturities = [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

    moneyness = np.linspace(0.80, 1.20, n_strikes)
    surface = []

    for T in maturities:
        F = spot * np.exp(rate * T)
        row = []
        for m in moneyness:
            K = spot * m
            iv = sabr_implied_vol(F, K, T, alpha, beta, rho, sigma0)
            row.append(max(0.001, iv))
        surface.append(row)

    return {
        "surface": surface,
        "moneyness": moneyness.tolist(),
        "maturities": maturities,
        "spot": spot,
        "params": {"alpha": alpha, "beta": beta, "rho": rho, "sigma0": sigma0},
        "model": "sabr-surface",
    }
