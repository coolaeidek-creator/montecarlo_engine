"""
Greeks Surface Generator.

Computes a 2D grid of Greeks (Delta, Gamma, Vega, Theta) across
spot × maturity or spot × volatility dimensions for visualization.
"""

import numpy as np
from .models import MarketEnvironment, OptionContract
from .greeks import compute_greeks


def greeks_surface_spot_time(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    option_type: str = "call",
    n_spot: int = 15,
    n_time: int = 12,
) -> dict:
    """
    Generate Greeks surface across spot price × time to maturity.

    Returns a dict with 2D arrays for each Greek.
    """
    spots = np.linspace(spot * 0.7, spot * 1.3, n_spot)
    times = np.linspace(0.05, 2.0, n_time)

    delta_grid = []
    gamma_grid = []
    vega_grid = []
    theta_grid = []

    for t in times:
        d_row, g_row, v_row, th_row = [], [], [], []
        for s in spots:
            market = MarketEnvironment(
                spot=float(s), rate=rate,
                volatility=volatility, maturity=float(t),
            )
            contract = OptionContract(strike=strike, option_type=option_type)
            g = compute_greeks(market, contract)
            d_row.append(g["delta"])
            g_row.append(g["gamma"])
            v_row.append(g["vega"])
            th_row.append(g["theta"])
        delta_grid.append(d_row)
        gamma_grid.append(g_row)
        vega_grid.append(v_row)
        theta_grid.append(th_row)

    return {
        "spots": spots.tolist(),
        "times": times.tolist(),
        "delta": delta_grid,
        "gamma": gamma_grid,
        "vega": vega_grid,
        "theta": theta_grid,
        "strike": strike,
        "option_type": option_type,
    }


def greeks_surface_spot_vol(
    spot: float,
    strike: float,
    rate: float,
    maturity: float = 1.0,
    option_type: str = "call",
    n_spot: int = 15,
    n_vol: int = 12,
) -> dict:
    """
    Generate Greeks surface across spot price × volatility.
    """
    spots = np.linspace(spot * 0.7, spot * 1.3, n_spot)
    vols = np.linspace(0.05, 0.80, n_vol)

    delta_grid = []
    gamma_grid = []
    vega_grid = []
    theta_grid = []

    for v in vols:
        d_row, g_row, v_row, th_row = [], [], [], []
        for s in spots:
            market = MarketEnvironment(
                spot=float(s), rate=rate,
                volatility=float(v), maturity=maturity,
            )
            contract = OptionContract(strike=strike, option_type=option_type)
            g = compute_greeks(market, contract)
            d_row.append(g["delta"])
            g_row.append(g["gamma"])
            v_row.append(g["vega"])
            th_row.append(g["theta"])
        delta_grid.append(d_row)
        gamma_grid.append(g_row)
        vega_grid.append(v_row)
        theta_grid.append(th_row)

    return {
        "spots": spots.tolist(),
        "vols": vols.tolist(),
        "delta": delta_grid,
        "gamma": gamma_grid,
        "vega": vega_grid,
        "theta": theta_grid,
        "strike": strike,
        "maturity": maturity,
        "option_type": option_type,
    }
