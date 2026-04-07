"""
Scenario Analysis & Stress Testing Module.

Provides:
- Greeks sensitivity tables (how price changes with spot/vol/time)
- Stress testing (what happens in extreme market events)
- P&L scenario matrix across spot x vol grid
- Time decay projection
"""

import numpy as np
from .models import MarketEnvironment, OptionContract
from .analytical import bs_price
from .greeks import compute_greeks


def spot_sensitivity(
    market: MarketEnvironment,
    contract: OptionContract,
    spot_range: tuple = (0.7, 1.3),
    n_points: int = 25,
) -> dict:
    """
    How option price and Greeks change as spot price varies.

    Returns arrays of prices, deltas, gammas at each spot level.
    """
    spots = np.linspace(market.spot * spot_range[0], market.spot * spot_range[1], n_points)
    prices, deltas, gammas, thetas, vegas = [], [], [], [], []

    for s in spots:
        m = MarketEnvironment(spot=s, rate=market.rate,
                              volatility=market.volatility, maturity=market.maturity)
        prices.append(bs_price(m, contract))
        g = compute_greeks(m, contract)
        deltas.append(g["delta"])
        gammas.append(g["gamma"])
        thetas.append(g["theta"])
        vegas.append(g["vega"])

    return {
        "spots": spots.tolist(),
        "prices": prices,
        "deltas": deltas,
        "gammas": gammas,
        "thetas": thetas,
        "vegas": vegas,
        "strike": contract.strike,
        "option_type": contract.option_type,
    }


def vol_sensitivity(
    market: MarketEnvironment,
    contract: OptionContract,
    vol_range: tuple = (0.05, 0.80),
    n_points: int = 20,
) -> dict:
    """How option price changes as implied volatility varies."""
    vols = np.linspace(vol_range[0], vol_range[1], n_points)
    prices = []

    for v in vols:
        m = MarketEnvironment(spot=market.spot, rate=market.rate,
                              volatility=v, maturity=market.maturity)
        prices.append(bs_price(m, contract))

    return {
        "vols": vols.tolist(),
        "prices": prices,
    }


def time_decay_projection(
    market: MarketEnvironment,
    contract: OptionContract,
    n_points: int = 50,
) -> dict:
    """Project option value as time passes (theta decay curve)."""
    times = np.linspace(market.maturity, 0.001, n_points)
    prices, thetas = [], []

    for t in times:
        m = MarketEnvironment(spot=market.spot, rate=market.rate,
                              volatility=market.volatility, maturity=t)
        prices.append(bs_price(m, contract))
        g = compute_greeks(m, contract)
        thetas.append(g["theta"])

    return {
        "times": times.tolist(),
        "days_remaining": (times * 252).tolist(),
        "prices": prices,
        "thetas": thetas,
    }


def pnl_matrix(
    market: MarketEnvironment,
    contract: OptionContract,
    spot_shifts: list = None,
    vol_shifts: list = None,
) -> dict:
    """
    P&L matrix across spot and vol shifts.

    Shows how option value changes under different spot/vol scenarios.
    Returns a 2D grid of P&L values.
    """
    if spot_shifts is None:
        spot_shifts = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    if vol_shifts is None:
        vol_shifts = [-10, -5, 0, 5, 10]

    current_price = bs_price(market, contract)
    grid = []

    for ds in spot_shifts:
        row = []
        for dv in vol_shifts:
            new_spot = market.spot * (1 + ds / 100)
            new_vol = max(0.01, market.volatility + dv / 100)
            m = MarketEnvironment(spot=new_spot, rate=market.rate,
                                  volatility=new_vol, maturity=market.maturity)
            new_price = bs_price(m, contract)
            row.append(new_price - current_price)
        grid.append(row)

    return {
        "spot_shifts": spot_shifts,
        "vol_shifts": vol_shifts,
        "pnl_grid": grid,
        "current_price": current_price,
    }


def stress_test(
    market: MarketEnvironment,
    contract: OptionContract,
) -> list:
    """
    Run predefined stress scenarios (historical-style shocks).

    Returns P&L under each scenario.
    """
    current_price = bs_price(market, contract)

    scenarios = [
        {"name": "Black Monday (1987)", "spot_shock": -0.225, "vol_shock": 0.30},
        {"name": "Dot-Com Crash (2000)", "spot_shock": -0.10, "vol_shock": 0.15},
        {"name": "GFC (2008)", "spot_shock": -0.15, "vol_shock": 0.25},
        {"name": "COVID Crash (2020)", "spot_shock": -0.12, "vol_shock": 0.35},
        {"name": "Flash Crash (2010)", "spot_shock": -0.09, "vol_shock": 0.20},
        {"name": "Bull Rally (+10%)", "spot_shock": 0.10, "vol_shock": -0.05},
        {"name": "Melt-Up (+20%)", "spot_shock": 0.20, "vol_shock": -0.08},
        {"name": "Vol Spike (flat)", "spot_shock": 0.0, "vol_shock": 0.20},
        {"name": "Vol Crush (flat)", "spot_shock": 0.0, "vol_shock": -0.10},
    ]

    results = []
    for sc in scenarios:
        new_spot = market.spot * (1 + sc["spot_shock"])
        new_vol = max(0.01, market.volatility + sc["vol_shock"])
        m = MarketEnvironment(spot=new_spot, rate=market.rate,
                              volatility=new_vol, maturity=market.maturity)
        new_price = bs_price(m, contract)
        pnl = new_price - current_price

        results.append({
            "name": sc["name"],
            "spot_shock_pct": sc["spot_shock"] * 100,
            "vol_shock_pct": sc["vol_shock"] * 100,
            "new_price": new_price,
            "pnl": pnl,
            "pnl_pct": (pnl / current_price) * 100 if current_price > 0 else 0,
        })

    return results
