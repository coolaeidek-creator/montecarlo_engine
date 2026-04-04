"""
Multi-step GBM path simulator for path-dependent options.

Generates full price paths (not just terminal values) needed for
Asian, Barrier, and Lookback options.
"""

import numpy as np
from .models import MarketEnvironment


def simulate_paths(
    market: MarketEnvironment,
    n_simulations: int = 10000,
    n_steps: int = 252,
    antithetic: bool = False,
) -> np.ndarray:
    """
    Simulate full GBM price paths.

    S(t+dt) = S(t) * exp((r - 0.5*σ²)*dt + σ*√dt*Z)

    Parameters
    ----------
    market : MarketEnvironment
    n_simulations : int
        Number of independent paths
    n_steps : int
        Number of time steps (252 = daily for 1yr)
    antithetic : bool
        If True, generates n_simulations paths using antithetic variates

    Returns
    -------
    np.ndarray of shape (n_simulations, n_steps + 1)
        Full price paths including S(0) at index 0
    """
    dt = market.maturity / n_steps
    drift = (market.rate - 0.5 * market.volatility ** 2) * dt
    diffusion = market.volatility * np.sqrt(dt)

    if antithetic:
        half = n_simulations // 2
        z = np.random.standard_normal((half, n_steps))
        z = np.concatenate([z, -z], axis=0)  # (n_simulations, n_steps)
    else:
        z = np.random.standard_normal((n_simulations, n_steps))

    # Log-returns for each step
    log_returns = drift + diffusion * z  # (n_sim, n_steps)

    # Cumulative sum of log-returns → cumulative log price
    log_paths = np.cumsum(log_returns, axis=1)

    # Prepend zero (log(S0/S0) = 0) and exponentiate
    log_paths = np.concatenate(
        [np.zeros((z.shape[0], 1)), log_paths], axis=1
    )
    paths = market.spot * np.exp(log_paths)

    return paths


def simulate_paths_with_times(
    market: MarketEnvironment,
    n_simulations: int = 10000,
    n_steps: int = 252,
    antithetic: bool = False,
) -> tuple:
    """
    Returns (paths, time_grid) tuple.

    time_grid : np.ndarray of shape (n_steps + 1,)
        Time points from 0 to T
    """
    paths = simulate_paths(market, n_simulations, n_steps, antithetic)
    time_grid = np.linspace(0, market.maturity, n_steps + 1)
    return paths, time_grid
