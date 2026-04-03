import numpy as np
from .models import MarketEnvironment


def simulate_terminal_prices(
    market: MarketEnvironment,
    shocks: np.ndarray,
) -> np.ndarray:
    """
    Simulate terminal stock prices using
    geometric Brownian motion.

    S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z)
    """

    drift = (market.rate - 0.5 * market.volatility**2) * market.maturity
    diffusion = market.volatility * np.sqrt(market.maturity) * shocks

    terminal_prices = market.spot * np.exp(drift + diffusion)

    return terminal_prices
