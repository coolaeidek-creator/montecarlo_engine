"""
Exotic options pricing via Monte Carlo simulation.

Supports:
- Asian options (arithmetic & geometric average)
- Barrier options (up-and-out, up-and-in, down-and-out, down-and-in)
- Lookback options (floating strike)
- Digital/Binary options (cash-or-nothing)

All use full path simulation for accurate path-dependent pricing.
"""

import numpy as np
from .models import MarketEnvironment, OptionContract
from .paths import simulate_paths


# ─── Asian Options ────────────────────────────────────────────────────────────

def price_asian(
    market: MarketEnvironment,
    contract: OptionContract,
    n_simulations: int = 50000,
    n_steps: int = 252,
    averaging: str = "arithmetic",
    antithetic: bool = True,
) -> dict:
    """
    Price an Asian (average price) option.

    The payoff depends on the average price over the path:
      Call: max(A - K, 0)
      Put:  max(K - A, 0)

    where A = average of S along the path.

    Parameters
    ----------
    averaging : str
        'arithmetic' or 'geometric'
    """
    paths = simulate_paths(market, n_simulations, n_steps, antithetic)
    discount = np.exp(-market.rate * market.maturity)

    # Compute average along each path (exclude S(0) for standard convention)
    if averaging == "geometric":
        avg_prices = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    else:
        avg_prices = np.mean(paths[:, 1:], axis=1)

    # Payoff
    if contract.option_type == "call":
        payoffs = np.maximum(avg_prices - contract.strike, 0)
    else:
        payoffs = np.maximum(contract.strike - avg_prices, 0)

    pv = payoffs * discount
    price = np.mean(pv)
    se = np.std(pv) / np.sqrt(len(pv))

    return {
        "price": price,
        "std_error": se,
        "confidence_interval": (price - 1.96 * se, price + 1.96 * se),
        "averaging": averaging,
        "option_type": contract.option_type,
        "n_simulations": n_simulations,
        "n_steps": n_steps,
        "exotic_type": "asian",
    }


# ─── Barrier Options ─────────────────────────────────────────────────────────

def price_barrier(
    market: MarketEnvironment,
    contract: OptionContract,
    barrier: float,
    barrier_type: str = "down-and-out",
    n_simulations: int = 50000,
    n_steps: int = 252,
    antithetic: bool = True,
) -> dict:
    """
    Price a barrier option.

    Barrier types:
    - 'up-and-out':   knocked out if S ever goes ABOVE barrier
    - 'up-and-in':    activated only if S goes ABOVE barrier
    - 'down-and-out': knocked out if S ever goes BELOW barrier
    - 'down-and-in':  activated only if S goes BELOW barrier

    Parameters
    ----------
    barrier : float
        The barrier level
    barrier_type : str
        One of 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
    """
    paths = simulate_paths(market, n_simulations, n_steps, antithetic)
    discount = np.exp(-market.rate * market.maturity)
    terminal = paths[:, -1]

    # Determine if barrier was hit on each path
    max_prices = np.max(paths, axis=1)
    min_prices = np.min(paths, axis=1)

    if barrier_type == "up-and-out":
        alive = max_prices < barrier
    elif barrier_type == "up-and-in":
        alive = max_prices >= barrier
    elif barrier_type == "down-and-out":
        alive = min_prices > barrier
    elif barrier_type == "down-and-in":
        alive = min_prices <= barrier
    else:
        raise ValueError(f"Unknown barrier_type: {barrier_type}")

    # Vanilla payoff
    if contract.option_type == "call":
        payoffs = np.maximum(terminal - contract.strike, 0)
    else:
        payoffs = np.maximum(contract.strike - terminal, 0)

    # Apply barrier condition
    payoffs = payoffs * alive
    pv = payoffs * discount
    price = np.mean(pv)
    se = np.std(pv) / np.sqrt(len(pv))

    # Knock-out probability
    knocked_pct = 1.0 - np.mean(alive)

    return {
        "price": price,
        "std_error": se,
        "confidence_interval": (price - 1.96 * se, price + 1.96 * se),
        "barrier": barrier,
        "barrier_type": barrier_type,
        "knock_probability": knocked_pct,
        "option_type": contract.option_type,
        "n_simulations": n_simulations,
        "n_steps": n_steps,
        "exotic_type": "barrier",
    }


# ─── Lookback Options ────────────────────────────────────────────────────────

def price_lookback(
    market: MarketEnvironment,
    contract: OptionContract,
    n_simulations: int = 50000,
    n_steps: int = 252,
    antithetic: bool = True,
) -> dict:
    """
    Price a floating-strike lookback option.

    Payoff:
      Call: S_T - min(S)   (buy at the lowest)
      Put:  max(S) - S_T   (sell at the highest)

    These are always in-the-money (except degenerate cases),
    so they are more expensive than vanilla options.
    """
    paths = simulate_paths(market, n_simulations, n_steps, antithetic)
    discount = np.exp(-market.rate * market.maturity)
    terminal = paths[:, -1]

    if contract.option_type == "call":
        min_prices = np.min(paths, axis=1)
        payoffs = terminal - min_prices
    else:
        max_prices = np.max(paths, axis=1)
        payoffs = max_prices - terminal

    payoffs = np.maximum(payoffs, 0)
    pv = payoffs * discount
    price = np.mean(pv)
    se = np.std(pv) / np.sqrt(len(pv))

    return {
        "price": price,
        "std_error": se,
        "confidence_interval": (price - 1.96 * se, price + 1.96 * se),
        "option_type": contract.option_type,
        "n_simulations": n_simulations,
        "n_steps": n_steps,
        "exotic_type": "lookback",
    }


# ─── Digital / Binary Options ────────────────────────────────────────────────

def price_digital(
    market: MarketEnvironment,
    contract: OptionContract,
    payout: float = 1.0,
    n_simulations: int = 50000,
    antithetic: bool = True,
) -> dict:
    """
    Price a cash-or-nothing digital (binary) option.

    Pays fixed amount if option expires ITM, zero otherwise.
      Call: payout if S_T > K
      Put:  payout if S_T < K
    """
    paths = simulate_paths(market, n_simulations, 1, antithetic)
    discount = np.exp(-market.rate * market.maturity)
    terminal = paths[:, -1]

    if contract.option_type == "call":
        payoffs = np.where(terminal > contract.strike, payout, 0.0)
    else:
        payoffs = np.where(terminal < contract.strike, payout, 0.0)

    pv = payoffs * discount
    price = np.mean(pv)
    se = np.std(pv) / np.sqrt(len(pv))

    itm_prob = np.mean(payoffs > 0)

    return {
        "price": price,
        "std_error": se,
        "confidence_interval": (price - 1.96 * se, price + 1.96 * se),
        "payout": payout,
        "itm_probability": itm_prob,
        "option_type": contract.option_type,
        "n_simulations": n_simulations,
        "exotic_type": "digital",
    }
