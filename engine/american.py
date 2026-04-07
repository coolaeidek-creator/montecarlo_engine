"""
American Option Pricing via Least-Squares Monte Carlo (LSM).

Implements the Longstaff-Schwartz algorithm for pricing
American options that can be exercised early.

The key idea: at each time step, regress the continuation value
on basis functions of the stock price to decide whether to exercise.
"""

import numpy as np
from .models import MarketEnvironment, OptionContract
from .paths import simulate_paths


def price_american(
    market: MarketEnvironment,
    contract: OptionContract,
    n_simulations: int = 50000,
    n_steps: int = 100,
    n_basis: int = 3,
) -> dict:
    """
    Price an American option using Longstaff-Schwartz LSM.

    Parameters
    ----------
    market : MarketEnvironment
    contract : OptionContract
    n_simulations : int
        Number of MC paths
    n_steps : int
        Number of exercise opportunities
    n_basis : int
        Number of polynomial basis functions for regression

    Returns
    -------
    dict with price, std_error, early_exercise_pct, european_price
    """
    paths = simulate_paths(market, n_simulations, n_steps, antithetic=True)
    dt = market.maturity / n_steps
    discount = np.exp(-market.rate * dt)

    # Compute payoff at each time step
    if contract.option_type == "call":
        payoff_matrix = np.maximum(paths - contract.strike, 0)
    else:
        payoff_matrix = np.maximum(contract.strike - paths, 0)

    # Cash flows: initialized to payoff at maturity
    cash_flows = payoff_matrix[:, -1].copy()
    exercise_time = np.full(n_simulations, n_steps)

    # Backward induction
    for t in range(n_steps - 1, 0, -1):
        # Discount future cash flows by one step
        cash_flows *= discount

        # Find paths that are in-the-money at time t
        itm = payoff_matrix[:, t] > 0
        if np.sum(itm) < n_basis + 1:
            continue

        # Regression: continuation value = f(S_t)
        S_itm = paths[itm, t]
        CF_itm = cash_flows[itm]

        # Polynomial basis functions (Laguerre-like)
        X = np.column_stack([
            np.ones(len(S_itm)),
            S_itm / market.spot,
            (S_itm / market.spot) ** 2,
        ][:n_basis + 1])

        # OLS regression
        try:
            coeffs = np.linalg.lstsq(X, CF_itm, rcond=None)[0]
            continuation = X @ coeffs
        except np.linalg.LinAlgError:
            continue

        # Exercise decision: exercise if immediate payoff > continuation
        exercise_now = payoff_matrix[itm, t] > continuation

        # Update cash flows and exercise times for early exercisers
        itm_indices = np.where(itm)[0]
        early = itm_indices[exercise_now]
        cash_flows[early] = payoff_matrix[early, t]
        exercise_time[early] = t

    # Discount all cash flows to time 0
    for i in range(n_simulations):
        cash_flows[i] *= np.exp(-market.rate * exercise_time[i] * dt)

    # Recalculate: discount from exercise time to 0
    discount_factors = np.exp(-market.rate * exercise_time * dt)
    pv = payoff_matrix[np.arange(n_simulations), exercise_time] * discount_factors

    price = np.mean(pv)
    se = np.std(pv) / np.sqrt(n_simulations)
    early_pct = np.mean(exercise_time < n_steps)

    # European price for comparison
    euro_pv = payoff_matrix[:, -1] * np.exp(-market.rate * market.maturity)
    euro_price = np.mean(euro_pv)

    return {
        "price": price,
        "std_error": se,
        "confidence_interval": (price - 1.96 * se, price + 1.96 * se),
        "early_exercise_pct": early_pct,
        "european_price": euro_price,
        "early_exercise_premium": price - euro_price,
        "option_type": contract.option_type,
        "n_simulations": n_simulations,
        "n_steps": n_steps,
        "method": "longstaff-schwartz",
    }
