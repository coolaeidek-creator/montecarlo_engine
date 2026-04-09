"""
Delta Hedging Simulation.

Simulates a discrete delta hedging strategy to show how
rebalancing frequency affects hedge P&L. Demonstrates the
core concept behind options market-making: selling an option
and dynamically replicating the payoff via delta hedging.

Key insight: hedge error ~ σ²·Γ·(ΔS)² (Gamma P&L)
"""

import numpy as np
from scipy.stats import norm
from .models import MarketEnvironment, OptionContract


def _bs_delta(S, K, r, sigma, T, option_type):
    """BS delta for hedging."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def _bs_price(S, K, r, sigma, T, option_type):
    """BS price for P&L."""
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K, 0)
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def simulate_delta_hedge(
    market: MarketEnvironment,
    contract: OptionContract,
    n_simulations: int = 1000,
    rebalance_freq: int = 1,  # rebalance every N days
    n_days: int = 252,
) -> dict:
    """
    Simulate delta hedging of a short option position.

    The trader sells the option at BS price and delta hedges.
    Hedge P&L should be ~0 with frequent rebalancing.

    Parameters
    ----------
    rebalance_freq : int
        Rebalance every N trading days (1=daily, 5=weekly, 21=monthly)
    """
    S0 = market.spot
    K = contract.strike
    r = market.rate
    sigma = market.volatility
    T = market.maturity
    dt = T / n_days

    # Option premium received (BS price at inception)
    premium = _bs_price(S0, K, r, sigma, T, contract.option_type)

    hedge_pnls = []

    for _ in range(n_simulations):
        S = S0
        cash = premium  # received premium
        shares = 0.0
        rebalance_count = 0

        for day in range(n_days):
            time_left = T - day * dt
            if time_left <= 0:
                break

            # Rebalance delta
            if day % rebalance_freq == 0:
                new_delta = _bs_delta(S, K, r, sigma, time_left, contract.option_type)
                # Buy/sell shares to match delta (we are short the option)
                trade = new_delta - shares
                cash -= trade * S  # pay for shares
                shares = new_delta
                rebalance_count += 1

            # Stock price moves
            z = np.random.randn()
            S = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

            # Interest on cash
            cash *= np.exp(r * dt)

        # At expiry: unwind hedge and pay option payoff
        # Liquidate shares
        cash += shares * S
        # Pay option payoff (we are short)
        if contract.option_type == "call":
            payoff = max(S - K, 0)
        else:
            payoff = max(K - S, 0)
        cash -= payoff

        # Hedge P&L = final cash (should be ~0 for perfect hedge)
        hedge_pnls.append(cash)

    pnls = np.array(hedge_pnls)

    return {
        "mean_pnl": float(np.mean(pnls)),
        "std_pnl": float(np.std(pnls)),
        "median_pnl": float(np.median(pnls)),
        "max_pnl": float(np.max(pnls)),
        "min_pnl": float(np.min(pnls)),
        "pnl_95_ci": (
            float(np.percentile(pnls, 2.5)),
            float(np.percentile(pnls, 97.5)),
        ),
        "sharpe": float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0,
        "premium_received": premium,
        "hedge_efficiency": 1 - np.std(pnls) / premium if premium > 0 else 0,
        "rebalance_freq": rebalance_freq,
        "n_simulations": n_simulations,
        "option_type": contract.option_type,
        "model": "delta-hedge-sim",
    }


def compare_hedge_frequencies(
    market: MarketEnvironment,
    contract: OptionContract,
    n_simulations: int = 500,
) -> dict:
    """
    Compare hedge P&L across different rebalancing frequencies.
    """
    freqs = [1, 2, 5, 10, 21]  # daily, 2-day, weekly, bi-weekly, monthly
    labels = ["Daily", "2-Day", "Weekly", "Bi-Weekly", "Monthly"]

    results = []
    for freq, label in zip(freqs, labels):
        r = simulate_delta_hedge(market, contract, n_simulations, freq)
        results.append({
            "frequency": label,
            "freq_days": freq,
            "mean_pnl": r["mean_pnl"],
            "std_pnl": r["std_pnl"],
            "hedge_efficiency": r["hedge_efficiency"],
        })

    return {
        "results": results,
        "premium": _bs_price(
            market.spot, contract.strike, market.rate,
            market.volatility, market.maturity, contract.option_type,
        ),
        "model": "hedge-frequency-comparison",
    }
