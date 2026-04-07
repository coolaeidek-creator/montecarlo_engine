"""
Merton Jump-Diffusion Model.

Extends GBM by adding random jumps (Poisson-distributed) to capture
fat tails and sudden price moves that pure GBM cannot model.

S(t+dt) = S(t) * exp((r - λk - 0.5σ²)dt + σ√dt·Z + J)

where J ~ N(μ_J, σ_J) with Poisson arrival rate λ.
"""

import numpy as np
from .models import MarketEnvironment, OptionContract


def simulate_jump_diffusion(
    market: MarketEnvironment,
    contract: OptionContract,
    n_simulations: int = 50000,
    jump_intensity: float = 1.0,
    jump_mean: float = -0.05,
    jump_vol: float = 0.10,
) -> dict:
    """
    Price European option under Merton Jump-Diffusion.

    Parameters
    ----------
    jump_intensity : float (λ)
        Average number of jumps per year
    jump_mean : float (μ_J)
        Mean log-jump size (negative = crash-like)
    jump_vol : float (σ_J)
        Jump size volatility
    """
    S = market.spot
    K = contract.strike
    r = market.rate
    sigma = market.volatility
    T = market.maturity

    # Compensator: k = E[e^J] - 1
    k = np.exp(jump_mean + 0.5 * jump_vol ** 2) - 1
    drift = (r - jump_intensity * k - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T)

    # Standard GBM component
    z = np.random.standard_normal(n_simulations)

    # Jump component: number of jumps is Poisson(λT)
    n_jumps = np.random.poisson(jump_intensity * T, n_simulations)
    jump_sizes = np.zeros(n_simulations)
    for i in range(n_simulations):
        if n_jumps[i] > 0:
            jumps = np.random.normal(jump_mean, jump_vol, n_jumps[i])
            jump_sizes[i] = np.sum(jumps)

    # Terminal prices
    log_returns = drift + diffusion * z + jump_sizes
    terminal = S * np.exp(log_returns)

    # Payoff
    discount = np.exp(-r * T)
    if contract.option_type == "call":
        payoffs = np.maximum(terminal - K, 0) * discount
    else:
        payoffs = np.maximum(K - terminal, 0) * discount

    price = np.mean(payoffs)
    se = np.std(payoffs) / np.sqrt(n_simulations)

    # Compare with pure GBM (no jumps)
    gbm_terminal = S * np.exp(
        (r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z
    )
    if contract.option_type == "call":
        gbm_payoffs = np.maximum(gbm_terminal - K, 0) * discount
    else:
        gbm_payoffs = np.maximum(K - gbm_terminal, 0) * discount
    gbm_price = np.mean(gbm_payoffs)

    return {
        "price": price,
        "std_error": se,
        "confidence_interval": (price - 1.96 * se, price + 1.96 * se),
        "gbm_price": gbm_price,
        "jump_premium": price - gbm_price,
        "avg_jumps": np.mean(n_jumps),
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_vol": jump_vol,
        "option_type": contract.option_type,
        "n_simulations": n_simulations,
        "model": "merton-jump-diffusion",
    }
