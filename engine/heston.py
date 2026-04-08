"""
Heston Stochastic Volatility Model.

Unlike Black-Scholes (constant vol), Heston models volatility as a
mean-reverting stochastic process (CIR), producing realistic skew
and fat tails without jump components.

dS = r·S·dt + √v·S·dW_1
dv = κ(θ - v)dt + σ_v·√v·dW_2
Corr(dW_1, dW_2) = ρ

Parameters:
    κ (kappa)  — mean reversion speed
    θ (theta)  — long-run variance
    σ_v (vol of vol) — volatility of variance
    ρ (rho)    — correlation between spot and vol processes
    v0         — initial variance
"""

import numpy as np
from .models import MarketEnvironment, OptionContract


def price_heston(
    market: MarketEnvironment,
    contract: OptionContract,
    n_simulations: int = 50000,
    n_steps: int = 200,
    kappa: float = 2.0,
    theta: float = 0.04,
    vol_of_vol: float = 0.3,
    rho: float = -0.7,
    v0: float = None,
) -> dict:
    """
    Price European option under the Heston stochastic volatility model.

    Uses Euler-Maruyama discretization with full truncation scheme
    to prevent negative variance.

    Parameters
    ----------
    kappa : float
        Mean reversion speed of variance
    theta : float
        Long-run variance level (θ = σ² where σ is long-run vol)
    vol_of_vol : float
        Volatility of variance (σ_v)
    rho : float
        Correlation between spot and variance Brownian motions
    v0 : float or None
        Initial variance (defaults to market.volatility²)
    """
    S0 = market.spot
    K = contract.strike
    r = market.rate
    T = market.maturity
    dt = T / n_steps

    if v0 is None:
        v0 = market.volatility ** 2

    # Correlated Brownian motions via Cholesky
    # W1 = Z1, W2 = ρ·Z1 + √(1-ρ²)·Z2
    z1 = np.random.standard_normal((n_simulations, n_steps))
    z2 = np.random.standard_normal((n_simulations, n_steps))
    w1 = z1
    w2 = rho * z1 + np.sqrt(1 - rho ** 2) * z2

    # Simulate paths
    S = np.full(n_simulations, float(S0))
    v = np.full(n_simulations, float(v0))

    # Track variance path for diagnostics
    v_path_mean = [v0]

    for t in range(n_steps):
        # Full truncation: use max(v, 0) in diffusion terms
        v_pos = np.maximum(v, 0)
        sqrt_v = np.sqrt(v_pos)

        # Stock price update (log scheme for stability)
        S = S * np.exp(
            (r - 0.5 * v_pos) * dt + sqrt_v * np.sqrt(dt) * w1[:, t]
        )

        # Variance update (Euler with full truncation)
        v = v + kappa * (theta - v_pos) * dt + vol_of_vol * sqrt_v * np.sqrt(dt) * w2[:, t]

        v_path_mean.append(np.mean(np.maximum(v, 0)))

    # Payoff
    discount = np.exp(-r * T)
    if contract.option_type == "call":
        payoffs = np.maximum(S - K, 0) * discount
    else:
        payoffs = np.maximum(K - S, 0) * discount

    price = np.mean(payoffs)
    se = np.std(payoffs) / np.sqrt(n_simulations)

    # GBM comparison (constant vol = sqrt(v0))
    sigma_flat = np.sqrt(v0)
    z_gbm = np.random.standard_normal(n_simulations)
    S_gbm = S0 * np.exp((r - 0.5 * sigma_flat ** 2) * T + sigma_flat * np.sqrt(T) * z_gbm)
    if contract.option_type == "call":
        gbm_payoffs = np.maximum(S_gbm - K, 0) * discount
    else:
        gbm_payoffs = np.maximum(K - S_gbm, 0) * discount
    gbm_price = np.mean(gbm_payoffs)

    # Feller condition check: 2κθ > σ_v² ensures variance stays positive
    feller = 2 * kappa * theta / (vol_of_vol ** 2)
    feller_satisfied = feller > 1.0

    return {
        "price": price,
        "std_error": se,
        "confidence_interval": (price - 1.96 * se, price + 1.96 * se),
        "gbm_price": gbm_price,
        "stoch_vol_premium": price - gbm_price,
        "params": {
            "kappa": kappa,
            "theta": theta,
            "vol_of_vol": vol_of_vol,
            "rho": rho,
            "v0": v0,
        },
        "feller_ratio": feller,
        "feller_satisfied": feller_satisfied,
        "final_vol_mean": np.sqrt(np.mean(np.maximum(v, 0))),
        "option_type": contract.option_type,
        "n_simulations": n_simulations,
        "n_steps": n_steps,
        "model": "heston-stochastic-vol",
    }


def heston_smile(
    market: MarketEnvironment,
    n_simulations: int = 30000,
    n_steps: int = 150,
    kappa: float = 2.0,
    theta: float = 0.04,
    vol_of_vol: float = 0.3,
    rho: float = -0.7,
    v0: float = None,
    n_strikes: int = 11,
) -> dict:
    """
    Generate implied volatility smile from Heston model prices.

    Prices calls at multiple strikes and backs out implied vol
    using bisection, showing the skew produced by stochastic vol.
    """
    S0 = market.spot
    moneyness = np.linspace(0.8, 1.2, n_strikes)
    strikes = S0 * moneyness

    prices = []
    ivs = []

    for K in strikes:
        contract = OptionContract(strike=float(K), option_type="call")
        result = price_heston(
            market, contract, n_simulations, n_steps,
            kappa, theta, vol_of_vol, rho, v0,
        )
        mc_price = result["price"]
        prices.append(mc_price)

        # Back out IV via bisection
        iv = _bisection_iv(S0, float(K), market.rate, market.maturity, mc_price, "call")
        ivs.append(iv)

    return {
        "strikes": strikes.tolist(),
        "moneyness": moneyness.tolist(),
        "prices": prices,
        "implied_vols": ivs,
        "params": {
            "kappa": kappa, "theta": theta,
            "vol_of_vol": vol_of_vol, "rho": rho,
        },
        "model": "heston-smile",
    }


def _bisection_iv(S, K, r, T, market_price, option_type, tol=1e-6, max_iter=100):
    """Simple bisection to find implied vol from a given option price."""
    from scipy.stats import norm

    def bs_call(sigma):
        if sigma <= 0 or T <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def bs_put(sigma):
        if sigma <= 0 or T <= 0:
            return 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    price_func = bs_call if option_type == "call" else bs_put

    lo, hi = 0.001, 3.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p = price_func(mid)
        if abs(p - market_price) < tol:
            return mid
        if p < market_price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2
