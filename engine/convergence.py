"""
Monte Carlo Convergence Test.

Validates the 1/√N convergence rate of MC estimators against the
Black-Scholes analytical benchmark across multiple variance-reduction
methods (standard, antithetic, control variate, stratified, sobol).

Returns per-method price/SE/error trajectories so the variance-reduction
gain is directly observable.
"""

from typing import List, Optional

import numpy as np

from .analytical import bs_price
from .models import MarketEnvironment, OptionContract
from .pricer import OptionPricer


DEFAULT_SAMPLE_SIZES: List[int] = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
DEFAULT_METHODS: List[str] = ["standard", "antithetic", "control_variate", "stratified", "sobol"]


def mc_convergence_test(
    market: MarketEnvironment,
    contract: OptionContract,
    sample_sizes: Optional[List[int]] = None,
    methods: Optional[List[str]] = None,
) -> dict:
    """
    Run MC convergence test across sample sizes and methods.

    For each (method, N) pair: price the option, record price, SE,
    absolute error vs BS, and 95% CI width. Also reports a fitted
    convergence-rate exponent for the SE decay (theory: SE ∝ N^-0.5).

    Returns a dict with per-method trajectories and summary stats.
    """
    if sample_sizes is None:
        sample_sizes = list(DEFAULT_SAMPLE_SIZES)
    if methods is None:
        methods = list(DEFAULT_METHODS)

    if any(n <= 1 for n in sample_sizes):
        raise ValueError("sample sizes must be > 1")
    if not methods:
        raise ValueError("at least one method required")

    bs_ref = bs_price(market, contract)

    method_results = {}
    for method in methods:
        trajectory = []
        for n in sample_sizes:
            pricer = OptionPricer(n_simulations=n, method=method)
            r = pricer.price(market, contract)
            ci_lo, ci_hi = r["confidence_interval"]
            trajectory.append({
                "n": n,
                "price": r["price"],
                "std_error": r["std_error"],
                "abs_error": abs(r["price"] - bs_ref),
                "rel_error_pct": (abs(r["price"] - bs_ref) / bs_ref * 100.0) if bs_ref > 0 else 0.0,
                "ci_width": ci_hi - ci_lo,
            })

        rate = _fit_convergence_rate(
            [t["n"] for t in trajectory],
            [t["std_error"] for t in trajectory],
        )

        final = trajectory[-1]
        method_results[method] = {
            "trajectory": trajectory,
            "convergence_rate": rate,
            "final_price": final["price"],
            "final_std_error": final["std_error"],
            "final_abs_error": final["abs_error"],
            "final_rel_error_pct": final["rel_error_pct"],
        }

    # Variance-reduction factor: how much smaller is each method's final SE
    # compared to standard MC at the same N.
    baseline_se = method_results.get("standard", {}).get("final_std_error")
    if baseline_se and baseline_se > 0:
        for method, mr in method_results.items():
            mr["variance_reduction_factor"] = baseline_se / mr["final_std_error"] if mr["final_std_error"] > 0 else None
    else:
        for mr in method_results.values():
            mr["variance_reduction_factor"] = None

    return {
        "bs_reference": bs_ref,
        "sample_sizes": list(sample_sizes),
        "methods": method_results,
        "option_type": contract.option_type,
        "model": "mc-convergence",
    }


def _fit_convergence_rate(ns: List[int], ses: List[float]) -> Optional[float]:
    """
    Fit log(SE) = a + b * log(N) and return b. Theory predicts b = -0.5
    for standard MC. Returns None if fit not possible.
    """
    valid = [(n, s) for n, s in zip(ns, ses) if s > 0 and n > 0]
    if len(valid) < 2:
        return None
    log_n = np.log([v[0] for v in valid])
    log_se = np.log([v[1] for v in valid])
    slope, _ = np.polyfit(log_n, log_se, 1)
    return float(slope)
