import numpy as np
from typing import Tuple


def generate_standard_normal(n_simulations: int) -> np.ndarray:
    """
    Generate standard normal random numbers N(0,1)

    Parameters
    ----------
    n_simulations : int
        Number of Monte Carlo simulations

    Returns
    -------
    np.ndarray
        Array of random shocks
    """
    return np.random.standard_normal(n_simulations)


def generate_antithetic(n_simulations: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate antithetic pairs of standard normal shocks.

    For each random Z, also uses -Z. This halves variance
    with no extra simulation cost (variance reduction technique).

    Returns
    -------
    Tuple of (Z, -Z) arrays, each of size n_simulations // 2
    """
    half = n_simulations // 2
    z = np.random.standard_normal(half)
    return z, -z


def generate_stratified(n_simulations: int) -> np.ndarray:
    """
    Generate stratified (quasi-random) normal samples.

    Divides [0,1] into n equal strata, samples uniformly within
    each stratum, then applies inverse normal CDF.

    This ensures better coverage of the distribution tails
    compared to pure random sampling.
    """
    from scipy.stats import norm
    u = (np.arange(n_simulations) + np.random.uniform(size=n_simulations)) / n_simulations
    return norm.ppf(u)


def generate_sobol(n_simulations: int) -> np.ndarray:
    """
    Generate quasi-random Sobol sequence normal samples.

    Sobol sequences are low-discrepancy sequences that provide
    more uniform coverage of the sample space than pseudo-random numbers.
    Falls back to stratified sampling if scipy.stats.qmc not available.
    """
    try:
        from scipy.stats.qmc import Sobol
        from scipy.stats import norm
        # Sobol requires power of 2
        m = int(np.ceil(np.log2(n_simulations)))
        sampler = Sobol(d=1, scramble=True)
        u = sampler.random(2 ** m)[:n_simulations, 0]
        # Clip to avoid inf at 0 and 1
        u = np.clip(u, 1e-10, 1 - 1e-10)
        return norm.ppf(u)
    except ImportError:
        return generate_stratified(n_simulations)
