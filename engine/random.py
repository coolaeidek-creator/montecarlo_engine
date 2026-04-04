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
