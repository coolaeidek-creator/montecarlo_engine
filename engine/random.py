import numpy as np


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
