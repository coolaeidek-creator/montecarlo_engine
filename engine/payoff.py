import numpy as np
from .models import OptionContract


def calculate_payoff(
    terminal_prices: np.ndarray,
    contract: OptionContract,
) -> np.ndarray:
    """
    Calculate option payoff at maturity.

    For a call: max(S_T - K, 0)
    For a put: max(K - S_T, 0)

    Parameters
    ----------
    terminal_prices : np.ndarray
        Terminal stock prices from simulation
    contract : OptionContract
        Option contract specification

    Returns
    -------
    np.ndarray
        Payoff values for each simulation path
    """
    if contract.option_type == "call":
        return np.maximum(terminal_prices - contract.strike, 0)
    elif contract.option_type == "put":
        return np.maximum(contract.strike - terminal_prices, 0)
    else:
        raise ValueError(f"Unknown option type: {contract.option_type}")
