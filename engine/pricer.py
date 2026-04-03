import numpy as np
from .models import MarketEnvironment, OptionContract
from .random import generate_standard_normal
from .simulator import simulate_terminal_prices
from .payoff import calculate_payoff


class OptionPricer:
    """
    Price European options using Monte Carlo simulation.
    """

    def __init__(self, n_simulations: int = 10000):
        """
        Initialize the pricer.

        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo paths to simulate
        """
        self.n_simulations = n_simulations

    def price(
        self,
        market: MarketEnvironment,
        contract: OptionContract,
    ) -> dict:
        """
        Price an option using Monte Carlo simulation.

        Algorithm:
        1. Generate random shocks N(0,1)
        2. Simulate terminal stock prices
        3. Calculate payoffs at maturity
        4. Discount expected payoff to present value

        Parameters
        ----------
        market : MarketEnvironment
            Market parameters (spot, rate, volatility, maturity)
        contract : OptionContract
            Option contract (strike, type)

        Returns
        -------
        dict
            Dictionary with price, std_error, and confidence interval
        """
        # Generate random shocks
        shocks = generate_standard_normal(self.n_simulations)

        # Simulate terminal prices
        terminal_prices = simulate_terminal_prices(market, shocks)

        # Calculate payoffs
        payoffs = calculate_payoff(terminal_prices, contract)

        # Discount to present value
        discount_factor = np.exp(-market.rate * market.maturity)
        pv_payoffs = payoffs * discount_factor

        # Calculate statistics
        price = np.mean(pv_payoffs)
        std_error = np.std(pv_payoffs) / np.sqrt(self.n_simulations)

        # 95% confidence interval
        z_score = 1.96
        ci_lower = price - z_score * std_error
        ci_upper = price + z_score * std_error

        return {
            "price": price,
            "std_error": std_error,
            "confidence_interval": (ci_lower, ci_upper),
            "n_simulations": self.n_simulations,
        }


def price_option(
    market: MarketEnvironment,
    contract: OptionContract,
    n_simulations: int = 10000,
) -> dict:
    """
    Convenience function to price an option.

    Parameters
    ----------
    market : MarketEnvironment
        Market parameters
    contract : OptionContract
        Option contract
    n_simulations : int
        Number of Monte Carlo simulations

    Returns
    -------
    dict
        Pricing result with price, std_error, and confidence interval
    """
    pricer = OptionPricer(n_simulations=n_simulations)
    return pricer.price(market, contract)
