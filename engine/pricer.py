import numpy as np
from .models import MarketEnvironment, OptionContract
from .random import generate_standard_normal, generate_antithetic
from .simulator import simulate_terminal_prices
from .payoff import calculate_payoff
from .analytical import bs_price


class OptionPricer:
    """
    Price European options using Monte Carlo simulation.

    Supports:
    - Standard Monte Carlo
    - Antithetic variates (variance reduction)
    - Control variates (variance reduction using delta hedge)
    - Black-Scholes analytical validation
    """

    def __init__(self, n_simulations: int = 10000, method: str = "standard"):
        """
        Parameters
        ----------
        n_simulations : int
            Number of Monte Carlo paths to simulate
        method : str
            'standard', 'antithetic', or 'control_variate'
        """
        self.n_simulations = n_simulations
        self.method = method

    def price(
        self,
        market: MarketEnvironment,
        contract: OptionContract,
    ) -> dict:
        """
        Price an option using Monte Carlo simulation.

        Returns
        -------
        dict with: price, std_error, confidence_interval, bs_price,
                   bs_diff, n_simulations, method
        """
        discount_factor = np.exp(-market.rate * market.maturity)

        if self.method == "antithetic":
            pv_payoffs = self._price_antithetic(market, contract, discount_factor)
        elif self.method == "control_variate":
            pv_payoffs = self._price_control_variate(market, contract, discount_factor)
        else:
            pv_payoffs = self._price_standard(market, contract, discount_factor)

        # Statistics
        price_mc = np.mean(pv_payoffs)
        std_error = np.std(pv_payoffs) / np.sqrt(len(pv_payoffs))
        ci_lower = price_mc - 1.96 * std_error
        ci_upper = price_mc + 1.96 * std_error

        # BS analytical benchmark
        price_bs = bs_price(market, contract)

        return {
            "price": price_mc,
            "std_error": std_error,
            "confidence_interval": (ci_lower, ci_upper),
            "bs_price": price_bs,
            "bs_diff": abs(price_mc - price_bs),
            "n_simulations": self.n_simulations,
            "method": self.method,
        }

    def _price_standard(self, market, contract, discount_factor):
        shocks = generate_standard_normal(self.n_simulations)
        terminal_prices = simulate_terminal_prices(market, shocks)
        payoffs = calculate_payoff(terminal_prices, contract)
        return payoffs * discount_factor

    def _price_antithetic(self, market, contract, discount_factor):
        z_pos, z_neg = generate_antithetic(self.n_simulations)

        term_pos = simulate_terminal_prices(market, z_pos)
        term_neg = simulate_terminal_prices(market, z_neg)

        pay_pos = calculate_payoff(term_pos, contract)
        pay_neg = calculate_payoff(term_neg, contract)

        # Average each antithetic pair
        avg_payoffs = (pay_pos + pay_neg) / 2.0
        return avg_payoffs * discount_factor

    def _price_control_variate(self, market, contract, discount_factor):
        """
        Control variate method: use the stock price as a control variate.

        The idea: S_T is correlated with the payoff. We know E[S_T] = S * e^(rT).
        Adjust the payoff estimate by subtracting the error in the control.

        Y_cv = payoff - c * (S_T - E[S_T])

        where c = -Cov(payoff, S_T) / Var(S_T) is the optimal coefficient.
        """
        shocks = generate_standard_normal(self.n_simulations)
        terminal_prices = simulate_terminal_prices(market, shocks)
        payoffs = calculate_payoff(terminal_prices, contract)
        discounted_payoffs = payoffs * discount_factor

        # Control variate: S_T vs its known expectation
        expected_ST = market.spot * np.exp(market.rate * market.maturity)
        control = terminal_prices - expected_ST

        # Optimal coefficient via regression
        cov = np.cov(discounted_payoffs, control)
        if cov[1, 1] > 0:
            c_star = -cov[0, 1] / cov[1, 1]
        else:
            c_star = 0.0

        # Adjusted payoffs
        adjusted = discounted_payoffs + c_star * control
        return adjusted


def price_option(
    market: MarketEnvironment,
    contract: OptionContract,
    n_simulations: int = 10000,
    method: str = "standard",
) -> dict:
    """Convenience function to price an option."""
    pricer = OptionPricer(n_simulations=n_simulations, method=method)
    return pricer.price(market, contract)
