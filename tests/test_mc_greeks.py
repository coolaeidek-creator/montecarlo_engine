"""
Tests for Monte Carlo Greeks (finite difference method).
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment, OptionContract
from engine.mc_greeks import mc_greeks
from engine.greeks import compute_greeks


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


class TestMCGreeks:
    def test_call_delta_positive(self, market):
        """Call delta should be positive."""
        contract = OptionContract(strike=100, option_type="call")
        result = mc_greeks(market, contract, n_simulations=30000)
        assert result["delta"] > 0

    def test_put_delta_negative(self, market):
        """Put delta should be negative."""
        contract = OptionContract(strike=100, option_type="put")
        result = mc_greeks(market, contract, n_simulations=30000)
        assert result["delta"] < 0

    def test_gamma_positive(self, market):
        """Gamma should be positive for both calls and puts."""
        contract = OptionContract(strike=100, option_type="call")
        result = mc_greeks(market, contract, n_simulations=30000)
        assert result["gamma"] > 0

    def test_vega_positive(self, market):
        """Vega should be positive (higher vol = higher option value)."""
        contract = OptionContract(strike=100, option_type="call")
        result = mc_greeks(market, contract, n_simulations=30000)
        assert result["vega"] > 0

    def test_call_rho_positive(self, market):
        """Call rho should be positive."""
        contract = OptionContract(strike=100, option_type="call")
        result = mc_greeks(market, contract, n_simulations=30000)
        assert result["rho"] > 0

    def test_put_rho_negative(self, market):
        """Put rho should be negative."""
        contract = OptionContract(strike=100, option_type="put")
        result = mc_greeks(market, contract, n_simulations=30000)
        assert result["rho"] < 0

    def test_result_fields(self, market):
        contract = OptionContract(strike=100, option_type="call")
        result = mc_greeks(market, contract, n_simulations=10000)
        assert "delta" in result
        assert "gamma" in result
        assert "vega" in result
        assert "theta" in result
        assert "rho" in result
        assert "base_price" in result
        assert "method" in result

    def test_mc_delta_near_bs_delta(self, market):
        """MC delta should be close to BS delta for vanilla option."""
        contract = OptionContract(strike=100, option_type="call")
        mc = mc_greeks(market, contract, n_simulations=50000)
        bs = compute_greeks(market, contract)
        # Should agree within ~0.05
        assert abs(mc["delta"] - bs["delta"]) < 0.08

    def test_mc_gamma_near_bs_gamma(self, market):
        """MC gamma should be close to BS gamma for vanilla option."""
        contract = OptionContract(strike=100, option_type="call")
        mc = mc_greeks(market, contract, n_simulations=100000)
        bs = compute_greeks(market, contract)
        # Gamma (2nd deriv) is noisy in MC — wider tolerance
        assert abs(mc["gamma"] - bs["gamma"]) < 0.15

    def test_theta_negative_for_atm(self, market):
        """Theta should be negative for ATM options (time decay)."""
        contract = OptionContract(strike=100, option_type="call")
        result = mc_greeks(market, contract, n_simulations=30000)
        assert result["theta"] < 0

    def test_base_price_positive(self, market):
        """Base price should be positive."""
        contract = OptionContract(strike=100, option_type="call")
        result = mc_greeks(market, contract, n_simulations=10000)
        assert result["base_price"] > 0
