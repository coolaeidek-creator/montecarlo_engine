"""
Tests for Greeks computation.
"""

import pytest
from engine.models import MarketEnvironment, OptionContract
from engine.greeks import compute_greeks


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


class TestGreeks:
    def test_call_delta_between_0_1(self, market):
        contract = OptionContract(strike=100, option_type="call")
        g = compute_greeks(market, contract)
        assert 0 < g["delta"] < 1

    def test_put_delta_between_neg1_0(self, market):
        contract = OptionContract(strike=100, option_type="put")
        g = compute_greeks(market, contract)
        assert -1 < g["delta"] < 0

    def test_gamma_positive(self, market):
        contract = OptionContract(strike=100, option_type="call")
        g = compute_greeks(market, contract)
        assert g["gamma"] > 0

    def test_call_theta_negative(self, market):
        """Options lose value over time (theta decay)."""
        contract = OptionContract(strike=100, option_type="call")
        g = compute_greeks(market, contract)
        assert g["theta"] < 0

    def test_vega_positive(self, market):
        """Higher vol = higher option value."""
        contract = OptionContract(strike=100, option_type="call")
        g = compute_greeks(market, contract)
        assert g["vega"] > 0

    def test_call_rho_positive(self, market):
        """Higher rates benefit call holders."""
        contract = OptionContract(strike=100, option_type="call")
        g = compute_greeks(market, contract)
        assert g["rho"] > 0

    def test_put_rho_negative(self, market):
        contract = OptionContract(strike=100, option_type="put")
        g = compute_greeks(market, contract)
        assert g["rho"] < 0

    def test_deep_itm_call_delta_near_1(self, market):
        contract = OptionContract(strike=50, option_type="call")
        g = compute_greeks(market, contract)
        assert g["delta"] > 0.95

    def test_deep_otm_call_delta_near_0(self, market):
        contract = OptionContract(strike=200, option_type="call")
        g = compute_greeks(market, contract)
        assert g["delta"] < 0.05

    def test_put_call_delta_relation(self, market):
        """call_delta - put_delta = 1."""
        call_g = compute_greeks(market, OptionContract(strike=100, option_type="call"))
        put_g = compute_greeks(market, OptionContract(strike=100, option_type="put"))
        assert abs(call_g["delta"] - put_g["delta"] - 1.0) < 1e-10
