"""
Tests for Black-Scholes analytical pricing.

Validates against known closed-form results and put-call parity.
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment, OptionContract
from engine.analytical import bs_price, bs_price_both


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


class TestBSPrice:
    def test_call_positive(self, market):
        contract = OptionContract(strike=100, option_type="call")
        price = bs_price(market, contract)
        assert price > 0

    def test_put_positive(self, market):
        contract = OptionContract(strike=100, option_type="put")
        price = bs_price(market, contract)
        assert price > 0

    def test_atm_call_greater_than_put(self, market):
        """ATM call > put when rates are positive (forward > spot)."""
        call = bs_price(market, OptionContract(strike=100, option_type="call"))
        put = bs_price(market, OptionContract(strike=100, option_type="put"))
        assert call > put

    def test_deep_itm_call_near_intrinsic(self, market):
        """Deep ITM call ≈ S - K*e^(-rT)."""
        contract = OptionContract(strike=50, option_type="call")
        price = bs_price(market, contract)
        intrinsic = 100 - 50 * np.exp(-0.05)
        assert abs(price - intrinsic) < 1.0

    def test_deep_otm_call_near_zero(self, market):
        contract = OptionContract(strike=200, option_type="call")
        price = bs_price(market, contract)
        assert price < 1.0

    def test_put_call_parity(self, market):
        """C - P = S - K*e^(-rT)."""
        K = 100
        both = bs_price_both(market, K)
        parity_lhs = both["call"] - both["put"]
        parity_rhs = market.spot - K * np.exp(-market.rate * market.maturity)
        assert abs(parity_lhs - parity_rhs) < 1e-10

    def test_known_value(self):
        """Test against a known BS price (S=100, K=100, r=5%, σ=20%, T=1)."""
        market = MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)
        contract = OptionContract(strike=100, option_type="call")
        price = bs_price(market, contract)
        # Known value ≈ 10.4506
        assert abs(price - 10.4506) < 0.01

    def test_zero_vol_call(self):
        """With near-zero vol, call = max(S - K*e^(-rT), 0)."""
        market = MarketEnvironment(spot=100, rate=0.05, volatility=0.001, maturity=1.0)
        contract = OptionContract(strike=95, option_type="call")
        price = bs_price(market, contract)
        expected = 100 - 95 * np.exp(-0.05)
        assert abs(price - expected) < 0.1
