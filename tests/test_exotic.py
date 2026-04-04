"""
Tests for exotic options pricing.

Validates Asian, Barrier, Lookback, and Digital options.
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment, OptionContract
from engine.exotic import price_asian, price_barrier, price_lookback, price_digital
from engine.analytical import bs_price


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.25, maturity=1.0)


@pytest.fixture
def atm_call():
    return OptionContract(strike=100, option_type="call")


@pytest.fixture
def atm_put():
    return OptionContract(strike=100, option_type="put")


class TestAsianOptions:
    def test_asian_call_cheaper_than_vanilla(self, market, atm_call):
        """Asian call should be cheaper than vanilla (averaging reduces variance)."""
        asian = price_asian(market, atm_call, n_simulations=50000)
        vanilla_bs = bs_price(market, atm_call)
        assert asian["price"] < vanilla_bs * 1.05  # allow small MC noise

    def test_asian_put_cheaper_than_vanilla(self, market, atm_put):
        asian = price_asian(market, atm_put, n_simulations=50000)
        vanilla_bs = bs_price(market, atm_put)
        assert asian["price"] < vanilla_bs * 1.05

    def test_geometric_cheaper_than_arithmetic(self, market, atm_call):
        """Geometric average ≤ arithmetic average (AM-GM inequality)."""
        arith = price_asian(market, atm_call, averaging="arithmetic", n_simulations=50000)
        geom = price_asian(market, atm_call, averaging="geometric", n_simulations=50000)
        # Geometric should be ≤ arithmetic (with some MC noise tolerance)
        assert geom["price"] < arith["price"] * 1.10

    def test_asian_fields(self, market, atm_call):
        result = price_asian(market, atm_call, n_simulations=10000)
        assert result["exotic_type"] == "asian"
        assert "std_error" in result
        assert "confidence_interval" in result


class TestBarrierOptions:
    def test_down_and_out_cheaper_than_vanilla(self, market, atm_call):
        """Down-and-out call (can be knocked out) ≤ vanilla call."""
        barrier_result = price_barrier(
            market, atm_call, barrier=80, barrier_type="down-and-out",
            n_simulations=50000,
        )
        vanilla_bs = bs_price(market, atm_call)
        assert barrier_result["price"] <= vanilla_bs * 1.05

    def test_in_plus_out_equals_vanilla(self, market, atm_call):
        """Down-and-in + Down-and-out ≈ vanilla (in-out parity)."""
        np.random.seed(123)
        out = price_barrier(
            market, atm_call, barrier=80, barrier_type="down-and-out",
            n_simulations=100000,
        )
        np.random.seed(123)
        in_ = price_barrier(
            market, atm_call, barrier=80, barrier_type="down-and-in",
            n_simulations=100000,
        )
        vanilla = bs_price(market, atm_call)
        combined = out["price"] + in_["price"]
        # Should approximately equal vanilla (within MC noise)
        assert abs(combined - vanilla) < 2.0

    def test_knock_probability_between_0_1(self, market, atm_call):
        result = price_barrier(
            market, atm_call, barrier=80, barrier_type="down-and-out",
            n_simulations=10000,
        )
        assert 0 <= result["knock_probability"] <= 1

    def test_barrier_fields(self, market, atm_call):
        result = price_barrier(
            market, atm_call, barrier=120, barrier_type="up-and-out",
            n_simulations=10000,
        )
        assert result["exotic_type"] == "barrier"
        assert result["barrier"] == 120
        assert result["barrier_type"] == "up-and-out"


class TestLookbackOptions:
    def test_lookback_more_expensive_than_vanilla(self, market, atm_call):
        """Lookback always ≥ vanilla (hindsight is valuable)."""
        lookback = price_lookback(market, atm_call, n_simulations=50000)
        vanilla_bs = bs_price(market, atm_call)
        assert lookback["price"] > vanilla_bs * 0.9  # allow MC noise

    def test_lookback_always_positive(self, market, atm_call):
        """Lookback is almost always ITM."""
        result = price_lookback(market, atm_call, n_simulations=10000)
        assert result["price"] > 0

    def test_lookback_fields(self, market, atm_call):
        result = price_lookback(market, atm_call, n_simulations=10000)
        assert result["exotic_type"] == "lookback"


class TestDigitalOptions:
    def test_digital_call_less_than_payout(self, market, atm_call):
        """Digital option price ≤ discounted payout."""
        result = price_digital(market, atm_call, payout=100.0, n_simulations=50000)
        max_price = 100.0 * np.exp(-market.rate * market.maturity)
        assert result["price"] <= max_price * 1.01

    def test_digital_itm_probability(self, market, atm_call):
        result = price_digital(market, atm_call, n_simulations=50000)
        assert 0.3 < result["itm_probability"] < 0.8  # ATM ≈ 50-60%

    def test_digital_fields(self, market, atm_call):
        result = price_digital(market, atm_call, n_simulations=10000)
        assert result["exotic_type"] == "digital"
        assert "itm_probability" in result
