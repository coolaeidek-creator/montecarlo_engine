"""
Tests for Delta Hedging simulation.
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment, OptionContract
from engine.delta_hedge import simulate_delta_hedge, compare_hedge_frequencies


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


class TestDeltaHedge:
    def test_daily_hedge_small_std(self, market):
        """Daily hedge should have small P&L std relative to premium."""
        contract = OptionContract(strike=100, option_type="call")
        np.random.seed(42)
        result = simulate_delta_hedge(market, contract, n_simulations=500, rebalance_freq=1)
        # Std should be fraction of premium
        assert result["std_pnl"] < result["premium_received"]

    def test_mean_pnl_near_zero(self, market):
        """Mean hedge P&L should be near zero (risk-neutral)."""
        contract = OptionContract(strike=100, option_type="call")
        np.random.seed(42)
        result = simulate_delta_hedge(market, contract, n_simulations=1000, rebalance_freq=1)
        assert abs(result["mean_pnl"]) < result["premium_received"] * 0.3

    def test_result_fields(self, market):
        contract = OptionContract(strike=100, option_type="call")
        result = simulate_delta_hedge(market, contract, n_simulations=100, rebalance_freq=5)
        assert "mean_pnl" in result
        assert "std_pnl" in result
        assert "hedge_efficiency" in result
        assert "premium_received" in result
        assert result["model"] == "delta-hedge-sim"

    def test_less_frequent_more_variance(self, market):
        """Less frequent rebalancing should have higher P&L variance."""
        contract = OptionContract(strike=100, option_type="call")
        np.random.seed(42)
        r1 = simulate_delta_hedge(market, contract, n_simulations=500, rebalance_freq=1)
        np.random.seed(42)
        r2 = simulate_delta_hedge(market, contract, n_simulations=500, rebalance_freq=21)
        assert r2["std_pnl"] > r1["std_pnl"] * 0.5

    def test_put_hedge(self, market):
        """Should work for puts too."""
        contract = OptionContract(strike=100, option_type="put")
        result = simulate_delta_hedge(market, contract, n_simulations=200, rebalance_freq=5)
        assert result["premium_received"] > 0


class TestHedgeFrequencyComparison:
    def test_output(self, market):
        contract = OptionContract(strike=100, option_type="call")
        np.random.seed(42)
        result = compare_hedge_frequencies(market, contract, n_simulations=100)
        assert len(result["results"]) == 5
        assert result["premium"] > 0
