"""
Tests for Variance Swap pricing.
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment
from engine.variance_swap import price_variance_swap


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


class TestVarianceSwap:
    def test_fair_var_positive(self, market):
        result = price_variance_swap(market, n_simulations=20000, n_steps=100)
        assert result["fair_variance"] > 0
        assert result["fair_volatility"] > 0

    def test_fair_vol_near_implied(self, market):
        """Fair vol should be close to implied vol (with convexity adj)."""
        result = price_variance_swap(market, n_simulations=50000, n_steps=252)
        # Fair variance ≈ implied variance for GBM
        assert abs(result["fair_volatility"] - market.volatility) < 0.05

    def test_result_fields(self, market):
        result = price_variance_swap(market, n_simulations=10000, n_steps=50)
        assert "fair_variance" in result
        assert "fair_volatility" in result
        assert "implied_variance" in result
        assert "variance_risk_premium" in result
        assert "scenarios" in result
        assert result["model"] == "variance-swap"

    def test_scenarios_count(self, market):
        result = price_variance_swap(market, n_simulations=10000, n_steps=50)
        assert len(result["scenarios"]) == 8

    def test_scenario_pnl_sign(self, market):
        """If realized vol >> fair vol, long var swap profits."""
        result = price_variance_swap(market, n_simulations=20000, n_steps=100)
        high_vol_scenario = [s for s in result["scenarios"] if s["realized_vol"] == 50][0]
        assert high_vol_scenario["pnl"] > 0

    def test_realized_var_stats(self, market):
        result = price_variance_swap(market, n_simulations=20000, n_steps=100)
        stats = result["realized_var_stats"]
        assert stats["mean"] > 0
        assert stats["std"] > 0
        assert stats["p5"] < stats["p95"]

    def test_higher_vol_higher_fair_strike(self):
        """Higher implied vol should give higher fair variance strike."""
        m1 = MarketEnvironment(spot=100, rate=0.05, volatility=0.15, maturity=1.0)
        m2 = MarketEnvironment(spot=100, rate=0.05, volatility=0.35, maturity=1.0)
        np.random.seed(42)
        r1 = price_variance_swap(m1, n_simulations=30000, n_steps=100)
        np.random.seed(42)
        r2 = price_variance_swap(m2, n_simulations=30000, n_steps=100)
        assert r2["fair_variance"] > r1["fair_variance"]
