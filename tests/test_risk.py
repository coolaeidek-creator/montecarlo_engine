"""
Tests for risk metrics module — VaR, CVaR.
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment
from engine.risk import compute_var, compute_portfolio_var


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


class TestVaR:
    def test_parametric_var_positive(self, market):
        result = compute_var(market, confidence=0.95, horizon_days=10, method="parametric")
        assert result["var_dollar"] > 0
        assert result["cvar_dollar"] > 0

    def test_cvar_greater_than_var(self, market):
        """CVaR (Expected Shortfall) ≥ VaR by definition."""
        result = compute_var(market, confidence=0.95, horizon_days=10, method="parametric")
        assert result["cvar_dollar"] >= result["var_dollar"]

    def test_mc_var_positive(self, market):
        result = compute_var(
            market, confidence=0.95, horizon_days=10,
            n_simulations=50000, method="monte_carlo",
        )
        assert result["var_dollar"] > 0

    def test_mc_cvar_ge_var(self, market):
        result = compute_var(
            market, confidence=0.95, horizon_days=10,
            n_simulations=50000, method="monte_carlo",
        )
        assert result["cvar_dollar"] >= result["var_dollar"] * 0.99  # small tolerance

    def test_higher_confidence_higher_var(self, market):
        """99% VaR > 95% VaR."""
        var95 = compute_var(market, confidence=0.95, method="parametric")
        var99 = compute_var(market, confidence=0.99, method="parametric")
        assert var99["var_dollar"] > var95["var_dollar"]

    def test_higher_vol_higher_var(self):
        low_vol = MarketEnvironment(spot=100, rate=0.05, volatility=0.1, maturity=1.0)
        high_vol = MarketEnvironment(spot=100, rate=0.05, volatility=0.4, maturity=1.0)
        var_low = compute_var(low_vol, method="parametric")
        var_high = compute_var(high_vol, method="parametric")
        assert var_high["var_dollar"] > var_low["var_dollar"]

    def test_mc_has_extra_fields(self, market):
        result = compute_var(market, method="monte_carlo", n_simulations=10000)
        assert "max_drawdown" in result
        assert "pnl_skew" in result
        assert "pnl_kurtosis" in result
        assert "worst_case" in result


class TestPortfolioVaR:
    def test_portfolio_var_positive(self):
        markets = [
            MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0),
            MarketEnvironment(spot=50, rate=0.05, volatility=0.3, maturity=1.0),
        ]
        weights = np.array([0.6, 0.4])
        result = compute_portfolio_var(markets, weights, n_simulations=10000)
        assert result["var_dollar"] > 0
        assert result["n_assets"] == 2

    def test_diversification_benefit(self):
        """Portfolio VaR should be ≤ sum of individual VaRs (if not perfectly correlated)."""
        m1 = MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)
        m2 = MarketEnvironment(spot=100, rate=0.05, volatility=0.3, maturity=1.0)

        # Individual VaRs
        var1 = compute_var(m1, method="parametric")["var_dollar"]
        var2 = compute_var(m2, method="parametric")["var_dollar"]

        # Portfolio VaR (uncorrelated)
        port = compute_portfolio_var([m1, m2], np.array([0.5, 0.5]), n_simulations=50000)
        # Diversified VaR should be less than simple sum
        assert port["var_dollar"] < (0.5 * var1 + 0.5 * var2) * 1.1
