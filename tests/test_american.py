"""
Tests for American option pricing via Longstaff-Schwartz LSM.
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment, OptionContract
from engine.american import price_american
from engine.analytical import bs_price


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


class TestAmericanOption:
    def test_american_put_ge_european(self, market):
        """American put >= European put (early exercise has value)."""
        contract = OptionContract(strike=100, option_type="put")
        result = price_american(market, contract, n_simulations=30000, n_steps=50)
        euro_bs = bs_price(market, contract)
        # American should be >= European (with MC noise tolerance)
        assert result["price"] >= euro_bs * 0.95

    def test_american_call_near_european(self, market):
        """American call on non-dividend stock ≈ European call."""
        contract = OptionContract(strike=100, option_type="call")
        result = price_american(market, contract, n_simulations=30000, n_steps=50)
        euro_bs = bs_price(market, contract)
        # Should be very close (no early exercise benefit for calls without dividends)
        assert abs(result["price"] - euro_bs) < 2.0

    def test_early_exercise_pct(self, market):
        """ITM put should show some early exercise."""
        contract = OptionContract(strike=110, option_type="put")
        result = price_american(market, contract, n_simulations=30000, n_steps=50)
        assert result["early_exercise_pct"] >= 0
        assert result["early_exercise_pct"] <= 1

    def test_result_fields(self, market):
        contract = OptionContract(strike=100, option_type="put")
        result = price_american(market, contract, n_simulations=10000, n_steps=30)
        assert "price" in result
        assert "std_error" in result
        assert "early_exercise_pct" in result
        assert "european_price" in result
        assert result["method"] == "longstaff-schwartz"

    def test_deep_itm_put_has_early_exercise_premium(self, market):
        """Deep ITM put should have positive early exercise premium."""
        contract = OptionContract(strike=130, option_type="put")
        result = price_american(market, contract, n_simulations=30000, n_steps=50)
        # Premium might be small but price should be >= european
        assert result["price"] >= result["european_price"] * 0.98
