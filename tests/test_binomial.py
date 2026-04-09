"""
Tests for Binomial Tree option pricing.
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment, OptionContract
from engine.binomial import price_binomial, binomial_convergence
from engine.analytical import bs_price


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


class TestBinomialPricing:
    def test_call_positive(self, market):
        contract = OptionContract(strike=100, option_type="call")
        result = price_binomial(market, contract, n_steps=100)
        assert result["price"] > 0

    def test_put_positive(self, market):
        contract = OptionContract(strike=100, option_type="put")
        result = price_binomial(market, contract, n_steps=100)
        assert result["price"] > 0

    def test_european_near_bs(self, market):
        """Binomial European should converge to BS."""
        contract = OptionContract(strike=100, option_type="call")
        result = price_binomial(market, contract, n_steps=500)
        bs = bs_price(market, contract)
        assert abs(result["price"] - bs) < 0.1

    def test_american_put_ge_european(self, market):
        """American put >= European put."""
        contract = OptionContract(strike=100, option_type="put")
        am = price_binomial(market, contract, n_steps=200, american=True)
        eu = price_binomial(market, contract, n_steps=200, american=False)
        assert am["price"] >= eu["price"] - 0.01

    def test_result_fields(self, market):
        contract = OptionContract(strike=100, option_type="call")
        result = price_binomial(market, contract, n_steps=50)
        assert "price" in result
        assert "delta" in result
        assert "gamma" in result
        assert "u" in result
        assert "d" in result
        assert "p" in result
        assert result["method"] == "crr-binomial"

    def test_risk_neutral_prob(self, market):
        """Risk-neutral probability should be between 0 and 1."""
        contract = OptionContract(strike=100, option_type="call")
        result = price_binomial(market, contract, n_steps=100)
        assert 0 < result["p"] < 1

    def test_delta_call_range(self, market):
        """Call delta should be in [0, 1]."""
        contract = OptionContract(strike=100, option_type="call")
        result = price_binomial(market, contract, n_steps=100)
        assert 0 <= result["delta"] <= 1.1  # small tolerance


class TestBinomialConvergence:
    def test_convergence_output(self, market):
        contract = OptionContract(strike=100, option_type="call")
        result = binomial_convergence(
            market, contract, steps_list=[10, 50, 100],
        )
        assert len(result["results"]) == 3

    def test_convergence_improves(self, market):
        """More steps should give price closer to BS."""
        contract = OptionContract(strike=100, option_type="call")
        bs = bs_price(market, contract)
        result = binomial_convergence(
            market, contract, steps_list=[10, 100, 500],
        )
        errs = [abs(r["price"] - bs) for r in result["results"]]
        # Last should be closest to BS
        assert errs[-1] < errs[0]
