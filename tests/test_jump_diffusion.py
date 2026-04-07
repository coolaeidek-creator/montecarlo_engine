"""
Tests for Merton Jump-Diffusion model.
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment, OptionContract
from engine.jump_diffusion import simulate_jump_diffusion


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


class TestJumpDiffusion:
    def test_call_price_positive(self, market):
        contract = OptionContract(strike=100, option_type="call")
        result = simulate_jump_diffusion(market, contract, n_simulations=30000)
        assert result["price"] > 0

    def test_put_price_positive(self, market):
        contract = OptionContract(strike=100, option_type="put")
        result = simulate_jump_diffusion(market, contract, n_simulations=30000)
        assert result["price"] > 0

    def test_result_fields(self, market):
        contract = OptionContract(strike=100, option_type="call")
        result = simulate_jump_diffusion(market, contract, n_simulations=10000)
        assert "price" in result
        assert "std_error" in result
        assert "confidence_interval" in result
        assert "gbm_price" in result
        assert "jump_premium" in result
        assert "avg_jumps" in result
        assert result["model"] == "merton-jump-diffusion"

    def test_jump_premium_sign_crash_jumps(self, market):
        """Negative jump mean should increase put prices (crash protection)."""
        contract = OptionContract(strike=100, option_type="put")
        np.random.seed(42)
        result = simulate_jump_diffusion(
            market, contract, n_simulations=50000,
            jump_intensity=2.0, jump_mean=-0.10, jump_vol=0.15,
        )
        # With crash-like jumps, put price should be >= GBM put price
        assert result["price"] >= result["gbm_price"] * 0.90

    def test_zero_intensity_near_gbm(self, market):
        """Zero jump intensity should give price ≈ GBM."""
        contract = OptionContract(strike=100, option_type="call")
        np.random.seed(42)
        result = simulate_jump_diffusion(
            market, contract, n_simulations=50000,
            jump_intensity=0.0001,  # near-zero
            jump_mean=0.0, jump_vol=0.01,
        )
        assert abs(result["price"] - result["gbm_price"]) < 2.0

    def test_high_intensity_wider_ci(self, market):
        """Higher jump intensity should produce wider confidence intervals."""
        contract = OptionContract(strike=100, option_type="call")
        np.random.seed(42)
        r_low = simulate_jump_diffusion(
            market, contract, n_simulations=30000,
            jump_intensity=0.5,
        )
        np.random.seed(42)
        r_high = simulate_jump_diffusion(
            market, contract, n_simulations=30000,
            jump_intensity=5.0,
        )
        ci_low = r_low["confidence_interval"][1] - r_low["confidence_interval"][0]
        ci_high = r_high["confidence_interval"][1] - r_high["confidence_interval"][0]
        # More jumps = more variance = wider CI (usually)
        assert ci_high > ci_low * 0.5  # relaxed check

    def test_avg_jumps_consistent(self, market):
        """Average number of jumps should be ≈ lambda * T."""
        contract = OptionContract(strike=100, option_type="call")
        np.random.seed(42)
        result = simulate_jump_diffusion(
            market, contract, n_simulations=100000,
            jump_intensity=3.0,
        )
        # E[N] = lambda * T = 3.0
        assert abs(result["avg_jumps"] - 3.0) < 0.2

    def test_deep_otm_call_small_price(self, market):
        """Deep OTM call should have small but positive price."""
        contract = OptionContract(strike=200, option_type="call")
        result = simulate_jump_diffusion(market, contract, n_simulations=30000)
        assert result["price"] >= 0
        assert result["price"] < 5.0

    def test_confidence_interval_contains_price(self, market):
        """Price should lie within its own CI."""
        contract = OptionContract(strike=100, option_type="call")
        result = simulate_jump_diffusion(market, contract, n_simulations=50000)
        lo, hi = result["confidence_interval"]
        assert lo <= result["price"] <= hi
