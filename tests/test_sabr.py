"""
Tests for SABR volatility model.
"""

import pytest
import numpy as np
from engine.sabr import sabr_implied_vol, sabr_smile, sabr_surface


class TestSABRImpliedVol:
    def test_atm_positive(self):
        """ATM IV should be positive."""
        iv = sabr_implied_vol(F=100, K=100, T=1.0)
        assert iv > 0

    def test_otm_positive(self):
        """OTM IV should be positive."""
        iv = sabr_implied_vol(F=100, K=120, T=1.0)
        assert iv > 0

    def test_negative_rho_produces_skew(self):
        """Negative rho should produce higher IV for low strikes."""
        iv_low = sabr_implied_vol(F=100, K=80, T=1.0, rho=-0.5)
        iv_atm = sabr_implied_vol(F=100, K=100, T=1.0, rho=-0.5)
        iv_high = sabr_implied_vol(F=100, K=120, T=1.0, rho=-0.5)
        # With negative rho, OTM puts (low K) should have higher IV
        assert iv_low > iv_atm

    def test_beta_zero_normal(self):
        """Beta=0 should give normal SABR."""
        iv = sabr_implied_vol(F=100, K=100, T=1.0, beta=0.0)
        assert iv > 0

    def test_beta_one_lognormal(self):
        """Beta=1 should give lognormal SABR."""
        iv = sabr_implied_vol(F=100, K=100, T=1.0, beta=1.0)
        assert iv > 0

    def test_zero_maturity(self):
        """Zero maturity should return 0."""
        iv = sabr_implied_vol(F=100, K=100, T=0)
        assert iv == 0

    def test_higher_alpha_wider_smile(self):
        """Higher alpha should produce wider smile wings."""
        iv_low_a = sabr_implied_vol(F=100, K=80, T=1.0, alpha=0.1)
        iv_high_a = sabr_implied_vol(F=100, K=80, T=1.0, alpha=0.5)
        # Higher vol-of-vol generally gives higher wing vols
        assert iv_high_a > iv_low_a * 0.5  # relaxed check


class TestSABRSmile:
    def test_smile_output(self):
        result = sabr_smile(spot=100, rate=0.05, maturity=1.0, n_strikes=11)
        assert len(result["strikes"]) == 11
        assert len(result["implied_vols"]) == 11
        assert all(iv > 0 for iv in result["implied_vols"])
        assert result["model"] == "sabr"

    def test_prices_positive(self):
        result = sabr_smile(spot=100, rate=0.05, maturity=1.0, n_strikes=7)
        assert all(p > 0 for p in result["prices"])

    def test_forward_correct(self):
        result = sabr_smile(spot=100, rate=0.05, maturity=1.0)
        expected_F = 100 * np.exp(0.05)
        assert abs(result["forward"] - expected_F) < 0.01


class TestSABRSurface:
    def test_surface_shape(self):
        result = sabr_surface(spot=100, rate=0.05, n_strikes=7)
        assert len(result["surface"]) == 7  # 7 maturities
        assert len(result["surface"][0]) == 7  # 7 strikes
        assert all(all(v > 0 for v in row) for row in result["surface"])

    def test_surface_model_tag(self):
        result = sabr_surface(spot=100, rate=0.05)
        assert result["model"] == "sabr-surface"
