"""
Tests for implied volatility solver.
"""

import pytest
import numpy as np
from engine.implied_vol import implied_volatility, bs_price_for_iv, compute_smile


class TestImpliedVol:
    def test_roundtrip_call(self):
        """Compute BS price, then recover vol from it."""
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.25
        price = bs_price_for_iv(S, K, r, T, sigma, "call")
        result = implied_volatility(price, S, K, r, T, "call")
        assert result["converged"]
        assert abs(result["iv"] - sigma) < 1e-6

    def test_roundtrip_put(self):
        S, K, r, T, sigma = 100, 110, 0.05, 0.5, 0.30
        price = bs_price_for_iv(S, K, r, T, sigma, "put")
        result = implied_volatility(price, S, K, r, T, "put")
        assert result["converged"]
        assert abs(result["iv"] - sigma) < 1e-6

    def test_itm_call(self):
        S, K, r, T, sigma = 100, 80, 0.05, 1.0, 0.20
        price = bs_price_for_iv(S, K, r, T, sigma, "call")
        result = implied_volatility(price, S, K, r, T, "call")
        assert result["converged"]
        assert abs(result["iv"] - sigma) < 1e-4

    def test_otm_put(self):
        S, K, r, T, sigma = 100, 80, 0.05, 1.0, 0.35
        price = bs_price_for_iv(S, K, r, T, sigma, "put")
        result = implied_volatility(price, S, K, r, T, "put")
        assert result["converged"]
        assert abs(result["iv"] - sigma) < 1e-4

    def test_high_vol(self):
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.80
        price = bs_price_for_iv(S, K, r, T, sigma, "call")
        result = implied_volatility(price, S, K, r, T, "call")
        assert result["converged"]
        assert abs(result["iv"] - sigma) < 1e-4

    def test_newton_raphson_method(self):
        S, K, r, T, sigma = 100, 100, 0.05, 1.0, 0.25
        price = bs_price_for_iv(S, K, r, T, sigma, "call")
        result = implied_volatility(price, S, K, r, T, "call")
        assert result["method"] == "newton-raphson"
        assert result["iterations"] < 20


class TestVolSmile:
    def test_smile_output(self):
        strikes = [90, 95, 100, 105, 110]
        # Generate BS prices at vol=25% then solve back
        prices = [bs_price_for_iv(100, K, 0.05, 1.0, 0.25, "call") for K in strikes]
        smile = compute_smile(100, 0.05, 1.0, strikes, prices, "call")
        assert len(smile) == 5
        for pt in smile:
            assert pt["converged"]
            assert abs(pt["iv"] - 0.25) < 1e-4
