"""Tests for Quanto Options."""

import pytest
import numpy as np
from engine.quanto import quanto_bs_price, quanto_mc


class TestQuantoBS:
    def test_call_positive(self):
        r = quanto_bs_price(100, 100, 0.05, 0.03, 0.2, 0.1, 0.3, 1.0, "call")
        assert r["price"] > 0

    def test_put_positive(self):
        r = quanto_bs_price(100, 100, 0.05, 0.03, 0.2, 0.1, 0.3, 1.0, "put")
        assert r["price"] > 0

    def test_negative_corr_higher_call(self):
        """Negative correlation should increase quanto call price (lower drag)."""
        r_pos = quanto_bs_price(100, 100, 0.05, 0.03, 0.2, 0.15, 0.5, 1.0, "call")
        r_neg = quanto_bs_price(100, 100, 0.05, 0.03, 0.2, 0.15, -0.5, 1.0, "call")
        assert r_neg["price"] > r_pos["price"]

    def test_zero_corr_zero_fx_vol(self):
        """Zero correlation or zero FX vol => adjustment term is zero."""
        r = quanto_bs_price(100, 100, 0.05, 0.05, 0.2, 0.0, 0.3, 1.0, "call")
        # When fx_vol=0 and rates equal, quanto ~ vanilla
        assert abs(r["price"] - r["vanilla_price"]) < 0.01

    def test_fields(self):
        r = quanto_bs_price(100, 100, 0.05, 0.03, 0.2, 0.1, 0.3, 1.0, "call")
        assert "price" in r
        assert "vanilla_price" in r
        assert "quanto_adjustment" in r
        assert "adjusted_drift" in r
        assert r["model"] == "quanto-bs"


class TestQuantoMC:
    def test_call_positive(self):
        np.random.seed(42)
        r = quanto_mc(100, 100, 0.05, 0.03, 0.2, 0.1, 0.3, 1.0, "call", 20000)
        assert r["price"] > 0

    def test_mc_close_to_bs(self):
        np.random.seed(42)
        bs = quanto_bs_price(100, 100, 0.05, 0.03, 0.2, 0.1, 0.3, 1.0, "call")
        mc = quanto_mc(100, 100, 0.05, 0.03, 0.2, 0.1, 0.3, 1.0, "call", 100000)
        assert abs(bs["price"] - mc["price"]) < 0.5

    def test_std_error_positive(self):
        r = quanto_mc(100, 100, 0.05, 0.03, 0.2, 0.1, 0.3, 1.0, "call", 10000)
        assert r["std_error"] > 0
