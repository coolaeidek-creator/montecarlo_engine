"""
Tests for volatility surface generation.
"""

import pytest
import numpy as np
from engine.vol_surface import generate_vol_surface, generate_svi_surface


class TestVolSurface:
    def test_simple_surface_shape(self):
        data = generate_vol_surface(spot=100, rate=0.05)
        assert data["surface"].shape[0] == len(data["maturities"])
        assert data["surface"].shape[1] == len(data["strikes_pct"])

    def test_surface_positive(self):
        data = generate_vol_surface(spot=100, rate=0.05)
        assert np.all(data["surface"] > 0)

    def test_atm_near_base_vol(self):
        """ATM vol should be close to base_vol."""
        data = generate_vol_surface(spot=100, rate=0.05, base_vol=0.25)
        atm_idx = np.argmin(np.abs(data["strikes_pct"] - 1.0))
        mid_mat_idx = len(data["maturities"]) // 2
        atm_vol = data["surface"][mid_mat_idx, atm_idx]
        assert abs(atm_vol - 0.25) < 0.05

    def test_skew_present(self):
        """OTM puts (low strikes) should have higher vol than OTM calls (high strikes)."""
        data = generate_vol_surface(spot=100, rate=0.05, base_vol=0.25, skew=-0.15)
        mid = len(data["maturities"]) // 2
        low_strike_vol = data["surface"][mid, 0]
        high_strike_vol = data["surface"][mid, -1]
        assert low_strike_vol > high_strike_vol

    def test_svi_surface_shape(self):
        data = generate_svi_surface(spot=100, rate=0.05)
        assert data["surface"].shape[0] == len(data["maturities"])
        assert data["model"] == "SVI"

    def test_svi_surface_positive(self):
        data = generate_svi_surface(spot=100, rate=0.05)
        assert np.all(data["surface"] > 0)

    def test_strikes_scale_with_spot(self):
        data = generate_vol_surface(spot=1000, rate=0.05)
        assert data["strikes"][0] > 500
        assert data["strikes"][-1] < 2000
