"""
Tests for Greeks Surface generator.
"""

import pytest
from engine.greeks_surface import greeks_surface_spot_time, greeks_surface_spot_vol


class TestGreeksSurfaceSpotTime:
    def test_output_shape(self):
        result = greeks_surface_spot_time(
            spot=100, strike=100, rate=0.05, volatility=0.2,
            n_spot=5, n_time=4,
        )
        assert len(result["spots"]) == 5
        assert len(result["times"]) == 4
        assert len(result["delta"]) == 4
        assert len(result["delta"][0]) == 5

    def test_all_greeks_present(self):
        result = greeks_surface_spot_time(
            spot=100, strike=100, rate=0.05, volatility=0.2,
            n_spot=3, n_time=3,
        )
        assert "delta" in result
        assert "gamma" in result
        assert "vega" in result
        assert "theta" in result

    def test_call_delta_range(self):
        result = greeks_surface_spot_time(
            spot=100, strike=100, rate=0.05, volatility=0.2,
            option_type="call", n_spot=5, n_time=3,
        )
        for row in result["delta"]:
            for d in row:
                assert 0 <= d <= 1


class TestGreeksSurfaceSpotVol:
    def test_output_shape(self):
        result = greeks_surface_spot_vol(
            spot=100, strike=100, rate=0.05,
            n_spot=5, n_vol=4,
        )
        assert len(result["spots"]) == 5
        assert len(result["vols"]) == 4
        assert len(result["delta"]) == 4

    def test_gamma_positive(self):
        result = greeks_surface_spot_vol(
            spot=100, strike=100, rate=0.05,
            n_spot=5, n_vol=4,
        )
        for row in result["gamma"]:
            for g in row:
                assert g >= 0
