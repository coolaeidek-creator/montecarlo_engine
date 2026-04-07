"""
Tests for scenario analysis and stress testing module.
"""

import pytest
from engine.models import MarketEnvironment, OptionContract
from engine.scenarios import (
    spot_sensitivity, vol_sensitivity, time_decay_projection,
    pnl_matrix, stress_test,
)


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


@pytest.fixture
def atm_call():
    return OptionContract(strike=100, option_type="call")


class TestSpotSensitivity:
    def test_returns_correct_fields(self, market, atm_call):
        result = spot_sensitivity(market, atm_call)
        assert "spots" in result
        assert "prices" in result
        assert "deltas" in result
        assert "gammas" in result
        assert len(result["spots"]) == 25

    def test_price_increases_with_spot_for_call(self, market, atm_call):
        result = spot_sensitivity(market, atm_call)
        # Call price should generally increase with spot
        assert result["prices"][-1] > result["prices"][0]


class TestVolSensitivity:
    def test_price_increases_with_vol(self, market, atm_call):
        result = vol_sensitivity(market, atm_call)
        # ATM option price increases with vol
        assert result["prices"][-1] > result["prices"][0]

    def test_correct_length(self, market, atm_call):
        result = vol_sensitivity(market, atm_call, n_points=10)
        assert len(result["vols"]) == 10


class TestTimeDecay:
    def test_price_decreases_over_time(self, market, atm_call):
        result = time_decay_projection(market, atm_call)
        # Price at T=1 > price near T=0 for ATM option
        assert result["prices"][0] > result["prices"][-1]

    def test_theta_negative(self, market, atm_call):
        result = time_decay_projection(market, atm_call)
        # Theta should be negative for long options
        assert all(t < 0 for t in result["thetas"])


class TestPnlMatrix:
    def test_grid_dimensions(self, market, atm_call):
        result = pnl_matrix(market, atm_call)
        assert len(result["pnl_grid"]) == len(result["spot_shifts"])
        assert len(result["pnl_grid"][0]) == len(result["vol_shifts"])

    def test_center_is_zero(self, market, atm_call):
        result = pnl_matrix(market, atm_call)
        # Center cell (0% spot, 0% vol shift) should be ~0
        center_row = len(result["spot_shifts"]) // 2
        center_col = len(result["vol_shifts"]) // 2
        assert abs(result["pnl_grid"][center_row][center_col]) < 0.01


class TestStressTest:
    def test_returns_scenarios(self, market, atm_call):
        results = stress_test(market, atm_call)
        assert len(results) == 9
        assert all("pnl" in r for r in results)
        assert all("name" in r for r in results)

    def test_crash_scenarios_negative_pnl_for_call(self, market, atm_call):
        results = stress_test(market, atm_call)
        # Black Monday should hurt a long call (spot drops hard)
        black_monday = next(r for r in results if "Black Monday" in r["name"])
        assert black_monday["pnl"] < 0

    def test_bull_rally_positive_pnl_for_call(self, market, atm_call):
        results = stress_test(market, atm_call)
        bull = next(r for r in results if "Bull Rally" in r["name"])
        assert bull["pnl"] > 0
