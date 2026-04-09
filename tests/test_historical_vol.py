"""
Tests for Historical Volatility estimators.
"""

import pytest
import numpy as np
from engine.historical_vol import (
    close_to_close_vol, parkinson_vol, garman_klass_vol,
    yang_zhang_vol, ewma_vol, generate_synthetic_ohlc,
)


@pytest.fixture
def ohlc_data():
    np.random.seed(42)
    return generate_synthetic_ohlc(spot=100, volatility=0.2, n_days=252)


class TestCloseToClose:
    def test_basic(self, ohlc_data):
        result = close_to_close_vol(ohlc_data["closes"])
        assert result["current_vol"] > 0
        assert result["full_sample_vol"] > 0
        assert result["method"] == "close-to-close"

    def test_vol_near_true(self, ohlc_data):
        result = close_to_close_vol(ohlc_data["closes"])
        assert abs(result["full_sample_vol"] - 0.2) < 0.10

    def test_rolling_length(self, ohlc_data):
        result = close_to_close_vol(ohlc_data["closes"], window=20)
        assert len(result["rolling_vols"]) == 252 - 20

    def test_too_few_prices(self):
        result = close_to_close_vol([100, 101], window=20)
        assert "error" in result


class TestParkinson:
    def test_basic(self, ohlc_data):
        result = parkinson_vol(ohlc_data["highs"], ohlc_data["lows"])
        assert result["volatility"] > 0
        assert result["method"] == "parkinson"

    def test_efficiency(self, ohlc_data):
        result = parkinson_vol(ohlc_data["highs"], ohlc_data["lows"])
        assert result["efficiency_vs_cc"] > 1


class TestGarmanKlass:
    def test_basic(self, ohlc_data):
        result = garman_klass_vol(
            ohlc_data["opens"], ohlc_data["highs"],
            ohlc_data["lows"], ohlc_data["closes"],
        )
        assert result["volatility"] > 0
        assert result["method"] == "garman-klass"


class TestYangZhang:
    def test_basic(self, ohlc_data):
        result = yang_zhang_vol(
            ohlc_data["opens"], ohlc_data["highs"],
            ohlc_data["lows"], ohlc_data["closes"],
        )
        assert result["volatility"] > 0
        assert result["method"] == "yang-zhang"

    def test_most_efficient(self, ohlc_data):
        result = yang_zhang_vol(
            ohlc_data["opens"], ohlc_data["highs"],
            ohlc_data["lows"], ohlc_data["closes"],
        )
        assert result["efficiency_vs_cc"] > 10


class TestEWMA:
    def test_basic(self, ohlc_data):
        result = ewma_vol(ohlc_data["closes"])
        assert result["current_vol"] > 0
        assert result["method"] == "ewma"

    def test_half_life(self, ohlc_data):
        result = ewma_vol(ohlc_data["closes"], decay=0.94)
        expected_hl = np.log(2) / np.log(1 / 0.94)
        assert abs(result["half_life"] - expected_hl) < 0.01

    def test_ewma_vols_length(self, ohlc_data):
        result = ewma_vol(ohlc_data["closes"])
        assert len(result["ewma_vols"]) == 251  # n-1 returns


class TestSyntheticOHLC:
    def test_output_shape(self):
        np.random.seed(42)
        data = generate_synthetic_ohlc(spot=100, volatility=0.2, n_days=50)
        assert len(data["opens"]) == 50
        assert len(data["highs"]) == 50
        assert len(data["lows"]) == 50
        assert len(data["closes"]) == 50

    def test_high_ge_low(self):
        np.random.seed(42)
        data = generate_synthetic_ohlc(spot=100, volatility=0.3, n_days=100)
        for h, l in zip(data["highs"], data["lows"]):
            assert h >= l
