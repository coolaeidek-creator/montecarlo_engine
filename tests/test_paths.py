"""
Tests for multi-step path simulator.
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment
from engine.paths import simulate_paths, simulate_paths_with_times


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


class TestPaths:
    def test_shape(self, market):
        paths = simulate_paths(market, n_simulations=1000, n_steps=50)
        assert paths.shape == (1000, 51)  # 50 steps + initial

    def test_initial_price(self, market):
        paths = simulate_paths(market, n_simulations=100, n_steps=10)
        assert np.allclose(paths[:, 0], 100.0)

    def test_positive_prices(self, market):
        paths = simulate_paths(market, n_simulations=1000, n_steps=100)
        assert np.all(paths > 0)

    def test_antithetic_paths(self, market):
        paths = simulate_paths(market, n_simulations=1000, n_steps=50, antithetic=True)
        assert paths.shape == (1000, 51)

    def test_mean_terminal_near_forward(self, market):
        """Mean terminal price should approximate forward price S*e^(rT)."""
        np.random.seed(42)
        paths = simulate_paths(market, n_simulations=50000, n_steps=252)
        mean_terminal = np.mean(paths[:, -1])
        forward = 100 * np.exp(0.05)
        assert abs(mean_terminal - forward) < 2.0  # within $2

    def test_paths_with_times(self, market):
        paths, times = simulate_paths_with_times(market, n_simulations=100, n_steps=10)
        assert len(times) == 11
        assert times[0] == 0
        assert abs(times[-1] - 1.0) < 1e-10
        assert paths.shape[1] == len(times)
