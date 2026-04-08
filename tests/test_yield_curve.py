"""
Tests for Yield Curve / Term Structure module.
"""

import pytest
import numpy as np
from engine.yield_curve import (
    nelson_siegel, flat_curve, generate_yield_curve,
    price_with_term_structure,
)


class TestNelsonSiegel:
    def test_flat_at_long_maturity(self):
        """At very long maturities, NS approaches beta0."""
        rate = nelson_siegel(100.0, beta0=0.05)
        assert abs(rate - 0.05) < 0.005

    def test_short_rate(self):
        """At T→0, rate → beta0 + beta1."""
        rate = nelson_siegel(0.01, beta0=0.05, beta1=-0.02)
        assert abs(rate - 0.03) < 0.01

    def test_positive_rates(self):
        """Default params should produce positive rates."""
        T = np.linspace(0.1, 30, 50)
        rates = nelson_siegel(T)
        assert np.all(rates > 0)

    def test_array_input(self):
        """Should accept array inputs."""
        T = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        rates = nelson_siegel(T)
        assert len(rates) == 5


class TestFlatCurve:
    def test_constant(self):
        rates = flat_curve(np.array([1.0, 5.0, 10.0]), rate=0.03)
        assert np.allclose(rates, 0.03)


class TestGenerateYieldCurve:
    def test_flat_curve(self):
        result = generate_yield_curve(model="flat", rate=0.04, n_points=20)
        assert len(result["maturities"]) == 20
        assert all(abs(r - 0.04) < 1e-10 for r in result["rates"])

    def test_nelson_siegel_curve(self):
        result = generate_yield_curve(model="nelson-siegel", n_points=30)
        assert len(result["rates"]) == 30
        assert len(result["discount_factors"]) == 30
        assert len(result["forward_rates"]) == 30
        assert all(df > 0 and df <= 1.0 for df in result["discount_factors"])

    def test_inverted_curve(self):
        result = generate_yield_curve(model="inverted", n_points=20)
        # Short rates should be higher than long rates
        assert result["rates"][0] > result["rates"][-1]

    def test_steep_curve(self):
        result = generate_yield_curve(model="steep", n_points=20)
        # Long rates should be notably higher
        assert result["rates"][-1] > result["rates"][0]


class TestPriceWithTermStructure:
    def test_basic_pricing(self):
        result = price_with_term_structure(
            spot=100, strike=100, volatility=0.2,
            maturity=1.0, option_type="call",
        )
        assert result["curve_price"] > 0
        assert result["flat_price"] > 0
        assert "rate_impact" in result

    def test_rate_impact_sign(self):
        """Steep curve at long maturity should give higher rate than flat."""
        result = price_with_term_structure(
            spot=100, strike=100, volatility=0.2,
            maturity=10.0, option_type="call",
            curve_model="steep",
        )
        # At long maturity, steep curve rate should exceed flat 5%
        assert result["curve_rate"] > result["flat_rate"]
        assert result["rate_impact"] > 0

    def test_inverted_vs_flat_put(self):
        """Inverted curve at long maturity has lower rate → higher put price."""
        result = price_with_term_structure(
            spot=100, strike=100, volatility=0.2,
            maturity=10.0, option_type="put",
            curve_model="inverted",
        )
        # Inverted curve: lower long rate → higher put value
        assert result["curve_rate"] < result["flat_rate"]
        assert result["rate_impact"] > 0
