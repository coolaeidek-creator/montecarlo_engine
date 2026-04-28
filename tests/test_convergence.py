"""
Tests for Monte Carlo convergence test module.
"""

import pytest

from engine.analytical import bs_price
from engine.convergence import mc_convergence_test
from engine.models import MarketEnvironment, OptionContract


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


@pytest.fixture
def contract():
    return OptionContract(strike=100, option_type="call")


class TestConvergenceOutput:
    def test_returns_methods(self, market, contract):
        out = mc_convergence_test(
            market, contract,
            sample_sizes=[500, 2000],
            methods=["standard", "antithetic"],
        )
        assert set(out["methods"].keys()) == {"standard", "antithetic"}

    def test_bs_reference_matches_analytical(self, market, contract):
        out = mc_convergence_test(
            market, contract,
            sample_sizes=[500],
            methods=["standard"],
        )
        assert abs(out["bs_reference"] - bs_price(market, contract)) < 1e-9

    def test_trajectory_length(self, market, contract):
        sizes = [500, 1000, 2000, 5000]
        out = mc_convergence_test(
            market, contract,
            sample_sizes=sizes,
            methods=["standard"],
        )
        assert len(out["methods"]["standard"]["trajectory"]) == len(sizes)

    def test_trajectory_fields(self, market, contract):
        out = mc_convergence_test(
            market, contract,
            sample_sizes=[500, 1000],
            methods=["standard"],
        )
        row = out["methods"]["standard"]["trajectory"][0]
        for key in ("n", "price", "std_error", "abs_error", "rel_error_pct", "ci_width"):
            assert key in row


class TestConvergenceBehavior:
    def test_standard_se_decays(self, market, contract):
        """Standard MC SE should decrease as N grows."""
        out = mc_convergence_test(
            market, contract,
            sample_sizes=[500, 5000, 50000],
            methods=["standard"],
        )
        ses = [t["std_error"] for t in out["methods"]["standard"]["trajectory"]]
        assert ses[-1] < ses[0]

    def test_convergence_rate_near_half(self, market, contract):
        """Standard MC SE should decay roughly as N^-0.5."""
        out = mc_convergence_test(
            market, contract,
            sample_sizes=[1000, 5000, 20000, 100000],
            methods=["standard"],
        )
        rate = out["methods"]["standard"]["convergence_rate"]
        # Theory: -0.5; allow generous tolerance for sampling noise.
        assert -0.7 < rate < -0.3

    def test_antithetic_reduces_variance(self, market, contract):
        """Antithetic should produce lower SE than standard at same N."""
        out = mc_convergence_test(
            market, contract,
            sample_sizes=[10000],
            methods=["standard", "antithetic"],
        )
        std_se = out["methods"]["standard"]["final_std_error"]
        anti_se = out["methods"]["antithetic"]["final_std_error"]
        assert anti_se < std_se

    def test_variance_reduction_factor(self, market, contract):
        """Antithetic VR factor relative to standard should be > 1."""
        out = mc_convergence_test(
            market, contract,
            sample_sizes=[10000],
            methods=["standard", "antithetic"],
        )
        vrf = out["methods"]["antithetic"]["variance_reduction_factor"]
        assert vrf is not None and vrf > 1.0

    def test_final_price_close_to_bs(self, market, contract):
        """At large N, MC price should be close to BS."""
        out = mc_convergence_test(
            market, contract,
            sample_sizes=[100000],
            methods=["antithetic"],
        )
        rel = out["methods"]["antithetic"]["final_rel_error_pct"]
        assert rel < 2.0  # < 2% rel error at 100K antithetic


class TestConvergenceValidation:
    def test_rejects_tiny_sample_size(self, market, contract):
        with pytest.raises(ValueError):
            mc_convergence_test(
                market, contract,
                sample_sizes=[1],
                methods=["standard"],
            )

    def test_rejects_empty_methods(self, market, contract):
        with pytest.raises(ValueError):
            mc_convergence_test(
                market, contract,
                sample_sizes=[500],
                methods=[],
            )

    def test_default_args(self, market, contract):
        """Should work with no sample_sizes/methods (use defaults)."""
        out = mc_convergence_test(market, contract)
        assert "methods" in out
        assert len(out["sample_sizes"]) > 0
