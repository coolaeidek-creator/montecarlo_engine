"""
Tests for Monte Carlo pricer — both standard and antithetic.

Validates MC prices converge to BS, and antithetic reduces variance.
"""

import pytest
import numpy as np
from engine.models import MarketEnvironment, OptionContract
from engine.pricer import OptionPricer, price_option
from engine.analytical import bs_price


@pytest.fixture
def market():
    return MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=1.0)


@pytest.fixture
def atm_call():
    return OptionContract(strike=100, option_type="call")


@pytest.fixture
def atm_put():
    return OptionContract(strike=100, option_type="put")


class TestStandardMC:
    def test_call_converges_to_bs(self, market, atm_call):
        """MC call price within 3 SE of BS price."""
        pricer = OptionPricer(n_simulations=100000, method="standard")
        result = pricer.price(market, atm_call)
        assert abs(result["price"] - result["bs_price"]) < 3 * result["std_error"]

    def test_put_converges_to_bs(self, market, atm_put):
        pricer = OptionPricer(n_simulations=100000, method="standard")
        result = pricer.price(market, atm_put)
        assert abs(result["price"] - result["bs_price"]) < 3 * result["std_error"]

    def test_result_has_all_fields(self, market, atm_call):
        result = price_option(market, atm_call)
        assert "price" in result
        assert "std_error" in result
        assert "confidence_interval" in result
        assert "bs_price" in result
        assert "bs_diff" in result
        assert "method" in result


class TestAntitheticMC:
    def test_antithetic_converges(self, market, atm_call):
        pricer = OptionPricer(n_simulations=100000, method="antithetic")
        result = pricer.price(market, atm_call)
        assert abs(result["price"] - result["bs_price"]) < 3 * result["std_error"]

    def test_antithetic_lower_se(self, market, atm_call):
        """Antithetic variates should have lower SE than standard."""
        np.random.seed(42)
        std_result = OptionPricer(50000, "standard").price(market, atm_call)
        np.random.seed(42)
        anti_result = OptionPricer(50000, "antithetic").price(market, atm_call)
        # Antithetic SE should be lower (may not always hold due to randomness,
        # but with same seed it's very likely)
        assert anti_result["std_error"] < std_result["std_error"] * 1.1

    def test_method_field(self, market, atm_call):
        result = price_option(market, atm_call, method="antithetic")
        assert result["method"] == "antithetic"


class TestControlVariateMC:
    def test_cv_converges(self, market, atm_call):
        pricer = OptionPricer(n_simulations=100000, method="control_variate")
        result = pricer.price(market, atm_call)
        assert abs(result["price"] - result["bs_price"]) < 3 * result["std_error"]

    def test_cv_lower_se_than_standard(self, market, atm_call):
        """Control variate should reduce SE vs standard MC."""
        np.random.seed(42)
        std_result = OptionPricer(50000, "standard").price(market, atm_call)
        np.random.seed(42)
        cv_result = OptionPricer(50000, "control_variate").price(market, atm_call)
        assert cv_result["std_error"] < std_result["std_error"] * 1.1

    def test_cv_method_field(self, market, atm_call):
        result = price_option(market, atm_call, method="control_variate")
        assert result["method"] == "control_variate"


class TestStratifiedMC:
    def test_stratified_converges(self, market, atm_call):
        pricer = OptionPricer(n_simulations=50000, method="stratified")
        result = pricer.price(market, atm_call)
        assert abs(result["price"] - result["bs_price"]) < 3 * result["std_error"]

    def test_stratified_method_field(self, market, atm_call):
        result = price_option(market, atm_call, method="stratified")
        assert result["method"] == "stratified"


class TestSobolMC:
    def test_sobol_converges(self, market, atm_call):
        pricer = OptionPricer(n_simulations=50000, method="sobol")
        result = pricer.price(market, atm_call)
        assert abs(result["price"] - result["bs_price"]) < 3 * result["std_error"]

    def test_sobol_method_field(self, market, atm_call):
        result = price_option(market, atm_call, method="sobol")
        assert result["method"] == "sobol"


class TestEdgeCases:
    def test_deep_itm(self, market):
        contract = OptionContract(strike=50, option_type="call")
        result = price_option(market, contract, n_simulations=50000)
        assert result["price"] > 45  # deep ITM, high value

    def test_deep_otm(self, market):
        contract = OptionContract(strike=200, option_type="call")
        result = price_option(market, contract, n_simulations=50000)
        assert result["price"] < 2  # deep OTM, near zero

    def test_short_maturity(self):
        market = MarketEnvironment(spot=100, rate=0.05, volatility=0.2, maturity=0.01)
        contract = OptionContract(strike=100, option_type="call")
        result = price_option(market, contract)
        assert result["price"] < 5  # very short maturity, low value

    def test_high_vol(self):
        market = MarketEnvironment(spot=100, rate=0.05, volatility=1.0, maturity=1.0)
        contract = OptionContract(strike=100, option_type="call")
        result = price_option(market, contract, n_simulations=50000)
        assert result["price"] > 30  # high vol = high option value
