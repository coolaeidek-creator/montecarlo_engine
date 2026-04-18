"""Tests for Dividend Adjustments."""

import pytest
import numpy as np
from engine.dividend import (
    bs_with_continuous_dividend,
    bs_with_discrete_dividends,
    dividend_schedule,
)


class TestContinuousDividend:
    def test_zero_div_matches_bs(self):
        """q=0 should match standard BS."""
        r = bs_with_continuous_dividend(100, 100, 0.05, 0.0, 0.2, 1.0, "call")
        # BS call ~ 10.45 for these params
        assert abs(r["price"] - 10.45) < 0.1

    def test_higher_div_lower_call(self):
        low = bs_with_continuous_dividend(100, 100, 0.05, 0.01, 0.2, 1.0, "call")
        high = bs_with_continuous_dividend(100, 100, 0.05, 0.05, 0.2, 1.0, "call")
        assert high["price"] < low["price"]

    def test_higher_div_higher_put(self):
        low = bs_with_continuous_dividend(100, 100, 0.05, 0.01, 0.2, 1.0, "put")
        high = bs_with_continuous_dividend(100, 100, 0.05, 0.05, 0.2, 1.0, "put")
        assert high["price"] > low["price"]

    def test_delta_call_positive(self):
        r = bs_with_continuous_dividend(100, 100, 0.05, 0.03, 0.2, 1.0, "call")
        assert 0 < r["delta"] < 1

    def test_delta_put_negative(self):
        r = bs_with_continuous_dividend(100, 100, 0.05, 0.03, 0.2, 1.0, "put")
        assert -1 < r["delta"] < 0


class TestDiscreteDividend:
    def test_no_divs_matches_vanilla(self):
        r = bs_with_discrete_dividends(100, 100, 0.05, 0.2, 1.0, [], "call")
        assert abs(r["price"] - r["vanilla_price"]) < 1e-6

    def test_div_reduces_call(self):
        r = bs_with_discrete_dividends(100, 100, 0.05, 0.2, 1.0, [(0.5, 2.0)], "call")
        assert r["price"] < r["vanilla_price"]

    def test_div_increases_put(self):
        r = bs_with_discrete_dividends(100, 100, 0.05, 0.2, 1.0, [(0.5, 2.0)], "put")
        assert r["price"] > r["vanilla_price"]

    def test_pv_dividends_positive(self):
        r = bs_with_discrete_dividends(100, 100, 0.05, 0.2, 1.0, [(0.25, 1.0), (0.75, 1.0)], "call")
        assert r["pv_dividends"] > 0
        assert r["n_dividends"] == 2


class TestDividendSchedule:
    def test_quarterly(self):
        sched = dividend_schedule(0.04, 1.0, "quarterly", 100)
        assert len(sched) == 4

    def test_monthly(self):
        sched = dividend_schedule(0.04, 1.0, "monthly", 100)
        assert len(sched) == 12

    def test_amounts_sum_to_yield(self):
        sched = dividend_schedule(0.04, 1.0, "quarterly", 100)
        total = sum(d for _, d in sched)
        assert abs(total - 4.0) < 1e-6
