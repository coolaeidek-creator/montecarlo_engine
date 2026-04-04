"""
Monte Carlo Options Pricing Engine v2.0

Modules:
    models      — MarketEnvironment, OptionContract (Pydantic)
    pricer      — MC pricer (standard, antithetic, control variate)
    analytical  — Black-Scholes closed-form pricing
    greeks      — Full Greeks (Delta, Gamma, Theta, Vega, Rho)
    exotic      — Asian, Barrier, Lookback, Digital options
    risk        — VaR, CVaR, portfolio risk metrics
    implied_vol — Newton-Raphson IV solver
    vol_surface — Volatility surface generation (SVI)
    paths       — Multi-step GBM path simulator
    stocks      — 40 real stocks across 4 global regions
"""

from .models import MarketEnvironment, OptionContract
from .pricer import OptionPricer, price_option
from .analytical import bs_price, bs_price_both
from .greeks import compute_greeks, compute_greeks_both

__version__ = "2.0.0"
