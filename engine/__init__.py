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
from .jump_diffusion import simulate_jump_diffusion
from .mc_greeks import mc_greeks
from .american import price_american
from .heston import price_heston, heston_smile
from .yield_curve import generate_yield_curve, price_with_term_structure

__version__ = "2.2.0"
