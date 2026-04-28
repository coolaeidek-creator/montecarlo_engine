"""
FastAPI REST API for the Monte Carlo Options Pricing Engine v3.1

Endpoints:
  POST /api/price              — Price vanilla European option (MC + BS)
  POST /api/exotic             — Price exotic options (Asian, Barrier, Lookback, Digital)
  POST /api/greeks             — Compute BS Greeks
  POST /api/greeks-surface     — Generate Greeks surface (spot×time or spot×vol)
  POST /api/risk               — Compute VaR, CVaR risk metrics
  POST /api/iv                 — Solve for implied volatility
  POST /api/vol-surface        — Generate volatility surface
  POST /api/stock-price        — Price using predefined stock data
  POST /api/binomial           — Binomial tree pricing (CRR)
  POST /api/binomial/converge  — Binomial convergence analysis
  POST /api/delta-hedge        — Delta hedging simulation
  POST /api/delta-hedge/compare — Compare hedge frequencies
  POST /api/quanto             — Quanto option pricing (BS)
  POST /api/quanto/mc          — Quanto option pricing (MC)
  POST /api/dividend/continuous — BS with continuous dividend yield
  POST /api/dividend/discrete  — BS with discrete cash dividends
  POST /api/sabr/smile         — SABR volatility smile
  POST /api/sabr/surface       — SABR volatility surface
  POST /api/variance-swap      — Variance swap pricing
  POST /api/historical-vol     — Historical volatility estimation
  POST /api/paths              — GBM path simulation
  POST /api/convergence        — MC convergence test (multi-method vs BS)
  GET  /api/stocks             — List all available stocks by region
  GET  /api/regions            — List available regions
  GET  /api/health             — Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PriceRequest, ExoticRequest, RiskRequest,
    IVRequest, GreeksRequest, VolSurfaceRequest, StockPriceRequest,
    JumpDiffusionRequest, MCGreeksRequest, AmericanRequest,
    HestonRequest, YieldCurveRequest,
    BinomialRequest, DeltaHedgeRequest, QuantoRequest,
    DividendContinuousRequest, DividendDiscreteRequest,
    GreeksSurfaceRequest, HistoricalVolRequest, SABRRequest,
    VarianceSwapRequest, PathsRequest, ConvergenceTestRequest,
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.models import MarketEnvironment, OptionContract
from engine.pricer import OptionPricer
from engine.analytical import bs_price
from engine.greeks import compute_greeks
from engine.exotic import price_asian, price_barrier, price_lookback, price_digital
from engine.risk import compute_var
from engine.implied_vol import implied_volatility
from engine.vol_surface import generate_vol_surface, generate_svi_surface
from engine.scenarios import spot_sensitivity, pnl_matrix, stress_test, time_decay_projection
from engine.jump_diffusion import simulate_jump_diffusion
from engine.mc_greeks import mc_greeks
from engine.american import price_american
from engine.heston import price_heston, heston_smile
from engine.yield_curve import generate_yield_curve
from engine.stocks import REGIONS, get_stock, get_region
from engine.binomial import price_binomial, binomial_convergence
from engine.delta_hedge import simulate_delta_hedge, compare_hedge_frequencies
from engine.quanto import quanto_bs_price, quanto_mc
from engine.dividend import bs_with_continuous_dividend, bs_with_discrete_dividends
from engine.greeks_surface import greeks_surface_spot_time, greeks_surface_spot_vol
from engine.historical_vol import close_to_close_vol, ewma_vol
from engine.sabr import sabr_smile, sabr_surface
from engine.variance_swap import price_variance_swap
from engine.paths import simulate_paths
from engine.convergence import mc_convergence_test


app = FastAPI(
    title="Monte Carlo Options Pricing Engine",
    description="Industry-grade options pricing with MC simulation, BS validation, "
                "exotic options, risk metrics, and implied volatility.",
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health & Info ────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "engine": "monte-carlo-v3.1", "features": [
        "vanilla_mc", "antithetic_variates", "black_scholes",
        "greeks", "mc_greeks", "greeks_surface", "exotic_options",
        "american_lsm", "binomial_tree", "jump_diffusion",
        "heston_sv", "sabr", "risk_metrics",
        "implied_volatility", "vol_surface", "yield_curve",
        "delta_hedging", "variance_swaps", "quanto_options",
        "dividend_adjustments", "historical_vol", "path_simulation",
        "40_stocks_4_regions",
    ]}


@app.get("/api/regions")
def list_regions():
    return {
        rid: {
            "name": r.name,
            "currency": r.currency,
            "rate": r.rate,
            "stock_count": len(r.stocks),
        }
        for rid, r in REGIONS.items()
    }


@app.get("/api/stocks")
def list_stocks(region: str = None):
    if region and region not in REGIONS:
        raise HTTPException(404, f"Region '{region}' not found")

    result = {}
    regions = {region: REGIONS[region]} if region else REGIONS
    for rid, r in regions.items():
        result[rid] = [
            {
                "ticker": s.ticker,
                "name": s.name,
                "price": s.price,
                "volatility": s.volatility,
                "sector": s.sector,
                "currency": r.currency,
            }
            for s in r.stocks
        ]
    return result


# ─── Vanilla Pricing ─────────────────────────────────────────────────────────

@app.post("/api/price")
def price_vanilla(req: PriceRequest):
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    pricer = OptionPricer(n_simulations=req.n_simulations, method=req.method)
    result = pricer.price(market, contract)

    # Add Greeks
    greeks = compute_greeks(market, contract)

    return {
        **result,
        "confidence_interval": list(result["confidence_interval"]),
        "greeks": greeks,
    }


# ─── Stock-Based Pricing ─────────────────────────────────────────────────────

@app.post("/api/stock-price")
def price_stock(req: StockPriceRequest):
    stock = get_stock(req.region, req.ticker)
    if not stock:
        raise HTTPException(404, f"Stock '{req.ticker}' not found in region '{req.region}'")

    region = get_region(req.region)
    strike = req.strike or stock.price  # ATM if not specified

    market = MarketEnvironment(
        spot=stock.price, rate=region.rate,
        volatility=stock.volatility, maturity=1.0,
    )
    contract = OptionContract(strike=strike, option_type=req.option_type)
    pricer = OptionPricer(n_simulations=req.n_simulations, method=req.method)
    result = pricer.price(market, contract)
    greeks = compute_greeks(market, contract)

    return {
        **result,
        "confidence_interval": list(result["confidence_interval"]),
        "greeks": greeks,
        "stock": {
            "ticker": stock.ticker,
            "name": stock.name,
            "price": stock.price,
            "sector": stock.sector,
        },
        "region": req.region,
        "currency": region.currency,
    }


# ─── Exotic Options ──────────────────────────────────────────────────────────

@app.post("/api/exotic")
def price_exotic(req: ExoticRequest):
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)

    if req.exotic_type == "asian":
        result = price_asian(
            market, contract, req.n_simulations, req.n_steps, req.averaging,
        )
    elif req.exotic_type == "barrier":
        if req.barrier is None:
            raise HTTPException(400, "Barrier level required for barrier options")
        result = price_barrier(
            market, contract, req.barrier, req.barrier_type,
            req.n_simulations, req.n_steps,
        )
    elif req.exotic_type == "lookback":
        result = price_lookback(market, contract, req.n_simulations, req.n_steps)
    elif req.exotic_type == "digital":
        result = price_digital(market, contract, req.payout, req.n_simulations)
    else:
        raise HTTPException(400, f"Unknown exotic type: {req.exotic_type}")

    result["confidence_interval"] = list(result["confidence_interval"])
    return result


# ─── Greeks ───────────────────────────────────────────────────────────────────

@app.post("/api/greeks")
def get_greeks(req: GreeksRequest):
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    greeks = compute_greeks(market, contract)
    bs = bs_price(market, contract)
    return {"greeks": greeks, "bs_price": bs}


# ─── Risk Metrics ─────────────────────────────────────────────────────────────

@app.post("/api/risk")
def get_risk(req: RiskRequest):
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.horizon_days / 252.0,
    )
    result = compute_var(
        market, req.confidence, req.horizon_days,
        req.n_simulations, req.method,
    )
    return result


# ─── Implied Volatility ──────────────────────────────────────────────────────

@app.post("/api/iv")
def get_implied_vol(req: IVRequest):
    result = implied_volatility(
        market_price=req.market_price,
        spot=req.spot, strike=req.strike,
        rate=req.rate, maturity=req.maturity,
        option_type=req.option_type,
    )
    return result


# ─── Volatility Surface ──────────────────────────────────────────────────────

@app.post("/api/vol-surface")
def get_vol_surface(req: VolSurfaceRequest):
    if req.model == "svi":
        data = generate_svi_surface(req.spot, req.rate)
    else:
        data = generate_vol_surface(req.spot, req.rate, req.base_vol)

    return {
        "surface": data["surface"].tolist(),
        "strikes": data["strikes"].tolist(),
        "maturities": data["maturities"].tolist(),
        "spot": data["spot"],
        "model": req.model,
    }


# ─── Scenario Analysis ───────────────────────────────────────────────────────

@app.post("/api/scenarios/sensitivity")
def get_sensitivity(req: GreeksRequest):
    """Spot sensitivity — price and Greeks across spot range."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    return spot_sensitivity(market, contract)


@app.post("/api/scenarios/pnl-matrix")
def get_pnl_matrix(req: GreeksRequest):
    """P&L matrix across spot and vol shifts."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    return pnl_matrix(market, contract)


@app.post("/api/scenarios/stress-test")
def get_stress_test(req: GreeksRequest):
    """Run historical stress scenarios."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    return stress_test(market, contract)


@app.post("/api/scenarios/time-decay")
def get_time_decay(req: GreeksRequest):
    """Project theta decay over time."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    return time_decay_projection(market, contract)


# ─── Jump-Diffusion ─────────────────────────────────────────────────────────

@app.post("/api/jump-diffusion")
def price_jump_diffusion(req: JumpDiffusionRequest):
    """Price European option under Merton Jump-Diffusion model."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    result = simulate_jump_diffusion(
        market, contract, req.n_simulations,
        req.jump_intensity, req.jump_mean, req.jump_vol,
    )
    result["confidence_interval"] = list(result["confidence_interval"])
    return result


# ─── MC Greeks ───────────────────────────────────────────────────────────────

@app.post("/api/mc-greeks")
def get_mc_greeks(req: MCGreeksRequest):
    """Compute option Greeks via Monte Carlo finite differences."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    result = mc_greeks(market, contract, req.n_simulations, req.method)
    return result


# ─── American Options ───────────────────────────────────────────────────────

@app.post("/api/american")
def price_american_option(req: AmericanRequest):
    """Price American option via Longstaff-Schwartz LSM."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    result = price_american(
        market, contract, req.n_simulations, req.n_steps,
    )
    result["confidence_interval"] = list(result["confidence_interval"])
    return result


# ─── Heston Stochastic Volatility ───────────────────────────────────────────

@app.post("/api/heston")
def heston_price(req: HestonRequest):
    """Price option under Heston stochastic volatility model."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    result = price_heston(
        market, contract, req.n_simulations, req.n_steps,
        req.kappa, req.theta, req.vol_of_vol, req.rho,
    )
    result["confidence_interval"] = list(result["confidence_interval"])
    return result


@app.post("/api/heston/smile")
def heston_vol_smile(req: HestonRequest):
    """Generate implied volatility smile from Heston model."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    return heston_smile(
        market, req.n_simulations, req.n_steps,
        req.kappa, req.theta, req.vol_of_vol, req.rho,
    )


# ─── Yield Curve ────────────────────────────────────────────────────────────

@app.post("/api/yield-curve")
def get_yield_curve(req: YieldCurveRequest):
    """Generate yield curve / term structure."""
    return generate_yield_curve(
        model=req.model, rate=req.rate,
        n_points=req.n_points, max_maturity=req.max_maturity,
    )


# ─── Binomial Tree ─────────────────────────────────────────────────────────

@app.post("/api/binomial")
def binomial_price(req: BinomialRequest):
    """Price option using CRR binomial tree."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    return price_binomial(market, contract, req.n_steps, req.american)


@app.post("/api/binomial/converge")
def binomial_converge(req: BinomialRequest):
    """Show binomial price convergence across step counts."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    return binomial_convergence(market, contract, american=req.american)


# ─── Delta Hedging ─────────────────────────────────────────────────────────

@app.post("/api/delta-hedge")
def delta_hedge(req: DeltaHedgeRequest):
    """Simulate delta hedging of a short option position."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    result = simulate_delta_hedge(
        market, contract, req.n_simulations, req.rebalance_freq,
    )
    result["pnl_95_ci"] = list(result["pnl_95_ci"])
    return result


@app.post("/api/delta-hedge/compare")
def delta_hedge_compare(req: DeltaHedgeRequest):
    """Compare hedge P&L across rebalancing frequencies."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    return compare_hedge_frequencies(market, contract, req.n_simulations)


# ─── Quanto Options ────────────────────────────────────────────────────────

@app.post("/api/quanto")
def quanto_price(req: QuantoRequest):
    """Price quanto option using modified Black-Scholes."""
    return quanto_bs_price(
        spot=req.spot, strike=req.strike,
        rate_domestic=req.rate_domestic, rate_foreign=req.rate_foreign,
        volatility=req.volatility, fx_volatility=req.fx_volatility,
        correlation=req.correlation, maturity=req.maturity,
        option_type=req.option_type, dividend=req.dividend,
    )


@app.post("/api/quanto/mc")
def quanto_monte_carlo(req: QuantoRequest):
    """Price quanto option via Monte Carlo."""
    result = quanto_mc(
        spot=req.spot, strike=req.strike,
        rate_domestic=req.rate_domestic, rate_foreign=req.rate_foreign,
        volatility=req.volatility, fx_volatility=req.fx_volatility,
        correlation=req.correlation, maturity=req.maturity,
        option_type=req.option_type, n_simulations=req.n_simulations,
    )
    result["confidence_interval"] = list(result["confidence_interval"])
    return result


# ─── Dividend Adjustments ──────────────────────────────────────────────────

@app.post("/api/dividend/continuous")
def dividend_continuous(req: DividendContinuousRequest):
    """Price option with continuous dividend yield."""
    return bs_with_continuous_dividend(
        spot=req.spot, strike=req.strike, rate=req.rate,
        dividend_yield=req.dividend_yield, volatility=req.volatility,
        maturity=req.maturity, option_type=req.option_type,
    )


@app.post("/api/dividend/discrete")
def dividend_discrete(req: DividendDiscreteRequest):
    """Price option with discrete cash dividends."""
    divs = [(d[0], d[1]) for d in req.dividends]
    return bs_with_discrete_dividends(
        spot=req.spot, strike=req.strike, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
        dividends=divs, option_type=req.option_type,
    )


# ─── Greeks Surface ────────────────────────────────────────────────────────

@app.post("/api/greeks-surface")
def get_greeks_surface(req: GreeksSurfaceRequest):
    """Generate 2D Greeks surface for visualization."""
    if req.surface_type == "spot-time":
        return greeks_surface_spot_time(
            spot=req.spot, strike=req.strike, rate=req.rate,
            volatility=req.volatility, option_type=req.option_type,
            n_spot=req.n_spot, n_time=req.n_secondary,
        )
    else:
        return greeks_surface_spot_vol(
            spot=req.spot, strike=req.strike, rate=req.rate,
            maturity=req.maturity, option_type=req.option_type,
            n_spot=req.n_spot, n_vol=req.n_secondary,
        )


# ─── Historical Volatility ─────────────────────────────────────────────────

@app.post("/api/historical-vol")
def get_historical_vol(req: HistoricalVolRequest):
    """Estimate historical volatility from price data."""
    if req.method == "ewma":
        result = ewma_vol(req.prices, req.decay)
    else:
        result = close_to_close_vol(req.prices, req.window)
    return result


# ─── SABR Model ────────────────────────────────────────────────────────────

@app.post("/api/sabr/smile")
def get_sabr_smile(req: SABRRequest):
    """Generate SABR volatility smile."""
    return sabr_smile(
        spot=req.spot, rate=req.rate, maturity=req.maturity,
        alpha=req.alpha, beta=req.beta, rho=req.rho,
        sigma0=req.sigma0, n_strikes=req.n_strikes,
    )


@app.post("/api/sabr/surface")
def get_sabr_surface(req: SABRRequest):
    """Generate full SABR volatility surface."""
    return sabr_surface(
        spot=req.spot, rate=req.rate,
        alpha=req.alpha, beta=req.beta, rho=req.rho,
        sigma0=req.sigma0, n_strikes=req.n_strikes,
    )


# ─── Variance Swap ─────────────────────────────────────────────────────────

@app.post("/api/variance-swap")
def variance_swap(req: VarianceSwapRequest):
    """Price variance swap via Monte Carlo."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    return price_variance_swap(
        market, req.n_simulations, req.n_steps, req.notional,
    )


# ─── Path Simulation ──────────────────────────────────────────────────────

@app.post("/api/paths")
def get_paths(req: PathsRequest):
    """Simulate GBM price paths."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    paths = simulate_paths(
        market, req.n_simulations, req.n_steps, req.antithetic,
    )
    time_grid = [i * req.maturity / req.n_steps for i in range(req.n_steps + 1)]
    return {
        "paths": paths.tolist(),
        "time_grid": time_grid,
        "n_simulations": req.n_simulations,
        "n_steps": req.n_steps,
        "spot": req.spot,
    }


# ─── MC Convergence Test ──────────────────────────────────────────────────

@app.post("/api/convergence")
def convergence_test(req: ConvergenceTestRequest):
    """Run MC convergence test across sample sizes and variance-reduction methods."""
    market = MarketEnvironment(
        spot=req.spot, rate=req.rate,
        volatility=req.volatility, maturity=req.maturity,
    )
    contract = OptionContract(strike=req.strike, option_type=req.option_type)
    try:
        return mc_convergence_test(
            market, contract,
            sample_sizes=req.sample_sizes, methods=req.methods,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
