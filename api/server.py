"""
FastAPI REST API for the Monte Carlo Options Pricing Engine.

Endpoints:
  POST /api/price          — Price vanilla European option (MC + BS)
  POST /api/exotic         — Price exotic options (Asian, Barrier, Lookback, Digital)
  POST /api/greeks         — Compute BS Greeks
  POST /api/risk           — Compute VaR, CVaR risk metrics
  POST /api/iv             — Solve for implied volatility
  POST /api/vol-surface    — Generate volatility surface
  POST /api/stock-price    — Price using predefined stock data
  GET  /api/stocks         — List all available stocks by region
  GET  /api/regions        — List available regions
  GET  /api/health         — Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PriceRequest, ExoticRequest, RiskRequest,
    IVRequest, GreeksRequest, VolSurfaceRequest, StockPriceRequest,
    JumpDiffusionRequest, MCGreeksRequest, AmericanRequest,
    HestonRequest, YieldCurveRequest,
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


app = FastAPI(
    title="Monte Carlo Options Pricing Engine",
    description="Industry-grade options pricing with MC simulation, BS validation, "
                "exotic options, risk metrics, and implied volatility.",
    version="2.0.0",
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
    return {"status": "ok", "engine": "monte-carlo-v2.1", "features": [
        "vanilla_mc", "antithetic_variates", "black_scholes",
        "greeks", "mc_greeks", "exotic_options", "american_lsm",
        "jump_diffusion", "risk_metrics",
        "implied_volatility", "vol_surface", "40_stocks_4_regions",
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
