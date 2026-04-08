"""
Pydantic request/response schemas for the API.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


# ─── Request Schemas ──────────────────────────────────────────────────────────

class PriceRequest(BaseModel):
    """Request to price a vanilla European option."""
    spot: float = Field(..., gt=0, description="Current stock price")
    strike: float = Field(..., gt=0, description="Strike price")
    rate: float = Field(description="Risk-free rate (annualized)")
    volatility: float = Field(..., gt=0, description="Annualized volatility")
    maturity: float = Field(..., gt=0, description="Time to expiry in years")
    option_type: Literal["call", "put"] = "call"
    method: Literal["standard", "antithetic"] = "antithetic"
    n_simulations: int = Field(50000, gt=0, le=1000000)


class ExoticRequest(BaseModel):
    """Request to price an exotic option."""
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    rate: float
    volatility: float = Field(..., gt=0)
    maturity: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"
    exotic_type: Literal["asian", "barrier", "lookback", "digital"]
    n_simulations: int = Field(50000, gt=0, le=1000000)
    n_steps: int = Field(252, gt=0, le=1000)
    # Asian-specific
    averaging: Optional[Literal["arithmetic", "geometric"]] = "arithmetic"
    # Barrier-specific
    barrier: Optional[float] = None
    barrier_type: Optional[Literal[
        "up-and-out", "up-and-in", "down-and-out", "down-and-in"
    ]] = "down-and-out"
    # Digital-specific
    payout: Optional[float] = 1.0


class RiskRequest(BaseModel):
    """Request for risk metrics computation."""
    spot: float = Field(..., gt=0)
    rate: float
    volatility: float = Field(..., gt=0)
    confidence: float = Field(0.95, gt=0, lt=1)
    horizon_days: int = Field(10, gt=0, le=252)
    method: Literal["parametric", "monte_carlo"] = "monte_carlo"
    n_simulations: int = Field(50000, gt=0, le=1000000)


class IVRequest(BaseModel):
    """Request to compute implied volatility."""
    market_price: float = Field(..., gt=0)
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    rate: float
    maturity: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"


class GreeksRequest(BaseModel):
    """Request to compute option Greeks."""
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    rate: float
    volatility: float = Field(..., gt=0)
    maturity: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"


class VolSurfaceRequest(BaseModel):
    """Request to generate a volatility surface."""
    spot: float = Field(..., gt=0)
    rate: float = Field(0.05)
    base_vol: float = Field(0.25, gt=0)
    model: Literal["simple", "svi"] = "simple"


class StockPriceRequest(BaseModel):
    """Request to price using a predefined stock."""
    region: Literal["usa", "india", "europe", "worldwide"]
    ticker: str
    strike: Optional[float] = None  # defaults to ATM
    option_type: Literal["call", "put"] = "call"
    method: Literal["standard", "antithetic"] = "antithetic"
    n_simulations: int = Field(50000, gt=0, le=1000000)


class JumpDiffusionRequest(BaseModel):
    """Request for Merton Jump-Diffusion pricing."""
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    rate: float
    volatility: float = Field(..., gt=0)
    maturity: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"
    n_simulations: int = Field(50000, gt=0, le=1000000)
    jump_intensity: float = Field(1.0, ge=0, description="Avg jumps/year (λ)")
    jump_mean: float = Field(-0.05, description="Mean log-jump size (μ_J)")
    jump_vol: float = Field(0.10, gt=0, description="Jump volatility (σ_J)")


class MCGreeksRequest(BaseModel):
    """Request for Monte Carlo Greeks."""
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    rate: float
    volatility: float = Field(..., gt=0)
    maturity: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"
    n_simulations: int = Field(50000, gt=0, le=500000)
    method: Literal["standard", "antithetic"] = "antithetic"


class AmericanRequest(BaseModel):
    """Request for American option pricing (LSM)."""
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    rate: float
    volatility: float = Field(..., gt=0)
    maturity: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "put"
    n_simulations: int = Field(50000, gt=0, le=500000)
    n_steps: int = Field(100, gt=0, le=500)


class HestonRequest(BaseModel):
    """Request for Heston stochastic volatility pricing."""
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    rate: float
    volatility: float = Field(..., gt=0)
    maturity: float = Field(..., gt=0)
    option_type: Literal["call", "put"] = "call"
    n_simulations: int = Field(50000, gt=0, le=500000)
    n_steps: int = Field(200, gt=0, le=1000)
    kappa: float = Field(2.0, gt=0, description="Mean reversion speed")
    theta: float = Field(0.04, gt=0, description="Long-run variance")
    vol_of_vol: float = Field(0.3, gt=0, description="Vol of vol (σ_v)")
    rho: float = Field(-0.7, ge=-1, le=1, description="Spot-vol correlation")


class YieldCurveRequest(BaseModel):
    """Request to generate yield curve."""
    model: Literal["flat", "nelson-siegel", "inverted", "steep"] = "nelson-siegel"
    rate: float = Field(0.05)
    n_points: int = Field(50, gt=5, le=200)
    max_maturity: float = Field(30.0, gt=1)
