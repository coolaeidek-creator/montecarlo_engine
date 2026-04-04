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
