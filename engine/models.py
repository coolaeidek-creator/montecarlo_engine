from pydantic import BaseModel, Field # type: ignore
from typing import Literal


class MarketEnvironment(BaseModel):
    spot: float = Field(..., gt=0)
    rate: float
    volatility: float = Field(..., gt=0)
    maturity: float = Field(..., gt=0)


class OptionContract(BaseModel):
    strike: float = Field(..., gt=0)
    option_type: Literal["call", "put"]
