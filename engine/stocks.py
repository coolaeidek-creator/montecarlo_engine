"""
Curated stock data per region.

Each stock has: ticker, name, price, historical volatility, sector.
Prices reflect approximate market levels.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class StockData:
    ticker: str
    name: str
    price: float
    volatility: float
    sector: str


@dataclass
class RegionConfig:
    id: str
    name: str
    currency: str
    currency_symbol: str
    rate: float  # risk-free rate
    bank: str
    stocks: List[StockData]


REGIONS = {
    "usa": RegionConfig(
        id="usa", name="United States", currency="USD", currency_symbol="$",
        rate=0.053, bank="Federal Reserve",
        stocks=[
            StockData("AAPL",  "Apple Inc",        189, 0.24, "Technology"),
            StockData("MSFT",  "Microsoft Corp",   425, 0.22, "Technology"),
            StockData("GOOGL", "Alphabet Inc",     157, 0.26, "Technology"),
            StockData("AMZN",  "Amazon.com",       186, 0.30, "Consumer"),
            StockData("NVDA",  "NVIDIA Corp",      880, 0.48, "Semiconductor"),
            StockData("TSLA",  "Tesla Inc",        175, 0.55, "Automotive"),
            StockData("META",  "Meta Platforms",   505, 0.34, "Technology"),
            StockData("JPM",   "JPMorgan Chase",   198, 0.20, "Banking"),
            StockData("V",     "Visa Inc",         282, 0.19, "Finance"),
            StockData("SPY",   "S&P 500 ETF",      520, 0.14, "Index ETF"),
        ],
    ),
    "india": RegionConfig(
        id="india", name="India", currency="INR", currency_symbol="₹",
        rate=0.065, bank="Reserve Bank of India",
        stocks=[
            StockData("RELIANCE",  "Reliance Industries", 2950, 0.22, "Conglomerate"),
            StockData("TCS",       "Tata Consultancy",    3850, 0.20, "IT Services"),
            StockData("INFY",      "Infosys Ltd",         1520, 0.25, "IT Services"),
            StockData("HDFCBANK",  "HDFC Bank",           1640, 0.18, "Banking"),
            StockData("ICICIBANK", "ICICI Bank",          1100, 0.22, "Banking"),
            StockData("BHARTIARTL","Bharti Airtel",       1550, 0.22, "Telecom"),
            StockData("SBIN",      "State Bank of India",  780, 0.26, "Banking"),
            StockData("ITC",       "ITC Ltd",              440, 0.18, "FMCG"),
            StockData("TATAMOTORS","Tata Motors",          980, 0.35, "Automotive"),
            StockData("NIFTY50",   "Nifty 50 Index",    22500, 0.14, "Index"),
        ],
    ),
    "europe": RegionConfig(
        id="europe", name="Europe", currency="EUR", currency_symbol="€",
        rate=0.040, bank="European Central Bank",
        stocks=[
            StockData("SAP",     "SAP SE",          185, 0.24, "Technology"),
            StockData("ASML",    "ASML Holding",    900, 0.32, "Semiconductor"),
            StockData("LVMH",    "LVMH Moet",       780, 0.25, "Luxury"),
            StockData("NOVO",    "Novo Nordisk",    850, 0.30, "Pharma"),
            StockData("SIEMENS", "Siemens AG",      180, 0.22, "Industrial"),
            StockData("TTE",     "TotalEnergies",    62, 0.22, "Energy"),
            StockData("SHELL",   "Shell PLC",        32, 0.20, "Energy"),
            StockData("BMW",     "BMW Group",       105, 0.26, "Automotive"),
            StockData("AIR",     "Airbus SE",       155, 0.24, "Aerospace"),
            StockData("SX5E",    "Euro Stoxx 50",  5050, 0.15, "Index"),
        ],
    ),
    "worldwide": RegionConfig(
        id="worldwide", name="Worldwide", currency="USD", currency_symbol="$",
        rate=0.050, bank="Global Composite",
        stocks=[
            StockData("MSCIW",   "MSCI World ETF",    108, 0.16, "Global Index"),
            StockData("SPY",     "S&P 500 ETF",       520, 0.14, "US Index"),
            StockData("EEM",     "Emerging Mkts ETF",   42, 0.22, "EM Index"),
            StockData("NKY",     "Nikkei 225",       39500, 0.18, "Japan Index"),
            StockData("HSI",     "Hang Seng Index",  16800, 0.22, "HK Index"),
            StockData("UKX",     "FTSE 100",          7700, 0.14, "UK Index"),
            StockData("TSM",     "TSMC",               150, 0.30, "Semiconductor"),
            StockData("BABA",    "Alibaba Group",       73, 0.40, "Technology"),
            StockData("SAMSUNG", "Samsung Elec.",        72, 0.28, "Technology"),
            StockData("TOYOTA",  "Toyota Motor",       190, 0.22, "Automotive"),
        ],
    ),
}


def get_stock(region_id: str, ticker: str) -> StockData:
    """Look up a stock by region and ticker."""
    region = REGIONS.get(region_id)
    if not region:
        raise ValueError(f"Unknown region: {region_id}")
    for stock in region.stocks:
        if stock.ticker == ticker:
            return stock
    raise ValueError(f"Stock {ticker} not found in {region_id}")


def get_region(region_id: str) -> RegionConfig:
    """Get region config by ID."""
    region = REGIONS.get(region_id)
    if not region:
        raise ValueError(f"Unknown region: {region_id}")
    return region
