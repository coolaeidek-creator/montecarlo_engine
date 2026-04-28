# Datasets Used in the Monte Carlo Engine

This project does **not** consume any external dataset (no CSV, Parquet, JSON
market data, no API feeds such as yfinance / pandas-datareader / Quandl / FRED /
Alpha Vantage / Polygon / IEX / Tiingo). Everything that plays the role of
"data" is either **hardcoded reference values** or **synthetically generated**
at runtime.

Confirmed by:
- Repo-wide search for data-source libraries (`yfinance`, `pandas_datareader`,
  `quandl`, `fred`, `alpha_vantage`, `stooq`, `tiingo`, `polygon`, `iex`,
  `read_csv`, `read_parquet`, `download(`) → **zero hits**.
- `requirements.txt` contains only `numpy`, `pandas`, `scipy`, `matplotlib`,
  `fastapi`, `pydantic`, `uvicorn` and their transitive deps — no market-data
  client.
- `find` for `*.csv`, `*.parquet`, `*.xlsx`, `*.h5`, `*.pkl`, `*.feather`,
  `*.db`, `*.sqlite` under the project tree → no data files.

## 1. Curated stock / region reference table

File: [engine/stocks.py](engine/stocks.py)

A static, hand-curated table of 40 instruments across 4 regions. Each entry is
a `StockData(ticker, name, price, volatility, sector)`. Prices reflect
approximate market levels at the time of authoring; volatilities are assumed
historical vols. Risk-free rates are also hardcoded per region.

### USA — USD, rate = 5.30% (Federal Reserve)
| Ticker | Name              | Price | Vol  | Sector        |
|--------|-------------------|-------|------|---------------|
| AAPL   | Apple Inc         | 189   | 0.24 | Technology    |
| MSFT   | Microsoft Corp    | 425   | 0.22 | Technology    |
| GOOGL  | Alphabet Inc      | 157   | 0.26 | Technology    |
| AMZN   | Amazon.com        | 186   | 0.30 | Consumer      |
| NVDA   | NVIDIA Corp       | 880   | 0.48 | Semiconductor |
| TSLA   | Tesla Inc         | 175   | 0.55 | Automotive    |
| META   | Meta Platforms    | 505   | 0.34 | Technology    |
| JPM    | JPMorgan Chase    | 198   | 0.20 | Banking       |
| V      | Visa Inc          | 282   | 0.19 | Finance       |
| SPY    | S&P 500 ETF       | 520   | 0.14 | Index ETF     |

### India — INR, rate = 6.50% (Reserve Bank of India)
| Ticker      | Name                 | Price | Vol  | Sector       |
|-------------|----------------------|-------|------|--------------|
| RELIANCE    | Reliance Industries  | 2950  | 0.22 | Conglomerate |
| TCS         | Tata Consultancy     | 3850  | 0.20 | IT Services  |
| INFY        | Infosys Ltd          | 1520  | 0.25 | IT Services  |
| HDFCBANK    | HDFC Bank            | 1640  | 0.18 | Banking      |
| ICICIBANK   | ICICI Bank           | 1100  | 0.22 | Banking      |
| BHARTIARTL  | Bharti Airtel        | 1550  | 0.22 | Telecom      |
| SBIN        | State Bank of India  | 780   | 0.26 | Banking      |
| ITC         | ITC Ltd              | 440   | 0.18 | FMCG         |
| TATAMOTORS  | Tata Motors          | 980   | 0.35 | Automotive   |
| NIFTY50     | Nifty 50 Index       | 22500 | 0.14 | Index        |

### Europe — EUR, rate = 4.00% (European Central Bank)
| Ticker  | Name           | Price | Vol  | Sector        |
|---------|----------------|-------|------|---------------|
| SAP     | SAP SE         | 185   | 0.24 | Technology    |
| ASML    | ASML Holding   | 900   | 0.32 | Semiconductor |
| LVMH    | LVMH Moet      | 780   | 0.25 | Luxury        |
| NOVO    | Novo Nordisk   | 850   | 0.30 | Pharma        |
| SIEMENS | Siemens AG     | 180   | 0.22 | Industrial    |
| TTE     | TotalEnergies  | 62    | 0.22 | Energy        |
| SHELL   | Shell PLC      | 32    | 0.20 | Energy        |
| BMW     | BMW Group      | 105   | 0.26 | Automotive    |
| AIR     | Airbus SE      | 155   | 0.24 | Aerospace     |
| SX5E    | Euro Stoxx 50  | 5050  | 0.15 | Index         |

### Worldwide — USD, rate = 5.00% (Global Composite)
| Ticker  | Name              | Price | Vol  | Sector        |
|---------|-------------------|-------|------|---------------|
| MSCIW   | MSCI World ETF    | 108   | 0.16 | Global Index  |
| SPY     | S&P 500 ETF       | 520   | 0.14 | US Index      |
| EEM     | Emerging Mkts ETF | 42    | 0.22 | EM Index      |
| NKY     | Nikkei 225        | 39500 | 0.18 | Japan Index   |
| HSI     | Hang Seng Index   | 16800 | 0.22 | HK Index      |
| UKX     | FTSE 100          | 7700  | 0.14 | UK Index      |
| TSM     | TSMC              | 150   | 0.30 | Semiconductor |
| BABA    | Alibaba Group     | 73    | 0.40 | Technology    |
| SAMSUNG | Samsung Elec.     | 72    | 0.28 | Technology    |
| TOYOTA  | Toyota Motor      | 190   | 0.22 | Automotive    |

## 2. Synthetic OHLC generator

File: [engine/historical_vol.py:208](engine/historical_vol.py#L208) —
`generate_synthetic_ohlc(...)`

Produces synthetic Open / High / Low / Close series used to exercise the
historical-vol estimators (Parkinson, Garman-Klass, Rogers-Satchell,
Yang-Zhang). No real OHLC data is loaded from disk or network.

## 3. Simulated Monte Carlo paths

All pricing modules ([engine/paths.py](engine/paths.py),
[engine/heston.py](engine/heston.py),
[engine/jump_diffusion.py](engine/jump_diffusion.py),
[engine/sabr.py](engine/sabr.py),
[engine/binomial.py](engine/binomial.py),
[engine/exotic.py](engine/exotic.py),
[engine/variance_swap.py](engine/variance_swap.py),
[engine/quanto.py](engine/quanto.py),
[engine/delta_hedge.py](engine/delta_hedge.py), …) generate paths from the
chosen model (GBM, Heston, Merton jump-diffusion, SABR, binomial tree, etc.)
using `numpy`'s RNG. Inputs (S0, vol, rate, etc.) come from the curated table
above or from the API request payload — never from a historical dataset.

## How to plug in real market data (if needed)

The engine has no data layer today. To feed real prices you would:

1. Add a client (e.g. `yfinance`, `pandas-datareader`, FRED, your broker API)
   to `requirements.txt`.
2. Create a new module under `engine/` (e.g. `engine/market_data.py`) that
   returns the same `StockData` shape used in [engine/stocks.py](engine/stocks.py).
3. Either replace the hardcoded `REGIONS` lookup or layer a cache on top so
   pricing modules pick up live S0 / historical vol instead of static values.
