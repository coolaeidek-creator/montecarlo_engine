"""
Historical Volatility Estimators.

Implements multiple methods for estimating realized volatility from
price data, each with different statistical properties:

- Close-to-close (standard)
- Parkinson (high-low range)
- Garman-Klass (OHLC)
- Yang-Zhang (drift-independent, most efficient)
- Exponentially Weighted Moving Average (EWMA)
"""

import numpy as np


def close_to_close_vol(prices: list, window: int = 20, annualize: int = 252) -> dict:
    """
    Standard close-to-close historical volatility.

    σ = √(annualize × Var(log returns))
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < window + 1:
        return {"error": f"Need at least {window + 1} prices, got {n}"}

    log_returns = np.log(prices[1:] / prices[:-1])

    # Rolling volatility
    rolling_vols = []
    for i in range(window, len(log_returns) + 1):
        w = log_returns[i - window:i]
        vol = np.std(w, ddof=1) * np.sqrt(annualize)
        rolling_vols.append(vol)

    # Current vol (last window)
    current_vol = rolling_vols[-1] if rolling_vols else 0.0

    # Full-sample vol
    full_vol = np.std(log_returns, ddof=1) * np.sqrt(annualize)

    return {
        "current_vol": current_vol,
        "full_sample_vol": full_vol,
        "rolling_vols": rolling_vols,
        "window": window,
        "n_observations": n,
        "method": "close-to-close",
    }


def parkinson_vol(highs: list, lows: list, annualize: int = 252) -> dict:
    """
    Parkinson (1980) high-low range estimator.

    5× more efficient than close-to-close for continuous processes.
    σ² = (1/4n·ln2) × Σ(ln(H/L))²
    """
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    n = len(highs)

    if n < 2:
        return {"error": "Need at least 2 observations"}

    log_hl = np.log(highs / lows)
    factor = 1 / (4 * n * np.log(2))
    var = factor * np.sum(log_hl ** 2) * annualize

    return {
        "volatility": np.sqrt(var),
        "variance": var,
        "n_observations": n,
        "efficiency_vs_cc": 5.2,
        "method": "parkinson",
    }


def garman_klass_vol(
    opens: list, highs: list, lows: list, closes: list,
    annualize: int = 252,
) -> dict:
    """
    Garman-Klass (1980) OHLC estimator.

    ~7.4× more efficient than close-to-close.
    Uses open, high, low, close data for maximum information extraction.
    """
    o = np.asarray(opens, dtype=float)
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(o)

    if n < 2:
        return {"error": "Need at least 2 observations"}

    log_hl = np.log(h / l)
    log_co = np.log(c / o)

    var = (
        0.5 * np.mean(log_hl ** 2)
        - (2 * np.log(2) - 1) * np.mean(log_co ** 2)
    ) * annualize

    var = max(var, 0)

    return {
        "volatility": np.sqrt(var),
        "variance": var,
        "n_observations": n,
        "efficiency_vs_cc": 7.4,
        "method": "garman-klass",
    }


def yang_zhang_vol(
    opens: list, highs: list, lows: list, closes: list,
    annualize: int = 252,
) -> dict:
    """
    Yang-Zhang (2000) estimator.

    Drift-independent, handles overnight jumps. Most efficient
    OHLC estimator (~14× close-to-close).
    """
    o = np.asarray(opens, dtype=float)
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    n = len(o)

    if n < 3:
        return {"error": "Need at least 3 observations"}

    # Overnight returns: log(O_t / C_{t-1})
    log_oc = np.log(o[1:] / c[:-1])
    # Open-to-close returns
    log_co = np.log(c[1:] / o[1:])
    # Rogers-Satchell component
    log_ho = np.log(h[1:] / o[1:])
    log_hc = np.log(h[1:] / c[1:])
    log_lo = np.log(l[1:] / o[1:])
    log_lc = np.log(l[1:] / c[1:])

    n2 = len(log_oc)

    var_overnight = np.var(log_oc, ddof=1)
    var_open_close = np.var(log_co, ddof=1)
    var_rs = np.mean(log_ho * log_hc + log_lo * log_lc)

    k = 0.34 / (1.34 + (n2 + 1) / (n2 - 1))
    var = (var_overnight + k * var_open_close + (1 - k) * var_rs) * annualize

    var = max(var, 0)

    return {
        "volatility": np.sqrt(var),
        "variance": var,
        "n_observations": n,
        "efficiency_vs_cc": 14.0,
        "method": "yang-zhang",
    }


def ewma_vol(prices: list, decay: float = 0.94, annualize: int = 252) -> dict:
    """
    Exponentially Weighted Moving Average volatility (RiskMetrics).

    σ²_t = λ·σ²_{t-1} + (1-λ)·r²_t

    Standard decay factor λ = 0.94 (RiskMetrics daily).
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)

    if n < 3:
        return {"error": "Need at least 3 prices"}

    log_returns = np.log(prices[1:] / prices[:-1])

    # EWMA variance
    var = log_returns[0] ** 2
    ewma_vols = [np.sqrt(var * annualize)]

    for i in range(1, len(log_returns)):
        var = decay * var + (1 - decay) * log_returns[i] ** 2
        ewma_vols.append(np.sqrt(var * annualize))

    return {
        "current_vol": ewma_vols[-1],
        "ewma_vols": ewma_vols,
        "decay_factor": decay,
        "half_life": np.log(2) / np.log(1 / decay),
        "n_observations": n,
        "method": "ewma",
    }


def generate_synthetic_ohlc(
    spot: float,
    volatility: float,
    n_days: int = 252,
    rate: float = 0.05,
) -> dict:
    """
    Generate synthetic OHLC data for testing vol estimators.

    Uses intraday GBM simulation to create realistic OHLC bars.
    """
    dt = 1 / 252
    intraday_steps = 78  # ~5-min bars in a 6.5hr trading day

    opens, highs, lows, closes = [], [], [], []
    price = spot

    for _ in range(n_days):
        o = price
        h, l = o, o

        for _ in range(intraday_steps):
            price *= np.exp(
                (rate - 0.5 * volatility ** 2) * dt / intraday_steps
                + volatility * np.sqrt(dt / intraday_steps) * np.random.randn()
            )
            h = max(h, price)
            l = min(l, price)

        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(price)

    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "n_days": n_days,
    }
