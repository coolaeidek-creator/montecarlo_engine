"""
Monte Carlo Options Pricing Engine — Industry Demo

Prices real stocks across 4 regions (USA, India, Europe, Worldwide)
using Monte Carlo simulation with antithetic variates,
validated against Black-Scholes analytical prices.
"""

import numpy as np
from engine.models import MarketEnvironment, OptionContract
from engine.pricer import price_option

from engine.greeks import compute_greeks_both
from engine.stocks import REGIONS, get_stock, get_region


def divider(char="═", width=72):
    print(char * width)


def section(title):
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")


def price_stock(region_id: str, ticker: str, n_sims: int = 50000):
    """Price options on a specific stock with full analysis."""
    region = get_region(region_id)
    stock = get_stock(region_id, ticker)
    cur = region.currency_symbol

    market = MarketEnvironment(
        spot=stock.price,
        rate=region.rate,
        volatility=stock.volatility,
        maturity=1.0,
    )
    strike = stock.price  # ATM

    # Price with both methods
    call_c = OptionContract(strike=strike, option_type="call")
    put_c = OptionContract(strike=strike, option_type="put")

    std_call = price_option(market, call_c, n_sims, method="standard")
    std_put = price_option(market, put_c, n_sims, method="standard")
    ant_call = price_option(market, call_c, n_sims, method="antithetic")
    ant_put = price_option(market, put_c, n_sims, method="antithetic")

    # Greeks
    greeks = compute_greeks_both(market, strike)

    # Put-call parity
    df = np.exp(-market.rate * market.maturity)
    parity_lhs = ant_call["price"] - ant_put["price"]
    parity_rhs = stock.price - strike * df

    print(f"\n  {stock.ticker} — {stock.name}")
    print(f"  Spot: {cur}{stock.price:,.2f}  |  Vol: {stock.volatility*100:.0f}%  |  Sector: {stock.sector}")
    print(f"  Strike: {cur}{strike:,.2f} (ATM)  |  Rate: {region.rate*100:.1f}%  |  T: 1.0 yr")
    print()

    # MC Prices
    print(f"  {'Method':<22} {'Call':>12} {'Put':>12} {'Call SE':>12} {'Put SE':>12}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'Standard MC':<22} {cur}{std_call['price']:>10,.4f} {cur}{std_put['price']:>10,.4f} {cur}{std_call['std_error']:>10,.6f} {cur}{std_put['std_error']:>10,.6f}")
    print(f"  {'Antithetic MC':<22} {cur}{ant_call['price']:>10,.4f} {cur}{ant_put['price']:>10,.4f} {cur}{ant_call['std_error']:>10,.6f} {cur}{ant_put['std_error']:>10,.6f}")
    print(f"  {'Black-Scholes':<22} {cur}{ant_call['bs_price']:>10,.4f} {cur}{ant_put['bs_price']:>10,.4f} {'(analytical)':>12} {'(exact)':>12}")

    # Variance reduction
    reduction = (1 - ant_call["std_error"] / std_call["std_error"]) * 100 if std_call["std_error"] > 0 else 0
    print(f"\n  Variance Reduction: {reduction:+.1f}% (antithetic vs standard SE)")
    print(f"  MC vs BS Diff:      {cur}{ant_call['bs_diff']:.6f} (call)  |  {cur}{ant_put['bs_diff']:.6f} (put)")

    # Greeks
    g = greeks["call"]
    print(f"\n  Greeks (Call):")
    print(f"    Δ Delta  = {g['delta']:+.4f}   (if stock moves {cur}1, call moves {cur}{abs(g['delta']):.2f})")
    print(f"    Γ Gamma  = {g['gamma']:.5f}    (delta acceleration)")
    print(f"    Θ Theta  = {cur}{g['theta']:.4f}/day  (daily time decay)")
    print(f"    ν Vega   = {cur}{g['vega']:.4f}/1%σ   (vol sensitivity)")
    print(f"    ρ Rho    = {cur}{g['rho']:.4f}/1%r    (rate sensitivity)")

    # Parity
    print(f"\n  Put-Call Parity:  C-P = {cur}{parity_lhs:,.4f}  |  S-Ke^(-rT) = {cur}{parity_rhs:,.4f}  |  Err = {cur}{abs(parity_lhs-parity_rhs):.6f}")

    return ant_call, ant_put


def main():
    divider()
    print("  MONTE CARLO OPTIONS PRICING ENGINE")
    print("  Industry-Level Implementation — 4 Regions · 40 Stocks")
    divider()

    # Demo: one stock from each region
    demo_stocks = [
        ("usa",       "AAPL"),
        ("usa",       "NVDA"),
        ("india",     "RELIANCE"),
        ("india",     "TCS"),
        ("europe",    "ASML"),
        ("europe",    "SAP"),
        ("worldwide", "TSM"),
        ("worldwide", "BABA"),
    ]

    for region_id, ticker in demo_stocks:
        region = get_region(region_id)
        section(f"{region.name} ({region.currency}) — {region.bank} Rate: {region.rate*100:.1f}%")
        price_stock(region_id, ticker)

    # Summary
    divider()
    print("\n  AVAILABLE STOCKS PER REGION")
    divider("─")

    for _rid, region in REGIONS.items():
        print(f"\n  {region.name} ({region.currency_symbol}{region.currency})  —  {region.bank} @ {region.rate*100:.1f}%")
        for s in region.stocks:
            print(f"    {s.ticker:<12} {s.name:<24} {region.currency_symbol}{s.price:>10,.2f}   Vol {s.volatility*100:4.0f}%   {s.sector}")

    divider()
    print("  Engine: Standard MC · Antithetic Variates · Black-Scholes Validation")
    print("  Greeks: Delta · Gamma · Theta · Vega · Rho")
    print("  Live:   montecarloengine.vercel.app")
    divider()


if __name__ == "__main__":
    main()
