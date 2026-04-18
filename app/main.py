"""
Monte Carlo Options Pricing Engine — Full Demo

Showcases all engine capabilities:
- Vanilla MC pricing (Standard & Antithetic) with BS validation
- Exotic options (Asian, Barrier, Lookback, Digital)
- Risk metrics (VaR, CVaR — Parametric & Monte Carlo)
- Implied volatility solver (Newton-Raphson)
- Volatility surface generation (Simple & SVI)
- Full Greeks computation
- 40 real stocks across 4 global regions
"""

import numpy as np
from engine.models import MarketEnvironment, OptionContract
from engine.pricer import price_option
from engine.greeks import compute_greeks_both
from engine.stocks import REGIONS, get_stock, get_region
from engine.exotic import price_asian, price_barrier, price_lookback, price_digital
from engine.risk import compute_var
from engine.implied_vol import implied_volatility
from engine.vol_surface import generate_vol_surface, print_surface


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
        spot=stock.price, rate=region.rate,
        volatility=stock.volatility, maturity=1.0,
    )
    strike = stock.price  # ATM

    call_c = OptionContract(strike=strike, option_type="call")
    put_c = OptionContract(strike=strike, option_type="put")

    std_call = price_option(market, call_c, n_sims, method="standard")
    std_put = price_option(market, put_c, n_sims, method="standard")
    ant_call = price_option(market, call_c, n_sims, method="antithetic")
    ant_put = price_option(market, put_c, n_sims, method="antithetic")

    greeks = compute_greeks_both(market, strike)

    df = np.exp(-market.rate * market.maturity)
    parity_lhs = ant_call["price"] - ant_put["price"]
    parity_rhs = stock.price - strike * df

    print(f"\n  {stock.ticker} — {stock.name}")
    print(f"  Spot: {cur}{stock.price:,.2f}  |  Vol: {stock.volatility*100:.0f}%  |  Sector: {stock.sector}")
    print(f"  Strike: {cur}{strike:,.2f} (ATM)  |  Rate: {region.rate*100:.1f}%  |  T: 1.0 yr")
    print()

    print(f"  {'Method':<22} {'Call':>12} {'Put':>12} {'Call SE':>12} {'Put SE':>12}")
    print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    print(f"  {'Standard MC':<22} {cur}{std_call['price']:>10,.4f} {cur}{std_put['price']:>10,.4f} {cur}{std_call['std_error']:>10,.6f} {cur}{std_put['std_error']:>10,.6f}")
    print(f"  {'Antithetic MC':<22} {cur}{ant_call['price']:>10,.4f} {cur}{ant_put['price']:>10,.4f} {cur}{ant_call['std_error']:>10,.6f} {cur}{ant_put['std_error']:>10,.6f}")
    print(f"  {'Black-Scholes':<22} {cur}{ant_call['bs_price']:>10,.4f} {cur}{ant_put['bs_price']:>10,.4f} {'(analytical)':>12} {'(exact)':>12}")

    reduction = (1 - ant_call["std_error"] / std_call["std_error"]) * 100 if std_call["std_error"] > 0 else 0
    print(f"\n  Variance Reduction: {reduction:+.1f}% (antithetic vs standard SE)")
    print(f"  MC vs BS Diff:      {cur}{ant_call['bs_diff']:.6f} (call)  |  {cur}{ant_put['bs_diff']:.6f} (put)")

    g = greeks["call"]
    print(f"\n  Greeks (Call):")
    print(f"    Δ Delta  = {g['delta']:+.4f}   (if stock moves {cur}1, call moves {cur}{abs(g['delta']):.2f})")
    print(f"    Γ Gamma  = {g['gamma']:.5f}    (delta acceleration)")
    print(f"    Θ Theta  = {cur}{g['theta']:.4f}/day  (daily time decay)")
    print(f"    ν Vega   = {cur}{g['vega']:.4f}/1%σ   (vol sensitivity)")
    print(f"    ρ Rho    = {cur}{g['rho']:.4f}/1%r    (rate sensitivity)")

    print(f"\n  Put-Call Parity:  C-P = {cur}{parity_lhs:,.4f}  |  S-Ke^(-rT) = {cur}{parity_rhs:,.4f}  |  Err = {cur}{abs(parity_lhs-parity_rhs):.6f}")

    return market, ant_call, ant_put


def demo_exotic(market, strike, cur):
    """Demo exotic options pricing for a given market."""
    section("EXOTIC OPTIONS PRICING")

    call_c = OptionContract(strike=strike, option_type="call")
    put_c = OptionContract(strike=strike, option_type="put")

    # Asian
    asian_call = price_asian(market, call_c, n_simulations=30000)
    asian_put = price_asian(market, put_c, n_simulations=30000)
    print(f"\n  ASIAN (Arithmetic Average)")
    print(f"    Call: {cur}{asian_call['price']:.4f}  (SE: {cur}{asian_call['std_error']:.5f})")
    print(f"    Put:  {cur}{asian_put['price']:.4f}  (SE: {cur}{asian_put['std_error']:.5f})")
    print(f"    → Cheaper than vanilla (averaging reduces volatility)")

    # Barrier
    barrier_level = strike * 0.85
    barrier_call = price_barrier(market, call_c, barrier=barrier_level,
                                  barrier_type="down-and-out", n_simulations=30000)
    print(f"\n  BARRIER (Down-and-Out, B={cur}{barrier_level:,.0f})")
    print(f"    Call: {cur}{barrier_call['price']:.4f}  (SE: {cur}{barrier_call['std_error']:.5f})")
    print(f"    Knock probability: {barrier_call['knock_probability']*100:.1f}%")
    print(f"    → Cheaper than vanilla (can be knocked out)")

    # Lookback
    lb_call = price_lookback(market, call_c, n_simulations=30000)
    lb_put = price_lookback(market, put_c, n_simulations=30000)
    print(f"\n  LOOKBACK (Floating Strike)")
    print(f"    Call: {cur}{lb_call['price']:.4f}  (SE: {cur}{lb_call['std_error']:.5f})  — buy at lowest")
    print(f"    Put:  {cur}{lb_put['price']:.4f}  (SE: {cur}{lb_put['std_error']:.5f})  — sell at highest")
    print(f"    → More expensive (hindsight value)")

    # Digital
    dig_call = price_digital(market, call_c, payout=100.0, n_simulations=30000)
    print(f"\n  DIGITAL (Cash-or-Nothing, Payout={cur}100)")
    print(f"    Call: {cur}{dig_call['price']:.4f}  (SE: {cur}{dig_call['std_error']:.5f})")
    print(f"    ITM Probability: {dig_call['itm_probability']*100:.1f}%")
    print(f"    → Fixed payout if S_T > K at expiry")


def demo_risk(market, cur):
    """Demo risk metrics for a given market."""
    section("RISK METRICS — VaR & CVaR")

    # Parametric
    param = compute_var(market, confidence=0.95, horizon_days=10, method="parametric")
    print(f"\n  Parametric VaR (95%, 10-day)")
    print(f"    VaR:  {cur}{param['var_dollar']:.2f}  ({param['var_pct']*100:.2f}%)")
    print(f"    CVaR: {cur}{param['cvar_dollar']:.2f}  ({param['cvar_pct']*100:.2f}%)")

    # Monte Carlo
    mc = compute_var(market, confidence=0.95, horizon_days=10,
                     n_simulations=50000, method="monte_carlo")
    print(f"\n  Monte Carlo VaR (95%, 10-day, 50K sims)")
    print(f"    VaR:  {cur}{mc['var_dollar']:.2f}  ({mc['var_pct']*100:.2f}%)")
    print(f"    CVaR: {cur}{mc['cvar_dollar']:.2f}  ({mc['cvar_pct']*100:.2f}%)")
    print(f"    Max Drawdown (avg): {mc['max_drawdown']*100:.2f}%")
    print(f"    P&L Skew: {mc['pnl_skew']:.3f}  |  Kurtosis: {mc['pnl_kurtosis']:.3f}")
    print(f"    Worst Case: -{cur}{mc['worst_case']:.2f}  |  Best Case: +{cur}{mc['best_case']:.2f}")

    # 99% comparison
    mc99 = compute_var(market, confidence=0.99, horizon_days=10,
                       n_simulations=50000, method="monte_carlo")
    print(f"\n  99% VaR: {cur}{mc99['var_dollar']:.2f}  (vs 95% VaR: {cur}{mc['var_dollar']:.2f})")


def demo_implied_vol(market, cur):
    """Demo implied volatility solver."""
    section("IMPLIED VOLATILITY SOLVER")

    from engine.analytical import bs_price
    # Generate BS prices at known vol, then recover it
    true_vol = market.volatility
    strikes = [market.spot * pct for pct in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]]

    print(f"\n  Recovering IV from BS prices (true σ = {true_vol*100:.0f}%)")
    print(f"  {'Strike':>10}  {'BS Price':>10}  {'Recovered IV':>12}  {'Error':>10}  {'Iters':>6}  {'Method':<15}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*6}  {'─'*15}")

    for K in strikes:
        contract = OptionContract(strike=K, option_type="call")
        price = bs_price(market, contract)
        result = implied_volatility(
            market_price=price, spot=market.spot, strike=K,
            rate=market.rate, maturity=market.maturity,
        )
        err = abs(result["iv"] - true_vol) * 100
        print(f"  {cur}{K:>8,.0f}  {cur}{price:>8,.4f}  {result['iv']*100:>10.4f}%  {err:>8.6f}%  {result['iterations']:>5}  {result['method']}")


def demo_vol_surface(market, cur):
    """Demo volatility surface generation."""
    section("VOLATILITY SURFACE")

    surface_data = generate_vol_surface(market.spot, market.rate, market.volatility)
    print(f"\n  Synthetic Vol Surface for S={cur}{market.spot:,.0f}, σ_ATM={market.volatility*100:.0f}%")
    print()
    print_surface(surface_data, cur)


def main():
    divider()
    print("  MONTE CARLO OPTIONS PRICING ENGINE v3.1")
    print("  Full Demo — Vanilla · Exotic · Risk · IV · Vol Surface")
    print("  4 Regions · 40 Stocks · 22 Engine Modules")
    divider()

    # ── 1) VANILLA PRICING ───────────────────────────────
    demo_stocks = [
        ("usa",       "AAPL"),
        ("usa",       "NVDA"),
        ("india",     "RELIANCE"),
        ("europe",    "ASML"),
        ("worldwide", "TSM"),
    ]

    for region_id, ticker in demo_stocks:
        region = get_region(region_id)
        section(f"{region.name} ({region.currency}) — {region.bank} Rate: {region.rate*100:.1f}%")
        market, _, _ = price_stock(region_id, ticker)

    # ── 2) EXOTIC OPTIONS (using AAPL) ───────────────────
    aapl = get_stock("usa", "AAPL")
    aapl_region = get_region("usa")
    aapl_market = MarketEnvironment(
        spot=aapl.price, rate=aapl_region.rate,
        volatility=aapl.volatility, maturity=1.0,
    )
    demo_exotic(aapl_market, aapl.price, "$")

    # ── 3) RISK METRICS (using AAPL) ─────────────────────
    demo_risk(aapl_market, "$")

    # ── 4) IMPLIED VOLATILITY ─────────────────────────────
    demo_implied_vol(aapl_market, "$")

    # ── 5) VOLATILITY SURFACE ─────────────────────────────
    demo_vol_surface(aapl_market, "$")

    # ── 6) STOCK CATALOG ─────────────────────────────────
    divider()
    print("\n  AVAILABLE STOCKS PER REGION")
    divider("─")

    for _, region in REGIONS.items():
        print(f"\n  {region.name} ({region.currency_symbol}{region.currency})  —  {region.bank} @ {region.rate*100:.1f}%")
        for s in region.stocks:
            print(f"    {s.ticker:<12} {s.name:<24} {region.currency_symbol}{s.price:>10,.2f}   Vol {s.volatility*100:4.0f}%   {s.sector}")

    # ── FOOTER ────────────────────────────────────────────
    divider()
    print("  Engine v3.1 Capabilities:")
    print("    · Standard MC · Antithetic Variates · Black-Scholes Validation")
    print("    · Exotic: Asian · Barrier · Lookback · Digital")
    print("    · Risk: VaR · CVaR (Parametric + MC) · Max Drawdown")
    print("    · IV Solver: Newton-Raphson + Bisection · Vol Surface (SVI)")
    print("    · Greeks: Delta · Gamma · Theta · Vega · Rho · Greeks Surface")
    print("    · Models: Heston SV · SABR · Merton Jump-Diffusion · Binomial CRR")
    print("    · American: Longstaff-Schwartz LSM · Binomial Tree")
    print("    · Quanto Options · Dividend Adjustments · Variance Swaps")
    print("    · Delta Hedging · Historical Vol · Yield Curve")
    print("    · API: 30+ FastAPI REST endpoints · 40 stocks · 4 regions")
    print("  Live:   montecarloengine.vercel.app")
    divider()


if __name__ == "__main__":
    main()
