from engine.models import MarketEnvironment, OptionContract
from engine.pricer import price_option


def main():
    # Define market environment
    market = MarketEnvironment(
        spot=100,
        rate=0.05,
        volatility=0.2,
        maturity=1.0,
    )

    # Price a call option (strike = 100)
    call_contract = OptionContract(
        strike=100,
        option_type="call",
    )

    print("=" * 60)
    print("MONTE CARLO OPTIONS PRICING")
    print("=" * 60)
    print(f"\nMarket Parameters:")
    print(f"  Spot Price: ${market.spot:.2f}")
    print(f"  Risk-free Rate: {market.rate*100:.2f}%")
    print(f"  Volatility: {market.volatility*100:.2f}%")
    print(f"  Time to Maturity: {market.maturity} year(s)")

    # Price the call
    call_result = price_option(market, call_contract, n_simulations=50000)
    print(f"\nCall Option (K={call_contract.strike}):")
    print(f"  Price: ${call_result['price']:.4f}")
    print(f"  Std Error: ${call_result['std_error']:.6f}")
    print(f"  95% CI: [${call_result['confidence_interval'][0]:.4f}, ${call_result['confidence_interval'][1]:.4f}]")

    # Price a put option (strike = 100)
    put_contract = OptionContract(
        strike=100,
        option_type="put",
    )

    put_result = price_option(market, put_contract, n_simulations=50000)
    print(f"\nPut Option (K={put_contract.strike}):")
    print(f"  Price: ${put_result['price']:.4f}")
    print(f"  Std Error: ${put_result['std_error']:.6f}")
    print(f"  95% CI: [${put_result['confidence_interval'][0]:.4f}, ${put_result['confidence_interval'][1]:.4f}]")

    # Verify put-call parity (approximately)
    import numpy as np
    discount_factor = np.exp(-market.rate * market.maturity)
    parity_lhs = call_result['price'] - put_result['price']
    parity_rhs = market.spot - call_contract.strike * discount_factor
    print(f"\nPut-Call Parity Check:")
    print(f"  C - P = ${parity_lhs:.4f}")
    print(f"  S - K*e^(-rT) = ${parity_rhs:.4f}")
    print(f"  Difference: ${abs(parity_lhs - parity_rhs):.6f}")


if __name__ == "__main__":
    main()
