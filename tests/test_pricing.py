"""
Tests for options pricing module.
"""

import pytest
import numpy as np

from deep_hedging.pricing.black_scholes import (
    black_scholes_price,
    black_scholes_delta,
    black_scholes_gamma,
    BlackScholesCalculator,
)
from deep_hedging.config import MarketConfig


class TestBlackScholes:
    """Test Black-Scholes pricing functions."""

    def test_call_price_atm(self):
        """Test call option price at-the-money."""
        price = black_scholes_price(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='call'
        )
        # ATM call should be positive
        assert price > 0
        # Should be less than spot price
        assert price < 100.0

    def test_put_price_atm(self):
        """Test put option price at-the-money."""
        price = black_scholes_price(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='put'
        )
        # ATM put should be positive
        assert price > 0
        # Should be less than strike price
        assert price < 100.0

    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

        call_price = black_scholes_price(S, K, T, r, sigma, 'call')
        put_price = black_scholes_price(S, K, T, r, sigma, 'put')

        # Put-Call Parity: C - P = S - K * exp(-rT)
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)

        assert np.isclose(lhs, rhs, atol=1e-10)

    def test_call_delta_range(self):
        """Test that call delta is in valid range [0, 1]."""
        delta = black_scholes_delta(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='call'
        )
        assert 0 <= delta <= 1

    def test_put_delta_range(self):
        """Test that put delta is in valid range [-1, 0]."""
        delta = black_scholes_delta(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='put'
        )
        assert -1 <= delta <= 0

    def test_gamma_positive(self):
        """Test that gamma is always positive."""
        gamma = black_scholes_gamma(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2
        )
        assert gamma > 0

    def test_gamma_same_for_call_and_put(self):
        """Test that gamma is the same for calls and puts."""
        # Gamma shouldn't depend on option type
        gamma1 = black_scholes_gamma(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
        gamma2 = black_scholes_gamma(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)
        assert gamma1 == gamma2

    def test_expiration_call_payoff(self):
        """Test call option payoff at expiration."""
        # ITM call
        price_itm = black_scholes_price(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.2, option_type='call')
        assert np.isclose(price_itm, 10.0)

        # OTM call
        price_otm = black_scholes_price(S=90.0, K=100.0, T=0.0, r=0.05, sigma=0.2, option_type='call')
        assert np.isclose(price_otm, 0.0)

    def test_expiration_put_payoff(self):
        """Test put option payoff at expiration."""
        # ITM put
        price_itm = black_scholes_price(S=90.0, K=100.0, T=0.0, r=0.05, sigma=0.2, option_type='put')
        assert np.isclose(price_itm, 10.0)

        # OTM put
        price_otm = black_scholes_price(S=110.0, K=100.0, T=0.0, r=0.05, sigma=0.2, option_type='put')
        assert np.isclose(price_otm, 0.0)


class TestBlackScholesCalculator:
    """Test BlackScholesCalculator class."""

    def test_calculator_initialization(self):
        """Test calculator initialization with config."""
        config = MarketConfig(
            spot_price=100.0,
            strike_price=100.0,
            time_to_maturity=1.0,
            volatility=0.2,
            risk_free_rate=0.05,
        )
        calc = BlackScholesCalculator(config)
        assert calc.config == config

    def test_calculator_price(self):
        """Test calculator price method."""
        config = MarketConfig(option_type='call')
        calc = BlackScholesCalculator(config)

        price = calc.price(S=100.0, T=1.0)
        expected = black_scholes_price(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, option_type='call'
        )
        assert np.isclose(price, expected)

    def test_calculator_greeks(self):
        """Test calculator greeks method."""
        config = MarketConfig(option_type='call')
        calc = BlackScholesCalculator(config)

        greeks = calc.greeks(S=100.0, T=1.0)

        # Verify all greeks are present
        assert hasattr(greeks, 'delta')
        assert hasattr(greeks, 'gamma')
        assert hasattr(greeks, 'vega')
        assert hasattr(greeks, 'theta')
        assert hasattr(greeks, 'rho')

        # Verify values are reasonable
        assert 0 <= greeks.delta <= 1  # Call delta
        assert greeks.gamma > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
