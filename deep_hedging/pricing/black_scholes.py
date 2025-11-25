"""
Black-Scholes-Merton option pricing and Greeks calculation.

Provides functions for computing option prices and sensitivities (Greeks)
under the Black-Scholes-Merton framework.
"""

import numpy as np
from scipy.stats import norm
from typing import Literal, Union
from dataclasses import dataclass

from deep_hedging.config import MarketConfig
from deep_hedging.logger import get_logger

logger = get_logger(__name__)


def _d1(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Calculate d1 parameter for Black-Scholes formula."""
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Calculate d2 parameter for Black-Scholes formula."""
    return _d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'] = 'call',
    q: float = 0.0
) -> float:
    """
    Calculate Black-Scholes option price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
        q: Dividend yield (annual)

    Returns:
        Option price
    """
    if T <= 0:
        # At expiration
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(S, K, T, r, sigma, q)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    return price


def black_scholes_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'] = 'call',
    q: float = 0.0
) -> float:
    """
    Calculate Black-Scholes delta (∂V/∂S).

    Delta measures the rate of change of option price with respect to
    the underlying asset price.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
        q: Dividend yield (annual)

    Returns:
        Delta value
    """
    if T <= 0:
        # At expiration
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = _d1(S, K, T, r, sigma, q)

    if option_type == 'call':
        delta = np.exp(-q * T) * norm.cdf(d1)
    elif option_type == 'put':
        delta = -np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    return delta


def black_scholes_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Calculate Black-Scholes gamma (∂²V/∂S²).

    Gamma measures the rate of change of delta with respect to the
    underlying asset price. Same for calls and puts.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        q: Dividend yield (annual)

    Returns:
        Gamma value
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma, q)
    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    return gamma


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> float:
    """
    Calculate Black-Scholes vega (∂V/∂σ).

    Vega measures the sensitivity of option price to volatility changes.
    Same for calls and puts.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        q: Dividend yield (annual)

    Returns:
        Vega value (per 1% change in volatility)
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma, q)
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    return vega / 100  # Return per 1% change in volatility


def black_scholes_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'] = 'call',
    q: float = 0.0
) -> float:
    """
    Calculate Black-Scholes theta (∂V/∂t).

    Theta measures the rate of change of option price with respect to time.
    Typically negative, representing time decay.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
        q: Dividend yield (annual)

    Returns:
        Theta value (per day)
    """
    if T <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, sigma, q)
    d2 = _d2(S, K, T, r, sigma, q)

    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == 'call':
        term2 = q * S * np.exp(-q * T) * norm.cdf(d1)
        term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        theta = term1 + term2 + term3
    elif option_type == 'put':
        term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = term1 + term2 + term3
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    return theta / 365  # Return per day


def black_scholes_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put'] = 'call',
    q: float = 0.0
) -> float:
    """
    Calculate Black-Scholes rho (∂V/∂r).

    Rho measures the sensitivity of option price to interest rate changes.

    Args:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annual)
        sigma: Volatility (annual)
        option_type: 'call' or 'put'
        q: Dividend yield (annual)

    Returns:
        Rho value (per 1% change in interest rate)
    """
    if T <= 0:
        return 0.0

    d2 = _d2(S, K, T, r, sigma, q)

    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    return rho / 100  # Return per 1% change in interest rate


@dataclass
class OptionGreeks:
    """Container for option Greeks."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


class BlackScholesCalculator:
    """
    Black-Scholes calculator with caching for efficient computation.

    Useful for repeated calculations with the same market parameters.
    """

    def __init__(self, market_config: MarketConfig):
        """
        Initialize calculator with market configuration.

        Args:
            market_config: Market configuration dataclass
        """
        self.config = market_config
        logger.debug(f"Initialized BlackScholesCalculator with config: {market_config}")

    def price(self, S: float, T: float) -> float:
        """Calculate option price."""
        return black_scholes_price(
            S=S,
            K=self.config.strike_price,
            T=T,
            r=self.config.risk_free_rate,
            sigma=self.config.volatility,
            option_type=self.config.option_type,
            q=self.config.dividend_yield
        )

    def delta(self, S: float, T: float) -> float:
        """Calculate option delta."""
        return black_scholes_delta(
            S=S,
            K=self.config.strike_price,
            T=T,
            r=self.config.risk_free_rate,
            sigma=self.config.volatility,
            option_type=self.config.option_type,
            q=self.config.dividend_yield
        )

    def gamma(self, S: float, T: float) -> float:
        """Calculate option gamma."""
        return black_scholes_gamma(
            S=S,
            K=self.config.strike_price,
            T=T,
            r=self.config.risk_free_rate,
            sigma=self.config.volatility,
            q=self.config.dividend_yield
        )

    def vega(self, S: float, T: float) -> float:
        """Calculate option vega."""
        return black_scholes_vega(
            S=S,
            K=self.config.strike_price,
            T=T,
            r=self.config.risk_free_rate,
            sigma=self.config.volatility,
            q=self.config.dividend_yield
        )

    def theta(self, S: float, T: float) -> float:
        """Calculate option theta."""
        return black_scholes_theta(
            S=S,
            K=self.config.strike_price,
            T=T,
            r=self.config.risk_free_rate,
            sigma=self.config.volatility,
            option_type=self.config.option_type,
            q=self.config.dividend_yield
        )

    def rho(self, S: float, T: float) -> float:
        """Calculate option rho."""
        return black_scholes_rho(
            S=S,
            K=self.config.strike_price,
            T=T,
            r=self.config.risk_free_rate,
            sigma=self.config.volatility,
            option_type=self.config.option_type,
            q=self.config.dividend_yield
        )

    def greeks(self, S: float, T: float) -> OptionGreeks:
        """Calculate all Greeks at once."""
        return OptionGreeks(
            delta=self.delta(S, T),
            gamma=self.gamma(S, T),
            vega=self.vega(S, T),
            theta=self.theta(S, T),
            rho=self.rho(S, T)
        )


if __name__ == "__main__":
    # Example usage
    S0 = 100.0  # Current stock price
    K = 100.0   # Strike price
    T = 1.0     # 1 year to maturity
    r = 0.05    # 5% risk-free rate
    sigma = 0.2 # 20% volatility

    call_price = black_scholes_price(S0, K, T, r, sigma, 'call')
    put_price = black_scholes_price(S0, K, T, r, sigma, 'put')

    print(f"Call Price: ${call_price:.2f}")
    print(f"Put Price: ${put_price:.2f}")

    call_delta = black_scholes_delta(S0, K, T, r, sigma, 'call')
    put_delta = black_scholes_delta(S0, K, T, r, sigma, 'put')

    print(f"Call Delta: {call_delta:.4f}")
    print(f"Put Delta: {put_delta:.4f}")

    gamma = black_scholes_gamma(S0, K, T, r, sigma)
    print(f"Gamma: {gamma:.4f}")
