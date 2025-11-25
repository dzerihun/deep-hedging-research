"""
Monte Carlo simulation for stock price paths.

Implements various stochastic processes for simulating asset prices,
primarily Geometric Brownian Motion (GBM).
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from deep_hedging.config import MarketConfig, BacktestConfig
from deep_hedging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SimulatedPaths:
    """Container for simulated price paths and time grid."""

    spot_paths: np.ndarray  # Shape: (num_paths, time_steps + 1)
    time_grid: np.ndarray   # Shape: (time_steps + 1,)
    dt: float               # Time step size
    num_paths: int
    time_steps: int


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    num_paths: int,
    time_steps: int,
    random_seed: Optional[int] = None
) -> SimulatedPaths:
    """
    Simulate stock price paths using Geometric Brownian Motion.

    The GBM follows: dS_t = μ S_t dt + σ S_t dW_t

    Args:
        S0: Initial stock price
        mu: Drift (expected return)
        sigma: Volatility
        T: Time horizon (in years)
        num_paths: Number of paths to simulate
        time_steps: Number of time steps
        random_seed: Random seed for reproducibility

    Returns:
        SimulatedPaths object containing price paths and time grid
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Time grid
    dt = T / time_steps
    time_grid = np.linspace(0, T, time_steps + 1)

    # Generate random shocks
    dW = np.random.normal(0, np.sqrt(dt), size=(num_paths, time_steps))

    # Initialize price paths
    S = np.zeros((num_paths, time_steps + 1))
    S[:, 0] = S0

    # Simulate paths using exact solution of GBM
    # S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*dW)
    for t in range(time_steps):
        S[:, t + 1] = S[:, t] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * dW[:, t]
        )

    logger.info(
        f"Simulated {num_paths} GBM paths with S0={S0:.2f}, μ={mu:.4f}, "
        f"σ={sigma:.4f}, T={T:.2f}, steps={time_steps}"
    )

    return SimulatedPaths(
        spot_paths=S,
        time_grid=time_grid,
        dt=dt,
        num_paths=num_paths,
        time_steps=time_steps
    )


def simulate_gbm_from_config(
    market_config: MarketConfig,
    backtest_config: BacktestConfig
) -> SimulatedPaths:
    """
    Simulate GBM paths using configuration objects.

    Args:
        market_config: Market configuration
        backtest_config: Backtest configuration

    Returns:
        SimulatedPaths object
    """
    # Use risk-neutral drift (risk-free rate - dividend yield)
    mu = market_config.risk_free_rate - market_config.dividend_yield

    return simulate_gbm(
        S0=market_config.spot_price,
        mu=mu,
        sigma=market_config.volatility,
        T=market_config.time_to_maturity,
        num_paths=backtest_config.num_paths,
        time_steps=backtest_config.time_steps,
        random_seed=backtest_config.random_seed
    )


def simulate_heston(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    mu: float,
    T: float,
    num_paths: int,
    time_steps: int,
    random_seed: Optional[int] = None
) -> SimulatedPaths:
    """
    Simulate stock price paths using Heston stochastic volatility model.

    For future implementation when studying volatility effects.

    The Heston model:
        dS_t = μ S_t dt + √v_t S_t dW1_t
        dv_t = κ(θ - v_t)dt + σ_v √v_t dW2_t
        corr(dW1, dW2) = ρ

    Args:
        S0: Initial stock price
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma_v: Volatility of volatility
        rho: Correlation between price and volatility
        mu: Drift
        T: Time horizon
        num_paths: Number of paths
        time_steps: Number of time steps
        random_seed: Random seed

    Returns:
        SimulatedPaths object
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    dt = T / time_steps
    time_grid = np.linspace(0, T, time_steps + 1)

    # Initialize arrays
    S = np.zeros((num_paths, time_steps + 1))
    v = np.zeros((num_paths, time_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    # Generate correlated random variables
    for t in range(time_steps):
        # Generate independent standard normals
        Z1 = np.random.normal(0, 1, num_paths)
        Z2 = np.random.normal(0, 1, num_paths)

        # Create correlation
        dW1 = np.sqrt(dt) * Z1
        dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

        # Update variance (using Euler scheme with max to avoid negative variance)
        v[:, t + 1] = np.maximum(
            v[:, t] + kappa * (theta - v[:, t]) * dt + sigma_v * np.sqrt(v[:, t]) * dW2,
            0
        )

        # Update stock price
        S[:, t + 1] = S[:, t] * np.exp(
            (mu - 0.5 * v[:, t]) * dt + np.sqrt(v[:, t]) * dW1
        )

    logger.info(f"Simulated {num_paths} Heston paths")

    return SimulatedPaths(
        spot_paths=S,
        time_grid=time_grid,
        dt=dt,
        num_paths=num_paths,
        time_steps=time_steps
    )


def get_terminal_payoff(
    spot_prices: np.ndarray,
    strike: float,
    option_type: str
) -> np.ndarray:
    """
    Calculate terminal option payoff.

    Args:
        spot_prices: Terminal spot prices (can be 1D or 2D array)
        strike: Strike price
        option_type: 'call' or 'put'

    Returns:
        Option payoffs
    """
    if option_type == 'call':
        return np.maximum(spot_prices - strike, 0)
    elif option_type == 'put':
        return np.maximum(strike - spot_prices, 0)
    else:
        raise ValueError(f"Unknown option type: {option_type}")


if __name__ == "__main__":
    # Example: Simulate and visualize paths
    import matplotlib.pyplot as plt

    paths = simulate_gbm(
        S0=100.0,
        mu=0.05,
        sigma=0.2,
        T=1.0,
        num_paths=10,
        time_steps=252,
        random_seed=42
    )

    # Plot a few sample paths
    plt.figure(figsize=(12, 6))
    for i in range(min(10, paths.num_paths)):
        plt.plot(paths.time_grid, paths.spot_paths[i, :], alpha=0.7)
    plt.xlabel('Time (years)')
    plt.ylabel('Stock Price')
    plt.title('Sample GBM Paths')
    plt.grid(True, alpha=0.3)
    plt.show()

    # Print statistics
    terminal_prices = paths.spot_paths[:, -1]
    print(f"Terminal price statistics:")
    print(f"  Mean: {terminal_prices.mean():.2f}")
    print(f"  Std:  {terminal_prices.std():.2f}")
    print(f"  Min:  {terminal_prices.min():.2f}")
    print(f"  Max:  {terminal_prices.max():.2f}")
