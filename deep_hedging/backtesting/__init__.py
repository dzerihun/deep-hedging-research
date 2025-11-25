"""
Backtesting engine module.

Provides functionality for backtesting hedging strategies with
realistic transaction costs and market simulation.
"""

from deep_hedging.backtesting.simulation import (
    simulate_gbm,
    simulate_gbm_from_config,
    simulate_heston,
    SimulatedPaths,
    get_terminal_payoff
)
from deep_hedging.backtesting.engine import (
    BacktestEngine,
    BacktestResults,
    PathResult
)

__all__ = [
    "simulate_gbm",
    "simulate_gbm_from_config",
    "simulate_heston",
    "SimulatedPaths",
    "get_terminal_payoff",
    "BacktestEngine",
    "BacktestResults",
    "PathResult",
]
