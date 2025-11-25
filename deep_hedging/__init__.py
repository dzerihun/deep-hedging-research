"""
Deep Hedging Research Package

A research framework for comparing neural network-based hedging strategies
against traditional Black-Scholes delta hedging under transaction costs.
"""

__version__ = "0.1.0"
__author__ = "Deep Hedging Research Team"

from deep_hedging.config import (
    ExperimentConfig,
    MarketConfig,
    TradingConfig,
    BacktestConfig,
    AgentConfig,
    RewardConfig,
    MetricsConfig,
    VisualizationConfig,
)
from deep_hedging.logger import get_logger, setup_logger, ExperimentLogger

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Config
    "ExperimentConfig",
    "MarketConfig",
    "TradingConfig",
    "BacktestConfig",
    "AgentConfig",
    "RewardConfig",
    "MetricsConfig",
    "VisualizationConfig",
    # Logging
    "get_logger",
    "setup_logger",
    "ExperimentLogger",
]
