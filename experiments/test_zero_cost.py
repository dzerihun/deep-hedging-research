"""Quick test of zero-cost RL agent vs delta hedging."""

import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deep_hedging.config import MarketConfig, TradingConfig, BacktestConfig
from deep_hedging.backtesting.engine import BacktestEngine
from deep_hedging.strategies.delta_hedge import BlackScholesDeltaHedge
from deep_hedging.agents.rl_agent import RLAgent
from deep_hedging.backtesting.simulation import simulate_gbm
from deep_hedging.logger import get_logger

logger = get_logger(__name__)

# Config with ZERO transaction costs
market_config = MarketConfig(
    spot_price=100.0,
    volatility=0.2,
    risk_free_rate=0.05,
    strike_price=100.0,
    time_to_maturity=1.0,
    option_type='call'
)

trading_config = TradingConfig(
    transaction_cost_pct=0.0,  # ZERO COST
    rebalance_frequency='daily'
)

backtest_config = BacktestConfig(
    num_paths=1000,
    time_steps=252,
    random_seed=42
)

# Generate test paths
logger.info("Generating test paths (zero transaction cost)...")
price_paths = simulate_gbm(
    S0=market_config.spot_price,
    mu=market_config.risk_free_rate - market_config.dividend_yield,
    sigma=market_config.volatility,
    T=market_config.time_to_maturity,
    n_steps=backtest_config.time_steps,
    n_paths=backtest_config.num_paths,
    random_seed=backtest_config.random_seed
)

# Create backtesting engine
engine = BacktestEngine(
    market_config=market_config,
    trading_config=trading_config,
    backtest_config=backtest_config
)

# Test Delta Hedging
logger.info("\n" + "="*60)
logger.info("Delta Hedging (Zero Cost)")
logger.info("="*60)
delta_strategy = BlackScholesDeltaHedge(market_config, use_true_delta=True)
delta_results = engine.run_backtest(delta_strategy, price_paths)
logger.info(f"\nHedging Error: ${delta_results['std_pnl']:.4f}")
logger.info(f"Mean P&L: ${delta_results['mean_pnl']:.4f}")
logger.info(f"Transaction Costs: ${delta_results['mean_transaction_costs']:.4f}")

# Test RL Agent
logger.info("\n" + "="*60)
logger.info("RL Agent (Zero Cost)")
logger.info("="*60)
model_path = "models/cost_0bps/best/best_model.zip"
rl_strategy = RLAgent(market_config, model_path=model_path)
rl_results = engine.run_backtest(rl_strategy, price_paths)
logger.info(f"\nHedging Error: ${rl_results['std_pnl']:.4f}")
logger.info(f"Mean P&L: ${rl_results['mean_pnl']:.4f}")
logger.info(f"Transaction Costs: ${rl_results['mean_transaction_costs']:.4f}")

# Comparison
logger.info("\n" + "="*60)
logger.info("COMPARISON")
logger.info("="*60)
improvement = ((delta_results['std_pnl'] - rl_results['std_pnl']) / delta_results['std_pnl']) * 100
logger.info(f"Delta Hedging Error: ${delta_results['std_pnl']:.4f}")
logger.info(f"RL Hedging Error: ${rl_results['std_pnl']:.4f}")
logger.info(f"Improvement: {improvement:+.1f}%")

if rl_results['std_pnl'] < delta_results['std_pnl'] * 1.5:
    logger.info("\n✓ SUCCESS: RL agent performs reasonably at zero cost!")
else:
    logger.info("\n✗ ISSUE: RL agent underperforming even at zero cost")
