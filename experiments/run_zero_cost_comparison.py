#!/usr/bin/env python3
"""
Test Zero-Cost Performance: RL vs Delta Hedging

This tests whether RL can match delta hedging when there are NO transaction costs.
"""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from deep_hedging.config import MarketConfig, TradingConfig, BacktestConfig
from deep_hedging.strategies import BlackScholesDeltaHedge
from deep_hedging.backtesting import BacktestEngine, simulate_gbm_from_config
from deep_hedging.logger import get_logger

# Import the RL wrapper from the existing script
sys.path.insert(0, str(Path(__file__).parent))
from run_rl_comparison import RLHedgingStrategy

logger = get_logger(__name__)

def main():
    logger.info("="*60)
    logger.info("ZERO COST COMPARISON: RL vs Delta Hedging")
    logger.info("="*60)

    # Configuration with ZERO transaction costs
    market_config = MarketConfig(
        spot_price=100.0,
        volatility=0.2,
        risk_free_rate=0.05,
        dividend_yield=0.0,
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
    logger.info("Generating test paths...")
    price_paths = simulate_gbm_from_config(market_config, backtest_config)

    # Create backtesting engine
    engine = BacktestEngine(
        market_config=market_config,
        trading_config=trading_config,
        backtest_config=backtest_config
    )

    # Test Delta Hedging
    logger.info("")
    logger.info("="*60)
    logger.info("Running Delta Hedging Baseline...")
    logger.info("="*60)
    delta_strategy = BlackScholesDeltaHedge(market_config, trading_config, use_true_delta=True)
    delta_results = engine.run(delta_strategy, simulated_paths=price_paths)
    logger.info(delta_results.summary())

    # Test RL Agent
    logger.info("")
    logger.info("="*60)
    logger.info("Running RL Agent...")
    logger.info("="*60)
    model_path = Path("models/cost_0bps/best/best_model.zip")
    rl_strategy = RLHedgingStrategy(model_path, market_config, trading_config)
    rl_results = engine.run(rl_strategy, simulated_paths=price_paths)
    logger.info(rl_results.summary())

    # Comparison
    logger.info("")
    logger.info("="*60)
    logger.info("COMPARISON: Delta Hedge vs RL Agent (ZERO COST)")
    logger.info("="*60)

    delta_err = delta_results.mean_hedging_error
    rl_err = rl_results.mean_hedging_error
    improvement = ((delta_err - rl_err) / delta_err) * 100

    logger.info(f"Delta Hedging Error:  ${delta_err:.4f}")
    logger.info(f"RL Agent Hedging Error: ${rl_err:.4f}")
    logger.info(f"Improvement:          {improvement:+.1f}%")
    logger.info("")

    # Save results
    results = {
        "delta_hedge": {
            "mean_pnl": float(delta_results.mean_pnl),
            "std_pnl": float(delta_results.std_pnl),
            "hedging_error": float(delta_results.mean_hedging_error),
            "transaction_costs": float(delta_results.mean_transaction_costs)
        },
        "rl_agent": {
            "mean_pnl": float(rl_results.mean_pnl),
            "std_pnl": float(rl_results.std_pnl),
            "hedging_error": float(rl_results.mean_hedging_error),
            "transaction_costs": float(rl_results.mean_transaction_costs)
        },
        "improvement_pct": float(improvement)
    }

    output_dir = Path('results') / 'zero_cost_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_file}")

    # Success criteria
    if rl_err < delta_err * 2.0:  # Within 2x of delta
        logger.info("\n✓ SUCCESS: RL agent performs reasonably at zero cost!")
        return 0
    else:
        logger.info("\n✗ ISSUE: RL agent significantly underperforming")
        return 1

if __name__ == "__main__":
    sys.exit(main())
