#!/usr/bin/env python3
"""
Compare RL Agent vs Black-Scholes Delta Hedging

Loads a trained RL agent and compares its performance against
the baseline delta hedging strategy on the same test paths.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from deep_hedging.config import MarketConfig, TradingConfig, BacktestConfig
from deep_hedging.strategies import BlackScholesDeltaHedge
from deep_hedging.backtesting import BacktestEngine, simulate_gbm_from_config
from deep_hedging.agents.hedging_env import HedgingEnv
from deep_hedging.logger import get_logger, ExperimentLogger

logger = get_logger(__name__)


class RLHedgingStrategy:
    """
    Wrapper to use trained RL agent as a hedging strategy.

    Adapts the RL agent to work with the BacktestEngine framework.
    """

    def __init__(
        self,
        model_path: Path,
        market_config: MarketConfig,
        trading_config: TradingConfig
    ):
        """
        Initialize RL strategy wrapper.

        Args:
            model_path: Path to trained PPO model
            market_config: Market configuration
            trading_config: Trading configuration
        """
        self.model = PPO.load(model_path)
        self.market_config = market_config
        self.trading_config = trading_config
        self.name = "RLAgent"

        # Create environment for state calculation
        self.env = HedgingEnv(
            market_config=market_config,
            trading_config=trading_config,
            num_steps=252,
            random_seed=None
        )

        logger.info(f"Loaded RL agent from: {model_path}")

    def compute_hedge(self, state):
        """
        Compute hedge using RL agent.

        Args:
            state: Market state

        Returns:
            HedgeAction
        """
        from deep_hedging.strategies.base import HedgeAction
        from deep_hedging.pricing.black_scholes import black_scholes_delta

        # Create observation for RL agent
        spot_price = state.spot_price
        time_to_maturity = state.time_to_maturity

        # Calculate delta
        if time_to_maturity > 1e-6:
            delta = black_scholes_delta(
                S=spot_price,
                K=self.market_config.strike_price,
                T=time_to_maturity,
                r=self.market_config.risk_free_rate,
                sigma=self.market_config.volatility,
                option_type=self.market_config.option_type,
                q=self.market_config.dividend_yield
            )
        else:
            delta = 0.0

        # Normalized values
        normalized_spot = spot_price / self.market_config.spot_price
        moneyness = spot_price / self.market_config.strike_price
        time_remaining = time_to_maturity / self.market_config.time_to_maturity

        obs = np.array([
            delta,
            state.current_position,
            time_remaining,
            normalized_spot,
            moneyness
        ], dtype=np.float32)

        # Get action from RL agent
        action, _states = self.model.predict(obs, deterministic=True)
        target_position = float(np.clip(action[0], -1.5, 1.5))

        # Calculate trade
        trade_size = target_position - state.current_position

        # Calculate transaction cost
        abs_trade_value = abs(trade_size * spot_price)
        if self.trading_config.transaction_cost_type == 'proportional':
            transaction_cost = abs_trade_value * self.trading_config.transaction_cost_pct
        else:
            transaction_cost = 0.0

        return HedgeAction(
            position=target_position,
            trade_size=trade_size,
            transaction_cost=transaction_cost,
            metadata={'agent_action': action[0]}
        )

    def should_rebalance(self, current_time, last_rebalance_time):
        """Always rebalance (RL agent decides when to trade via action)."""
        return True


def run_comparison_experiment():
    """Run RL vs Delta Hedge comparison."""

    logger.info("="*60)
    logger.info("RL AGENT VS DELTA HEDGING COMPARISON")
    logger.info("="*60)

    # Configuration
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
        transaction_cost_pct=0.001,
        rebalance_frequency='daily'
    )

    backtest_config = BacktestConfig(
        num_paths=1000,
        time_steps=252,
        random_seed=42  # Same seed as baseline for fair comparison
    )

    # Create output directory
    output_dir = Path('results') / 'rl_comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup experiment logger
    exp_logger = ExperimentLogger(
        experiment_name='rl_comparison',
        output_dir=output_dir
    )

    exp_logger.log_config({
        'market': market_config.to_dict() if hasattr(market_config, 'to_dict') else vars(market_config),
        'trading': vars(trading_config),
        'backtest': vars(backtest_config)
    })

    # Generate price paths (same as baseline)
    logger.info("Generating test paths...")
    simulated_paths = simulate_gbm_from_config(market_config, backtest_config)

    # Create backtesting engine
    engine = BacktestEngine(market_config, trading_config, backtest_config)

    # Test 1: Delta Hedging (baseline)
    logger.info("\n" + "="*60)
    logger.info("Running Delta Hedging Baseline...")
    logger.info("="*60)

    delta_strategy = BlackScholesDeltaHedge(market_config, trading_config)
    delta_results = engine.run(delta_strategy, simulated_paths=simulated_paths)

    # Test 2: RL Agent
    logger.info("\n" + "="*60)
    logger.info("Running RL Agent...")
    logger.info("="*60)

    # Load trained model
    model_path = Path("models/best/best_model.zip")
    if not model_path.exists():
        model_path = Path("models/ppo_hedging_final.zip")

    if not model_path.exists():
        logger.error(f"No trained model found! Expected at: {model_path}")
        logger.error("Please train an agent first using: python deep_hedging/agents/train_agent.py")
        sys.exit(1)

    rl_strategy = RLHedgingStrategy(model_path, market_config, trading_config)
    rl_results = engine.run(rl_strategy, simulated_paths=simulated_paths)

    # Comparison
    logger.info("\n" + "="*60)
    logger.info("COMPARISON: Delta Hedge vs RL Agent")
    logger.info("="*60)
    logger.info(f"{'Metric':<30} {'Delta Hedge':>15} {'RL Agent':>15} {'Improvement':>15}")
    logger.info("-"*75)
    logger.info(f"{'Mean P&L':<30} ${delta_results.mean_pnl:>14.2f} ${rl_results.mean_pnl:>14.2f} {((rl_results.mean_pnl - delta_results.mean_pnl) / abs(delta_results.mean_pnl) * 100 if delta_results.mean_pnl != 0 else 0):>13.1f}%")
    logger.info(f"{'Hedging Error (Std P&L)':<30} ${delta_results.mean_hedging_error:>14.2f} ${rl_results.mean_hedging_error:>14.2f} {((delta_results.mean_hedging_error - rl_results.mean_hedging_error) / delta_results.mean_hedging_error * 100):>13.1f}%")
    logger.info(f"{'Mean Transaction Costs':<30} ${delta_results.mean_transaction_costs:>14.2f} ${rl_results.mean_transaction_costs:>14.2f} {((delta_results.mean_transaction_costs - rl_results.mean_transaction_costs) / delta_results.mean_transaction_costs * 100 if delta_results.mean_transaction_costs > 0 else 0):>13.1f}%")
    logger.info("="*75)

    # Save comparison
    import json
    comparison_data = {
        'delta_hedge': {
            'mean_pnl': delta_results.mean_pnl,
            'std_pnl': delta_results.std_pnl,
            'mean_hedging_error': delta_results.mean_hedging_error,
            'mean_transaction_costs': delta_results.mean_transaction_costs
        },
        'rl_agent': {
            'mean_pnl': rl_results.mean_pnl,
            'std_pnl': rl_results.std_pnl,
            'mean_hedging_error': rl_results.mean_hedging_error,
            'mean_transaction_costs': rl_results.mean_transaction_costs
        }
    }

    with open(output_dir / 'comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)

    # Generate comparison plot
    logger.info("\nGenerating comparison plots...")
    plot_comparison(delta_results, rl_results, output_dir)

    logger.info(f"\n✓ Comparison complete! Results saved to: {output_dir}")
    logger.info(f"✓ Check out: {output_dir / 'comparison.png'}")

    return delta_results, rl_results


def plot_comparison(delta_results, rl_results, output_dir):
    """Generate comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # P&L distributions
    axes[0, 0].hist(delta_results.all_pnls, bins=50, alpha=0.6, label='Delta Hedge', edgecolor='black')
    axes[0, 0].hist(rl_results.all_pnls, bins=50, alpha=0.6, label='RL Agent', edgecolor='black')
    axes[0, 0].axvline(delta_results.mean_pnl, color='blue', linestyle='--', linewidth=2, label=f'Delta Mean: ${delta_results.mean_pnl:.2f}')
    axes[0, 0].axvline(rl_results.mean_pnl, color='orange', linestyle='--', linewidth=2, label=f'RL Mean: ${rl_results.mean_pnl:.2f}')
    axes[0, 0].set_xlabel('Terminal P&L ($)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('P&L Distribution Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Hedging error comparison
    metrics = ['Hedging Error\n(Std P&L)', 'Mean\nTransaction Costs']
    delta_values = [delta_results.mean_hedging_error, delta_results.mean_transaction_costs]
    rl_values = [rl_results.mean_hedging_error, rl_results.mean_transaction_costs]

    x = np.arange(len(metrics))
    width = 0.35

    axes[0, 1].bar(x - width/2, delta_values, width, label='Delta Hedge', alpha=0.8)
    axes[0, 1].bar(x + width/2, rl_values, width, label='RL Agent', alpha=0.8)
    axes[0, 1].set_ylabel('Value ($)', fontsize=12)
    axes[0, 1].set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Transaction costs distribution
    axes[1, 0].hist(delta_results.all_transaction_costs, bins=50, alpha=0.6, label='Delta Hedge', edgecolor='black')
    axes[1, 0].hist(rl_results.all_transaction_costs, bins=50, alpha=0.6, label='RL Agent', edgecolor='black')
    axes[1, 0].set_xlabel('Total Transaction Costs ($)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Transaction Costs Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Summary statistics
    axes[1, 1].axis('off')
    summary_text = f"""
    COMPARISON SUMMARY
    {'='*40}

    Delta Hedging:
    • Mean P&L:          ${delta_results.mean_pnl:>8.2f}
    • Hedging Error:     ${delta_results.mean_hedging_error:>8.2f}
    • Transaction Costs: ${delta_results.mean_transaction_costs:>8.2f}

    RL Agent:
    • Mean P&L:          ${rl_results.mean_pnl:>8.2f}
    • Hedging Error:     ${rl_results.mean_hedging_error:>8.2f}
    • Transaction Costs: ${rl_results.mean_transaction_costs:>8.2f}

    Improvement:
    • Hedging Error: {((delta_results.mean_hedging_error - rl_results.mean_hedging_error) / delta_results.mean_hedging_error * 100):>6.1f}%
    • Trans. Costs:  {((delta_results.mean_transaction_costs - rl_results.mean_transaction_costs) / delta_results.mean_transaction_costs * 100 if delta_results.mean_transaction_costs > 0 else 0):>6.1f}%

    {'='*40}
    """

    axes[1, 1].text(0.5, 0.5, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=11, verticalalignment='center', horizontalalignment='center',
                   family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'comparison.png'}")
    plt.close()


if __name__ == "__main__":
    try:
        delta_results, rl_results = run_comparison_experiment()
        logger.info("\n" + "="*60)
        logger.info("SUCCESS! RL comparison completed.")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Comparison failed: {str(e)}", exc_info=True)
        sys.exit(1)
