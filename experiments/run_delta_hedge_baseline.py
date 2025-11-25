#!/usr/bin/env python3
"""
Delta Hedging Baseline Experiment

Run this to get your first results! This script:
1. Simulates stock price paths using GBM
2. Runs Black-Scholes delta hedging strategy
3. Computes P&L and hedging errors
4. Generates plots showing results

Usage:
    python experiments/run_delta_hedge_baseline.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deep_hedging.config import MarketConfig, TradingConfig, BacktestConfig
from deep_hedging.strategies import BlackScholesDeltaHedge, NoHedge
from deep_hedging.backtesting import BacktestEngine, simulate_gbm_from_config
from deep_hedging.logger import get_logger, ExperimentLogger

logger = get_logger(__name__)


def plot_results(results, output_dir: Path):
    """Generate plots from backtest results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. P&L Distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram of P&L
    axes[0, 0].hist(results.all_pnls, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(results.mean_pnl, color='red', linestyle='--',
                       linewidth=2, label=f'Mean: ${results.mean_pnl:.2f}')
    axes[0, 0].axvline(results.median_pnl, color='orange', linestyle='--',
                       linewidth=2, label=f'Median: ${results.median_pnl:.2f}')
    axes[0, 0].set_xlabel('Terminal P&L ($)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title(f'{results.strategy_name}: P&L Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Hedging Error Distribution
    axes[0, 1].hist(results.all_hedging_errors, bins=50, alpha=0.7,
                    edgecolor='black', color='orange')
    axes[0, 1].axvline(results.mean_hedging_error, color='red', linestyle='--',
                       linewidth=2, label=f'Mean: ${results.mean_hedging_error:.2f}')
    axes[0, 1].set_xlabel('Hedging Error ($)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Hedging Error Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Transaction Costs Distribution
    axes[1, 0].hist(results.all_transaction_costs, bins=50, alpha=0.7,
                    edgecolor='black', color='green')
    axes[1, 0].axvline(results.mean_transaction_costs, color='red', linestyle='--',
                       linewidth=2, label=f'Mean: ${results.mean_transaction_costs:.2f}')
    axes[1, 0].set_xlabel('Total Transaction Costs ($)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Transaction Costs Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # P&L vs Transaction Costs scatter
    axes[1, 1].scatter(results.all_transaction_costs, results.all_pnls,
                       alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Transaction Costs ($)', fontsize=12)
    axes[1, 1].set_ylabel('P&L ($)', fontsize=12)
    axes[1, 1].set_title('P&L vs Transaction Costs', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'pnl_distribution.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'pnl_distribution.png'}")
    plt.close()

    # 2. Sample Path Visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot a few sample paths
    num_sample_paths = min(5, len(results.path_results))
    for i in range(num_sample_paths):
        path_result = results.path_results[i]

        # P&L evolution
        time_grid = np.linspace(0, 1, len(path_result.pnl_history))
        axes[0].plot(time_grid, path_result.pnl_history, alpha=0.7,
                    label=f'Path {i+1}')

        # Position evolution
        axes[1].plot(time_grid, path_result.position_history, alpha=0.7,
                    label=f'Path {i+1}')

    axes[0].set_xlabel('Time (years)', fontsize=12)
    axes[0].set_ylabel('P&L ($)', fontsize=12)
    axes[0].set_title(f'{results.strategy_name}: P&L Evolution (Sample Paths)',
                     fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)

    axes[1].set_xlabel('Time (years)', fontsize=12)
    axes[1].set_ylabel('Hedge Position (shares)', fontsize=12)
    axes[1].set_title('Hedge Position Evolution (Sample Paths)',
                     fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sample_paths.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'sample_paths.png'}")
    plt.close()

    # 3. Summary Statistics Box
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')

    summary_text = f"""
    {results.strategy_name}
    {'='*50}

    Simulation Parameters
    • Number of Paths:     {results.num_paths:,}
    • Time Steps:          {results.time_steps}

    P&L Statistics
    • Mean P&L:            ${results.mean_pnl:>10.2f}
    • Median P&L:          ${results.median_pnl:>10.2f}
    • Std Dev:             ${results.std_pnl:>10.2f}
    • Min P&L:             ${results.min_pnl:>10.2f}
    • Max P&L:             ${results.max_pnl:>10.2f}

    Hedging Performance
    • Mean Hedging Error:  ${results.mean_hedging_error:>10.2f}
    • Median Hedging Error:${results.median_hedging_error:>10.2f}
    • Std Hedging Error:   ${results.std_hedging_error:>10.2f}
    • Max Hedging Error:   ${results.max_hedging_error:>10.2f}

    Transaction Costs
    • Mean per Path:       ${results.mean_transaction_costs:>10.2f}
    • Total (All Paths):   ${results.total_transaction_costs:>10.2f}

    Other Metrics
    • Avg Rebalances:      {results.mean_num_rebalances:>10.1f}
    • Success Rate:        {results.success_rate*100:>9.1f}%

    {'='*50}
    """

    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='center', horizontalalignment='center',
           family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(output_dir / 'summary_stats.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved: {output_dir / 'summary_stats.png'}")
    plt.close()


def run_baseline_experiment():
    """Run the baseline delta hedging experiment."""

    logger.info("="*60)
    logger.info("DELTA HEDGING BASELINE EXPERIMENT")
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
        transaction_cost_pct=0.001,  # 10 bps
        rebalance_frequency='daily'
    )

    backtest_config = BacktestConfig(
        num_paths=1000,
        time_steps=252,  # Daily for 1 year
        random_seed=42
    )

    # Create output directory
    output_dir = Path('results') / 'baseline_delta_hedge'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup experiment logger
    exp_logger = ExperimentLogger(
        experiment_name='baseline_delta_hedge',
        output_dir=output_dir
    )

    # Log configuration
    exp_logger.log_config({
        'market': {
            'spot_price': market_config.spot_price,
            'volatility': market_config.volatility,
            'risk_free_rate': market_config.risk_free_rate,
            'strike': market_config.strike_price,
            'maturity': market_config.time_to_maturity,
            'option_type': market_config.option_type
        },
        'trading': {
            'transaction_cost_pct': trading_config.transaction_cost_pct,
            'rebalance_frequency': trading_config.rebalance_frequency
        },
        'backtest': {
            'num_paths': backtest_config.num_paths,
            'time_steps': backtest_config.time_steps,
            'random_seed': backtest_config.random_seed
        }
    })

    # Generate price paths (once, to use for all strategies)
    logger.info("Generating stock price paths...")
    simulated_paths = simulate_gbm_from_config(market_config, backtest_config)

    # Create backtesting engine
    engine = BacktestEngine(market_config, trading_config, backtest_config)

    # Test 1: Delta Hedging
    logger.info("\n" + "="*60)
    logger.info("Running Delta Hedging Strategy...")
    logger.info("="*60)

    delta_strategy = BlackScholesDeltaHedge(market_config, trading_config)
    delta_results = engine.run(delta_strategy, simulated_paths=simulated_paths)

    # Log results
    exp_logger.log_result({
        'strategy': delta_results.strategy_name,
        'mean_pnl': delta_results.mean_pnl,
        'std_pnl': delta_results.std_pnl,
        'mean_hedging_error': delta_results.mean_hedging_error,
        'mean_transaction_costs': delta_results.mean_transaction_costs,
        'success_rate': delta_results.success_rate
    })

    # Generate plots
    logger.info("\nGenerating plots...")
    plot_results(delta_results, output_dir)

    # Test 2: No Hedge Baseline (for comparison)
    logger.info("\n" + "="*60)
    logger.info("Running No Hedge Baseline (for comparison)...")
    logger.info("="*60)

    no_hedge_strategy = NoHedge(market_config, trading_config)
    no_hedge_results = engine.run(no_hedge_strategy, simulated_paths=simulated_paths)

    # Comparison
    logger.info("\n" + "="*60)
    logger.info("COMPARISON: Delta Hedge vs No Hedge")
    logger.info("="*60)
    logger.info(f"{'Metric':<30} {'Delta Hedge':>15} {'No Hedge':>15}")
    logger.info("-"*60)
    logger.info(f"{'Mean P&L':<30} ${delta_results.mean_pnl:>14.2f} ${no_hedge_results.mean_pnl:>14.2f}")
    logger.info(f"{'Std P&L':<30} ${delta_results.std_pnl:>14.2f} ${no_hedge_results.std_pnl:>14.2f}")
    logger.info(f"{'Mean Hedging Error':<30} ${delta_results.mean_hedging_error:>14.2f} ${no_hedge_results.mean_hedging_error:>14.2f}")
    logger.info(f"{'Mean Transaction Costs':<30} ${delta_results.mean_transaction_costs:>14.2f} ${no_hedge_results.mean_transaction_costs:>14.2f}")
    logger.info("="*60)

    # Save comparison data
    import json
    comparison_data = {
        'delta_hedge': {
            'mean_pnl': delta_results.mean_pnl,
            'std_pnl': delta_results.std_pnl,
            'mean_hedging_error': delta_results.mean_hedging_error,
            'mean_transaction_costs': delta_results.mean_transaction_costs
        },
        'no_hedge': {
            'mean_pnl': no_hedge_results.mean_pnl,
            'std_pnl': no_hedge_results.std_pnl,
            'mean_hedging_error': no_hedge_results.mean_hedging_error,
            'mean_transaction_costs': no_hedge_results.mean_transaction_costs
        }
    }

    with open(output_dir / 'comparison.json', 'w') as f:
        json.dump(comparison_data, f, indent=2)

    logger.info(f"\n✓ Experiment complete! Results saved to: {output_dir}")
    logger.info(f"✓ Check out the plots:")
    logger.info(f"  - {output_dir / 'pnl_distribution.png'}")
    logger.info(f"  - {output_dir / 'sample_paths.png'}")
    logger.info(f"  - {output_dir / 'summary_stats.png'}")

    return delta_results, no_hedge_results


if __name__ == "__main__":
    try:
        delta_results, no_hedge_results = run_baseline_experiment()
        logger.info("\n" + "="*60)
        logger.info("SUCCESS! Baseline experiment completed.")
        logger.info("="*60)
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}", exc_info=True)
        sys.exit(1)
