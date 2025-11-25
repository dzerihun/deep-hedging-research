"""
Backtesting engine for hedging strategies.

Simulates hedging strategies on Monte Carlo paths and computes performance metrics.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import time
from tqdm import tqdm

from deep_hedging.config import (
    ExperimentConfig,
    MarketConfig,
    TradingConfig,
    BacktestConfig
)
from deep_hedging.strategies.base import (
    BaseHedgingStrategy,
    MarketState
)
from deep_hedging.pricing.black_scholes import black_scholes_price
from deep_hedging.backtesting.simulation import (
    simulate_gbm_from_config,
    SimulatedPaths,
    get_terminal_payoff
)
from deep_hedging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PathResult:
    """Results for a single simulated path."""

    path_id: int
    terminal_pnl: float
    total_transaction_costs: float
    hedging_error: float
    num_rebalances: int
    final_position: float

    # Time series data
    pnl_history: np.ndarray = field(repr=False)
    position_history: np.ndarray = field(repr=False)
    spot_history: np.ndarray = field(repr=False)
    transaction_costs_history: np.ndarray = field(repr=False)


@dataclass
class BacktestResults:
    """Aggregate results from backtesting."""

    strategy_name: str
    num_paths: int
    time_steps: int

    # P&L statistics
    mean_pnl: float
    std_pnl: float
    median_pnl: float
    min_pnl: float
    max_pnl: float

    # Hedging error statistics
    mean_hedging_error: float
    std_hedging_error: float
    median_hedging_error: float
    max_hedging_error: float

    # Transaction costs
    mean_transaction_costs: float
    total_transaction_costs: float

    # Additional metrics
    mean_num_rebalances: float
    success_rate: float  # Fraction of paths with positive P&L

    # Raw data for further analysis
    all_pnls: np.ndarray = field(repr=False)
    all_hedging_errors: np.ndarray = field(repr=False)
    all_transaction_costs: np.ndarray = field(repr=False)
    path_results: List[PathResult] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        """Generate a summary string of results."""
        summary = f"\n{'='*60}\n"
        summary += f"Backtest Results: {self.strategy_name}\n"
        summary += f"{'='*60}\n"
        summary += f"Simulation: {self.num_paths} paths, {self.time_steps} steps\n"
        summary += f"\nP&L Statistics:\n"
        summary += f"  Mean:   ${self.mean_pnl:>10.2f}\n"
        summary += f"  Median: ${self.median_pnl:>10.2f}\n"
        summary += f"  Std:    ${self.std_pnl:>10.2f}\n"
        summary += f"  Min:    ${self.min_pnl:>10.2f}\n"
        summary += f"  Max:    ${self.max_pnl:>10.2f}\n"
        summary += f"\nHedging Error:\n"
        summary += f"  Mean:   ${self.mean_hedging_error:>10.2f}\n"
        summary += f"  Median: ${self.median_hedging_error:>10.2f}\n"
        summary += f"  Std:    ${self.std_hedging_error:>10.2f}\n"
        summary += f"  Max:    ${self.max_hedging_error:>10.2f}\n"
        summary += f"\nTransaction Costs:\n"
        summary += f"  Mean:   ${self.mean_transaction_costs:>10.2f}\n"
        summary += f"  Total:  ${self.total_transaction_costs:>10.2f}\n"
        summary += f"\nOther Metrics:\n"
        summary += f"  Avg Rebalances: {self.mean_num_rebalances:>6.1f}\n"
        summary += f"  Success Rate:   {self.success_rate*100:>6.1f}%\n"
        summary += f"{'='*60}\n"
        return summary


class BacktestEngine:
    """
    Engine for backtesting hedging strategies.

    Simulates price paths and runs hedging strategies, tracking P&L,
    transaction costs, and hedging errors.
    """

    def __init__(
        self,
        market_config: MarketConfig,
        trading_config: TradingConfig,
        backtest_config: BacktestConfig
    ):
        """
        Initialize backtesting engine.

        Args:
            market_config: Market configuration
            trading_config: Trading configuration
            backtest_config: Backtest configuration
        """
        self.market_config = market_config
        self.trading_config = trading_config
        self.backtest_config = backtest_config

        logger.info("Initialized BacktestEngine")
        logger.info(f"  Market: {market_config.option_type} option, "
                   f"K={market_config.strike_price}, S0={market_config.spot_price}")
        logger.info(f"  Backtest: {backtest_config.num_paths} paths, "
                   f"{backtest_config.time_steps} steps")

    def run(
        self,
        strategy: BaseHedgingStrategy,
        simulated_paths: Optional[SimulatedPaths] = None,
        show_progress: bool = True
    ) -> BacktestResults:
        """
        Run backtest for a given strategy.

        Args:
            strategy: Hedging strategy to test
            simulated_paths: Pre-generated paths (if None, will generate)
            show_progress: Show progress bar

        Returns:
            BacktestResults object with aggregated metrics
        """
        logger.info(f"Running backtest for strategy: {strategy.name}")
        start_time = time.time()

        # Generate paths if not provided
        if simulated_paths is None:
            logger.info("Generating price paths...")
            simulated_paths = simulate_gbm_from_config(
                self.market_config,
                self.backtest_config
            )

        # Run strategy on each path
        path_results = []
        iterator = range(simulated_paths.num_paths)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Backtesting {strategy.name}")

        for path_id in iterator:
            result = self._run_single_path(
                strategy=strategy,
                spot_path=simulated_paths.spot_paths[path_id, :],
                time_grid=simulated_paths.time_grid,
                path_id=path_id
            )
            path_results.append(result)

        # Aggregate results
        results = self._aggregate_results(
            strategy_name=strategy.name,
            path_results=path_results,
            time_steps=simulated_paths.time_steps
        )

        elapsed = time.time() - start_time
        logger.info(f"Backtest completed in {elapsed:.2f}s")
        logger.info(results.summary())

        return results

    def _run_single_path(
        self,
        strategy: BaseHedgingStrategy,
        spot_path: np.ndarray,
        time_grid: np.ndarray,
        path_id: int
    ) -> PathResult:
        """
        Run strategy on a single price path.

        Args:
            strategy: Hedging strategy
            spot_path: Simulated spot prices
            time_grid: Time points
            path_id: Path identifier

        Returns:
            PathResult for this path
        """
        num_steps = len(time_grid) - 1
        T = self.market_config.time_to_maturity

        # Initialize tracking arrays
        position_history = np.zeros(num_steps + 1)
        pnl_history = np.zeros(num_steps + 1)
        transaction_costs_history = np.zeros(num_steps + 1)

        # Initialize state
        current_position = 0.0
        cumulative_pnl = 0.0
        cumulative_costs = 0.0
        num_rebalances = 0
        last_rebalance_time = 0.0

        # Initial option price (we sold this option)
        initial_option_price = black_scholes_price(
            S=spot_path[0],
            K=self.market_config.strike_price,
            T=T,
            r=self.market_config.risk_free_rate,
            sigma=self.market_config.volatility,
            option_type=self.market_config.option_type,
            q=self.market_config.dividend_yield
        )

        # Start with option premium (we sold the option)
        cumulative_pnl = initial_option_price

        # Step through time
        for t in range(num_steps + 1):
            current_time = time_grid[t]
            time_to_maturity = T - current_time
            spot_price = spot_path[t]

            # Current option price
            if time_to_maturity > 0:
                option_price = black_scholes_price(
                    S=spot_price,
                    K=self.market_config.strike_price,
                    T=time_to_maturity,
                    r=self.market_config.risk_free_rate,
                    sigma=self.market_config.volatility,
                    option_type=self.market_config.option_type,
                    q=self.market_config.dividend_yield
                )
            else:
                # At expiration
                option_price = get_terminal_payoff(
                    spot_prices=spot_price,
                    strike=self.market_config.strike_price,
                    option_type=self.market_config.option_type
                )

            # Create market state
            state = MarketState(
                spot_price=spot_price,
                time_to_maturity=max(time_to_maturity, 0),
                current_position=current_position,
                option_price=option_price,
                pnl=cumulative_pnl
            )

            # Check if we should rebalance
            should_rebalance = (
                t == 0 or  # Always rebalance at start
                t == num_steps or  # Always close at end
                strategy.should_rebalance(current_time, last_rebalance_time)
            )

            if should_rebalance and time_to_maturity >= 0:
                # Compute hedge action
                action = strategy.compute_hedge(state)

                # Execute trade
                cumulative_costs += action.transaction_cost

                # Update position
                current_position = action.position

                if abs(action.trade_size) > 1e-10:
                    num_rebalances += 1

                last_rebalance_time = current_time
                transaction_costs_history[t] = action.transaction_cost

            # Update P&L
            # P&L = option premium - option value + hedge position value - transaction costs
            hedge_value = current_position * spot_price
            cumulative_pnl = initial_option_price - option_price + hedge_value - cumulative_costs

            # Store history
            position_history[t] = current_position
            pnl_history[t] = cumulative_pnl

        # Terminal statistics
        terminal_pnl = pnl_history[-1]
        terminal_spot = spot_path[-1]
        terminal_option_payoff = get_terminal_payoff(
            spot_prices=terminal_spot,
            strike=self.market_config.strike_price,
            option_type=self.market_config.option_type
        )

        # Hedging error = |option payoff - hedged portfolio value|
        # We sold the option and owe the payoff; our hedge should match it
        final_hedge_value = current_position * terminal_spot
        hedging_error = abs(terminal_option_payoff - final_hedge_value)

        return PathResult(
            path_id=path_id,
            terminal_pnl=terminal_pnl,
            total_transaction_costs=cumulative_costs,
            hedging_error=hedging_error,
            num_rebalances=num_rebalances,
            final_position=current_position,
            pnl_history=pnl_history,
            position_history=position_history,
            spot_history=spot_path,
            transaction_costs_history=transaction_costs_history
        )

    def _aggregate_results(
        self,
        strategy_name: str,
        path_results: List[PathResult],
        time_steps: int
    ) -> BacktestResults:
        """
        Aggregate results from multiple paths.

        Args:
            strategy_name: Name of strategy
            path_results: List of PathResult objects
            time_steps: Number of time steps

        Returns:
            BacktestResults with aggregate statistics
        """
        num_paths = len(path_results)

        # Extract arrays
        all_pnls = np.array([r.terminal_pnl for r in path_results])
        all_hedging_errors = np.array([r.hedging_error for r in path_results])
        all_transaction_costs = np.array([r.total_transaction_costs for r in path_results])
        all_num_rebalances = np.array([r.num_rebalances for r in path_results])

        # Calculate statistics
        return BacktestResults(
            strategy_name=strategy_name,
            num_paths=num_paths,
            time_steps=time_steps,
            # P&L stats
            mean_pnl=float(np.mean(all_pnls)),
            std_pnl=float(np.std(all_pnls)),
            median_pnl=float(np.median(all_pnls)),
            min_pnl=float(np.min(all_pnls)),
            max_pnl=float(np.max(all_pnls)),
            # Hedging error stats
            mean_hedging_error=float(np.mean(all_hedging_errors)),
            std_hedging_error=float(np.std(all_hedging_errors)),
            median_hedging_error=float(np.median(all_hedging_errors)),
            max_hedging_error=float(np.max(all_hedging_errors)),
            # Transaction costs
            mean_transaction_costs=float(np.mean(all_transaction_costs)),
            total_transaction_costs=float(np.sum(all_transaction_costs)),
            # Other metrics
            mean_num_rebalances=float(np.mean(all_num_rebalances)),
            success_rate=float(np.mean(all_pnls > 0)),
            # Raw data
            all_pnls=all_pnls,
            all_hedging_errors=all_hedging_errors,
            all_transaction_costs=all_transaction_costs,
            path_results=path_results
        )


if __name__ == "__main__":
    # Example usage
    from deep_hedging.config import MarketConfig, TradingConfig, BacktestConfig
    from deep_hedging.strategies.delta_hedge import BlackScholesDeltaHedge

    # Configuration
    market_config = MarketConfig(
        spot_price=100.0,
        volatility=0.2,
        risk_free_rate=0.05,
        strike_price=100.0,
        time_to_maturity=1.0,
        option_type='call'
    )

    trading_config = TradingConfig(
        transaction_cost_pct=0.001,
        rebalance_frequency='daily'
    )

    backtest_config = BacktestConfig(
        num_paths=100,
        time_steps=252,
        random_seed=42
    )

    # Create strategy and engine
    strategy = BlackScholesDeltaHedge(market_config, trading_config)
    engine = BacktestEngine(market_config, trading_config, backtest_config)

    # Run backtest
    results = engine.run(strategy)

    # Print summary
    print(results.summary())
