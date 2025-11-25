"""
Black-Scholes delta hedging strategy.

Implements traditional delta hedging by continuously (or discretely) rebalancing
the hedge portfolio to maintain a delta-neutral position.
"""

import numpy as np
from typing import Optional

from deep_hedging.config import MarketConfig, TradingConfig
from deep_hedging.strategies.base import (
    BaseHedgingStrategy,
    HedgeAction,
    MarketState
)
from deep_hedging.pricing.black_scholes import (
    black_scholes_delta,
    BlackScholesCalculator
)
from deep_hedging.logger import get_logger

logger = get_logger(__name__)


class BlackScholesDeltaHedge(BaseHedgingStrategy):
    """
    Delta hedging strategy using Black-Scholes delta.

    Maintains a delta-neutral position by holding Δ units of the underlying,
    where Δ is the Black-Scholes delta of the option.
    """

    def __init__(
        self,
        market_config: MarketConfig,
        trading_config: TradingConfig,
        use_true_delta: bool = True
    ):
        """
        Initialize delta hedging strategy.

        Args:
            market_config: Market configuration
            trading_config: Trading configuration
            use_true_delta: If True, use true market parameters for delta.
                          If False, could implement misspecified model.
        """
        super().__init__(
            market_config=market_config,
            trading_config=trading_config,
            name="BlackScholesDeltaHedge"
        )

        self.use_true_delta = use_true_delta
        self.bs_calculator = BlackScholesCalculator(market_config)

        logger.info(f"Initialized {self.name} (use_true_delta={use_true_delta})")

    def compute_hedge(self, state: MarketState) -> HedgeAction:
        """
        Compute delta hedge for current state.

        Args:
            state: Current market state

        Returns:
            HedgeAction with target position and trade details
        """
        # Calculate Black-Scholes delta
        delta = self.bs_calculator.delta(
            S=state.spot_price,
            T=state.time_to_maturity
        )

        # Target position is the delta (number of shares to hold)
        target_position = delta

        # Apply position limits
        target_position = self.enforce_position_limits(target_position)

        # Calculate trade size
        trade_size = target_position - state.current_position

        # Calculate transaction cost
        transaction_cost = self.calculate_transaction_cost(
            trade_size=trade_size,
            spot_price=state.spot_price
        )

        return HedgeAction(
            position=target_position,
            trade_size=trade_size,
            transaction_cost=transaction_cost,
            metadata={
                'delta': delta,
                'spot': state.spot_price,
                'time_to_maturity': state.time_to_maturity
            }
        )


class DeltaGammaHedge(BaseHedgingStrategy):
    """
    Delta-gamma hedging strategy.

    Hedges both delta and gamma by using the underlying and another option.
    For future implementation.
    """

    def __init__(
        self,
        market_config: MarketConfig,
        trading_config: TradingConfig
    ):
        super().__init__(
            market_config=market_config,
            trading_config=trading_config,
            name="DeltaGammaHedge"
        )
        self.bs_calculator = BlackScholesCalculator(market_config)

    def compute_hedge(self, state: MarketState) -> HedgeAction:
        """
        Compute delta-gamma hedge.

        For now, just use delta hedge. Full implementation would require
        a second option for gamma hedging.
        """
        # Calculate delta
        delta = self.bs_calculator.delta(
            S=state.spot_price,
            T=state.time_to_maturity
        )

        # Calculate gamma
        gamma = self.bs_calculator.gamma(
            S=state.spot_price,
            T=state.time_to_maturity
        )

        # For now, just delta hedge (gamma hedging requires another instrument)
        target_position = delta
        target_position = self.enforce_position_limits(target_position)

        trade_size = target_position - state.current_position
        transaction_cost = self.calculate_transaction_cost(
            trade_size=trade_size,
            spot_price=state.spot_price
        )

        return HedgeAction(
            position=target_position,
            trade_size=trade_size,
            transaction_cost=transaction_cost,
            metadata={
                'delta': delta,
                'gamma': gamma,
                'spot': state.spot_price,
                'time_to_maturity': state.time_to_maturity
            }
        )


class StaticHedge(BaseHedgingStrategy):
    """
    Static hedging strategy (buy-and-hold).

    Sets initial hedge and never rebalances. Useful as a baseline.
    """

    def __init__(
        self,
        market_config: MarketConfig,
        trading_config: TradingConfig,
        initial_hedge_ratio: float = 0.5
    ):
        """
        Initialize static hedge.

        Args:
            market_config: Market configuration
            trading_config: Trading configuration
            initial_hedge_ratio: Initial hedge ratio (e.g., 0.5 for ATM option)
        """
        super().__init__(
            market_config=market_config,
            trading_config=trading_config,
            name="StaticHedge"
        )
        self.initial_hedge_ratio = initial_hedge_ratio
        self.initial_hedge_set = False

    def compute_hedge(self, state: MarketState) -> HedgeAction:
        """
        Compute static hedge (only sets position once).

        Args:
            state: Current market state

        Returns:
            HedgeAction (only trades on first call)
        """
        if not self.initial_hedge_set:
            # First time: set initial hedge
            target_position = self.initial_hedge_ratio
            self.initial_hedge_set = True
        else:
            # Subsequent times: maintain current position (no rebalancing)
            target_position = state.current_position

        trade_size = target_position - state.current_position
        transaction_cost = self.calculate_transaction_cost(
            trade_size=trade_size,
            spot_price=state.spot_price
        )

        return HedgeAction(
            position=target_position,
            trade_size=trade_size,
            transaction_cost=transaction_cost,
            metadata={
                'is_initial': not self.initial_hedge_set,
                'hedge_ratio': self.initial_hedge_ratio
            }
        )


class NoHedge(BaseHedgingStrategy):
    """
    No hedging strategy (naked option position).

    Useful as a baseline to show the value of hedging.
    """

    def __init__(
        self,
        market_config: MarketConfig,
        trading_config: TradingConfig
    ):
        super().__init__(
            market_config=market_config,
            trading_config=trading_config,
            name="NoHedge"
        )

    def compute_hedge(self, state: MarketState) -> HedgeAction:
        """
        No hedging - always maintain zero position.

        Args:
            state: Current market state

        Returns:
            HedgeAction with zero position
        """
        target_position = 0.0
        trade_size = target_position - state.current_position
        transaction_cost = self.calculate_transaction_cost(
            trade_size=trade_size,
            spot_price=state.spot_price
        )

        return HedgeAction(
            position=target_position,
            trade_size=trade_size,
            transaction_cost=transaction_cost,
            metadata={'strategy': 'no_hedge'}
        )


if __name__ == "__main__":
    # Example usage
    from deep_hedging.config import MarketConfig, TradingConfig

    # Setup
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

    # Create strategy
    strategy = BlackScholesDeltaHedge(market_config, trading_config)

    # Example state
    state = MarketState(
        spot_price=105.0,
        time_to_maturity=0.5,
        current_position=0.5,
        option_price=8.0,
        pnl=0.0
    )

    # Compute hedge
    action = strategy.compute_hedge(state)

    print(f"Strategy: {strategy.name}")
    print(f"Current state: S={state.spot_price}, T={state.time_to_maturity}")
    print(f"Current position: {state.current_position:.4f}")
    print(f"Target position: {action.position:.4f}")
    print(f"Trade size: {action.trade_size:.4f}")
    print(f"Transaction cost: ${action.transaction_cost:.2f}")
    print(f"Delta: {action.metadata['delta']:.4f}")
