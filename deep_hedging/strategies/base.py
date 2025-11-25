"""
Base class for hedging strategies.

Defines the interface that all hedging strategies must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

from deep_hedging.config import MarketConfig, TradingConfig
from deep_hedging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HedgeAction:
    """
    Represents a hedging action (trade) to be executed.

    Attributes:
        position: Target position in the underlying (shares to hold)
        trade_size: Size of the trade (positive for buy, negative for sell)
        transaction_cost: Cost incurred for this trade
        metadata: Optional dictionary for storing additional information
    """
    position: float
    trade_size: float
    transaction_cost: float
    metadata: Optional[dict] = None


@dataclass
class MarketState:
    """
    Represents the current state of the market.

    Attributes:
        spot_price: Current price of the underlying asset
        time_to_maturity: Remaining time to option maturity (years)
        current_position: Current hedge position (shares held)
        option_price: Current option price
        pnl: Current P&L
    """
    spot_price: float
    time_to_maturity: float
    current_position: float
    option_price: float
    pnl: float


class BaseHedgingStrategy(ABC):
    """
    Abstract base class for hedging strategies.

    All hedging strategies must inherit from this class and implement
    the compute_hedge() method.
    """

    def __init__(
        self,
        market_config: MarketConfig,
        trading_config: TradingConfig,
        name: str = "BaseStrategy"
    ):
        """
        Initialize hedging strategy.

        Args:
            market_config: Market configuration
            trading_config: Trading configuration
            name: Strategy name
        """
        self.market_config = market_config
        self.trading_config = trading_config
        self.name = name

        logger.info(f"Initialized {self.name} with market config: {market_config}")

    @abstractmethod
    def compute_hedge(self, state: MarketState) -> HedgeAction:
        """
        Compute the hedging action for the current market state.

        Args:
            state: Current market state

        Returns:
            HedgeAction specifying the hedge position and trade details
        """
        pass

    def calculate_transaction_cost(
        self,
        trade_size: float,
        spot_price: float
    ) -> float:
        """
        Calculate transaction cost for a given trade.

        Args:
            trade_size: Size of the trade (positive for buy, negative for sell)
            spot_price: Current spot price

        Returns:
            Transaction cost
        """
        abs_trade_value = abs(trade_size * spot_price)

        if self.trading_config.transaction_cost_type == 'proportional':
            cost = abs_trade_value * self.trading_config.transaction_cost_pct
        elif self.trading_config.transaction_cost_type == 'fixed':
            cost = self.trading_config.fixed_cost_per_trade if abs(trade_size) > 1e-10 else 0.0
        elif self.trading_config.transaction_cost_type == 'none':
            cost = 0.0
        else:
            raise ValueError(f"Unknown transaction cost type: {self.trading_config.transaction_cost_type}")

        # Add slippage if configured
        cost += abs_trade_value * self.trading_config.slippage_pct

        return cost

    def should_rebalance(self, current_time: float, last_rebalance_time: float) -> bool:
        """
        Determine if portfolio should be rebalanced based on rebalancing frequency.

        Args:
            current_time: Current time (in years)
            last_rebalance_time: Time of last rebalance (in years)

        Returns:
            True if should rebalance, False otherwise
        """
        if self.trading_config.rebalance_frequency == 'continuous':
            return True

        time_since_rebalance = current_time - last_rebalance_time

        if self.trading_config.rebalance_frequency == 'daily':
            return time_since_rebalance >= 1 / 252  # Trading days in a year
        elif self.trading_config.rebalance_frequency == 'weekly':
            return time_since_rebalance >= 1 / 52
        elif self.trading_config.rebalance_frequency == 'monthly':
            return time_since_rebalance >= 1 / 12
        elif self.trading_config.rebalance_frequency == 'custom':
            interval_years = self.trading_config.rebalance_interval_days / 365
            return time_since_rebalance >= interval_years
        else:
            raise ValueError(f"Unknown rebalance frequency: {self.trading_config.rebalance_frequency}")

    def enforce_position_limits(self, target_position: float) -> float:
        """
        Enforce position limits on target position.

        Args:
            target_position: Desired hedge position

        Returns:
            Position after applying limits
        """
        # Check short selling constraint
        if not self.trading_config.allow_short_selling and target_position < 0:
            logger.warning(f"Short selling not allowed, clamping position {target_position:.2f} to 0")
            target_position = 0.0

        # Check maximum position size
        if self.trading_config.max_position_size is not None:
            max_pos = self.trading_config.max_position_size
            if abs(target_position) > max_pos:
                logger.warning(
                    f"Position {target_position:.2f} exceeds limit {max_pos:.2f}, clamping"
                )
                target_position = np.sign(target_position) * max_pos

        return target_position

    def __repr__(self) -> str:
        """String representation of the strategy."""
        return f"{self.name}(market={self.market_config.option_type} option, " \
               f"K={self.market_config.strike_price}, " \
               f"T={self.market_config.time_to_maturity})"


if __name__ == "__main__":
    # This would fail because BaseHedgingStrategy is abstract
    # But we can demonstrate the dataclasses
    state = MarketState(
        spot_price=100.0,
        time_to_maturity=1.0,
        current_position=0.5,
        option_price=10.0,
        pnl=100.0
    )

    action = HedgeAction(
        position=0.6,
        trade_size=0.1,
        transaction_cost=0.05,
        metadata={"reason": "delta rebalance"}
    )

    print(f"Market State: {state}")
    print(f"Hedge Action: {action}")
