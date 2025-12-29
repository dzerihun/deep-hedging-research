"""
Gymnasium environment for deep hedging with options.

Implements a custom environment where an RL agent learns to hedge
an option position by trading the underlying asset under transaction costs.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

from deep_hedging.config import MarketConfig, TradingConfig
from deep_hedging.pricing.black_scholes import (
    black_scholes_price,
    black_scholes_delta,
    BlackScholesCalculator
)
from deep_hedging.backtesting.simulation import get_terminal_payoff
from deep_hedging.logger import get_logger

logger = get_logger(__name__)


class HedgingEnv(gym.Env):
    """
    Gymnasium environment for learning option hedging strategies.

    State Space:
        - option_delta: Black-Scholes delta [0, 1] for calls, [-1, 0] for puts
        - current_position: Current hedge position [-1.5, 1.5]
        - time_to_maturity: Normalized time remaining [0, 1]
        - spot_price: Normalized spot price (S/S0) [0.5, 2.0]
        - moneyness: S/K [0.5, 2.0]

    Action Space:
        - target_hedge_ratio: Continuous action in [-1.5, 1.5]
          Determines target hedge position as a fraction

    Reward:
        - Per-step: -0.01 * transaction_cost (light penalty for trading)
        - Terminal: -10.0 * hedging_error^2 (moderate penalty for poor hedge at expiration)
        - Ratio: Hedging error is 1,000x more important than transaction costs

    Episode:
        - One complete option lifetime (e.g., 252 days)
        - Resets with new random path
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_config: MarketConfig,
        trading_config: TradingConfig,
        num_steps: int = 252,
        random_seed: Optional[int] = None
    ):
        """
        Initialize hedging environment.

        Args:
            market_config: Market configuration
            trading_config: Trading configuration
            num_steps: Number of time steps per episode
            random_seed: Random seed for reproducibility
        """
        super().__init__()

        self.market_config = market_config
        self.trading_config = trading_config
        self.num_steps = num_steps
        self.dt = market_config.time_to_maturity / num_steps

        # Random number generator
        self.rng = np.random.default_rng(random_seed)

        # Black-Scholes calculator
        self.bs_calculator = BlackScholesCalculator(market_config)

        # State space: [delta, position, time_remaining, normalized_spot, moneyness]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.5, 0.0, 0.5, 0.5]),
            high=np.array([1.0, 1.5, 1.0, 2.0, 2.0]),
            dtype=np.float32
        )

        # Action space: target hedge ratio [-1.5, 1.5]
        self.action_space = spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(1,),
            dtype=np.float32
        )

        # Episode state
        self.current_step = 0
        self.spot_path = None
        self.time_grid = None
        self.current_position = 0.0
        self.cash_account = 0.0
        self.cumulative_costs = 0.0
        self.initial_option_price = 0.0

        logger.info(f"Initialized HedgingEnv with {num_steps} steps per episode")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to start a new episode.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reset episode state
        self.current_step = 0
        self.current_position = 0.0
        self.cumulative_costs = 0.0

        # Generate new price path using GBM
        self._generate_price_path()

        # Initial option price (we sell the option)
        S0 = self.spot_path[0]
        T = self.market_config.time_to_maturity
        self.initial_option_price = black_scholes_price(
            S=S0,
            K=self.market_config.strike_price,
            T=T,
            r=self.market_config.risk_free_rate,
            sigma=self.market_config.volatility,
            option_type=self.market_config.option_type,
            q=self.market_config.dividend_yield
        )

        # Cash account starts with option premium
        self.cash_account = self.initial_option_price

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step.

        Args:
            action: Target hedge ratio

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Extract action (target hedge ratio)
        target_ratio = float(np.clip(action[0], -1.5, 1.5))

        # Current market state
        spot_price = self.spot_path[self.current_step]
        time_to_maturity = self.time_grid[self.current_step]

        # Execute trade
        trade_size = target_ratio - self.current_position
        transaction_cost = self._calculate_transaction_cost(trade_size, spot_price)

        # Update portfolio (self-financing)
        self.cash_account += -trade_size * spot_price - transaction_cost
        self.cumulative_costs += transaction_cost
        self.current_position = target_ratio

        # Reward: per-step transaction cost penalty (reduced weight)
        step_reward = -0.01 * transaction_cost

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.num_steps
        truncated = False

        if terminated:
            # Terminal reward: penalize squared hedging error (moderate weight)
            terminal_pnl = self._calculate_terminal_pnl()
            hedging_error = abs(terminal_pnl)  # Distance from zero P&L
            terminal_reward = -10.0 * hedging_error ** 2  # Moderate penalty for hedging error

            reward = step_reward + terminal_reward
        else:
            reward = step_reward

        # Get observation and info
        obs = self._get_observation() if not terminated else self._get_terminal_observation()
        info = self._get_info()
        info['terminal_pnl'] = self._calculate_terminal_pnl() if terminated else 0.0

        return obs, reward, terminated, truncated, info

    def _generate_price_path(self):
        """Generate a new GBM price path for this episode."""
        S0 = self.market_config.spot_price
        mu = self.market_config.risk_free_rate - self.market_config.dividend_yield
        sigma = self.market_config.volatility
        T = self.market_config.time_to_maturity

        # Time grid
        self.time_grid = np.linspace(T, 0, self.num_steps + 1)  # Countdown to expiration

        # Generate random shocks
        dW = self.rng.normal(0, np.sqrt(self.dt), size=self.num_steps)

        # Simulate GBM path
        S = np.zeros(self.num_steps + 1)
        S[0] = S0

        for t in range(self.num_steps):
            S[t + 1] = S[t] * np.exp((mu - 0.5 * sigma ** 2) * self.dt + sigma * dW[t])

        self.spot_path = S

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        spot_price = self.spot_path[self.current_step]
        time_to_maturity = self.time_grid[self.current_step]

        # Calculate delta
        if time_to_maturity > 1e-6:
            delta = self.bs_calculator.delta(S=spot_price, T=time_to_maturity)
        else:
            delta = 0.0

        # Normalized spot price (S/S0)
        normalized_spot = spot_price / self.market_config.spot_price

        # Moneyness (S/K)
        moneyness = spot_price / self.market_config.strike_price

        # Normalized time remaining [0, 1]
        time_remaining = time_to_maturity / self.market_config.time_to_maturity

        obs = np.array([
            delta,
            self.current_position,
            time_remaining,
            normalized_spot,
            moneyness
        ], dtype=np.float32)

        return obs

    def _get_terminal_observation(self) -> np.ndarray:
        """Get observation at terminal step (all zeros for padding)."""
        return np.zeros(5, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        return {
            'step': self.current_step,
            'spot_price': self.spot_path[self.current_step] if self.current_step < len(self.spot_path) else self.spot_path[-1],
            'position': self.current_position,
            'cash': self.cash_account,
            'cumulative_costs': self.cumulative_costs
        }

    def _calculate_transaction_cost(self, trade_size: float, spot_price: float) -> float:
        """Calculate transaction cost for a trade."""
        abs_trade_value = abs(trade_size * spot_price)

        if self.trading_config.transaction_cost_type == 'proportional':
            cost = abs_trade_value * self.trading_config.transaction_cost_pct
        elif self.trading_config.transaction_cost_type == 'fixed':
            cost = self.trading_config.fixed_cost_per_trade if abs(trade_size) > 1e-10 else 0.0
        elif self.trading_config.transaction_cost_type == 'none':
            cost = 0.0
        else:
            cost = abs_trade_value * self.trading_config.transaction_cost_pct

        # Add slippage
        cost += abs_trade_value * self.trading_config.slippage_pct

        return cost

    def _calculate_terminal_pnl(self) -> float:
        """Calculate terminal P&L."""
        terminal_spot = self.spot_path[-1]
        terminal_payoff = get_terminal_payoff(
            spot_prices=terminal_spot,
            strike=self.market_config.strike_price,
            option_type=self.market_config.option_type
        )

        # Portfolio value at expiration
        portfolio_value = self.cash_account + self.current_position * terminal_spot

        # P&L = portfolio_value - option_payoff
        pnl = portfolio_value - terminal_payoff

        return pnl


if __name__ == "__main__":
    # Test environment
    from deep_hedging.config import MarketConfig, TradingConfig

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

    env = HedgingEnv(market_config, trading_config, num_steps=252, random_seed=42)

    # Test episode
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    total_reward = 0
    for step in range(252):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            print(f"Episode finished after {step + 1} steps")
            print(f"Total reward: {total_reward:.4f}")
            print(f"Terminal P&L: {info['terminal_pnl']:.4f}")
            break

    print("Environment test passed!")
