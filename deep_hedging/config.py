"""
Configuration module using dataclasses for type-safe, clean configuration management.

This module defines all configuration dataclasses used throughout the deep hedging project,
including market parameters, trading settings, agent configurations, and experiment settings.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any
from pathlib import Path
import yaml
import json


@dataclass
class MarketConfig:
    """Market parameters and option specifications."""

    # Underlying asset parameters
    spot_price: float = 100.0
    volatility: float = 0.2  # Annual volatility (sigma)
    risk_free_rate: float = 0.05  # Annual risk-free rate
    dividend_yield: float = 0.0  # Annual dividend yield

    # Option specifications
    strike_price: float = 100.0
    time_to_maturity: float = 1.0  # In years
    option_type: Literal['call', 'put'] = 'call'

    # Market dynamics
    price_process: Literal['gbm', 'heston', 'jump_diffusion'] = 'gbm'

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.spot_price > 0, "Spot price must be positive"
        assert self.volatility > 0, "Volatility must be positive"
        assert self.strike_price > 0, "Strike price must be positive"
        assert self.time_to_maturity > 0, "Time to maturity must be positive"
        assert 0 <= self.dividend_yield < 1, "Dividend yield must be in [0, 1)"


@dataclass
class TradingConfig:
    """Trading and transaction cost parameters."""

    # Transaction costs
    transaction_cost_pct: float = 0.001  # 0.1% per trade (10 bps)
    transaction_cost_type: Literal['proportional', 'fixed', 'none'] = 'proportional'
    fixed_cost_per_trade: float = 0.0  # Fixed cost if using 'fixed' type

    # Rebalancing
    rebalance_frequency: Literal['continuous', 'daily', 'weekly', 'monthly', 'custom'] = 'daily'
    rebalance_interval_days: Optional[float] = None  # For 'custom' frequency

    # Position limits
    max_position_size: Optional[float] = None  # Maximum hedge position
    allow_short_selling: bool = True

    # Slippage
    slippage_pct: float = 0.0  # Additional slippage cost

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.transaction_cost_pct >= 0, "Transaction cost must be non-negative"
        assert self.slippage_pct >= 0, "Slippage must be non-negative"
        if self.rebalance_frequency == 'custom':
            assert self.rebalance_interval_days is not None, \
                "Must specify rebalance_interval_days for custom frequency"


@dataclass
class BacktestConfig:
    """Backtesting engine configuration."""

    # Simulation parameters
    num_paths: int = 1000  # Number of Monte Carlo paths
    time_steps: int = 252  # Number of time steps (e.g., daily for 1 year)
    random_seed: Optional[int] = 42  # For reproducibility

    # Execution
    parallel: bool = True
    num_workers: int = -1  # -1 means use all available cores

    # Data
    use_historical_data: bool = False
    historical_data_path: Optional[Path] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.num_paths > 0, "Number of paths must be positive"
        assert self.time_steps > 0, "Number of time steps must be positive"
        if self.use_historical_data:
            assert self.historical_data_path is not None, \
                "Must specify historical_data_path when use_historical_data=True"


@dataclass
class AgentConfig:
    """Deep RL agent configuration."""

    # Algorithm
    algorithm: Literal['ppo', 'a2c', 'sac', 'td3', 'dqn'] = 'ppo'

    # Network architecture
    policy_type: Literal['MlpPolicy', 'CnnPolicy'] = 'MlpPolicy'
    hidden_layers: tuple = (256, 256)  # Network architecture
    activation_fn: Literal['relu', 'tanh', 'elu'] = 'relu'

    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation

    # Training duration
    total_timesteps: int = 100_000

    # Exploration
    ent_coef: float = 0.01  # Entropy coefficient for exploration

    # Saving and logging
    save_frequency: int = 10_000  # Save model every N steps
    tensorboard_log: Optional[Path] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert 0 <= self.gamma <= 1, "Gamma must be in [0, 1]"
        assert self.total_timesteps > 0, "Total timesteps must be positive"


@dataclass
class RewardConfig:
    """Reward function configuration for RL agent."""

    # Reward components
    hedging_error_weight: float = 1.0  # Weight for hedging error minimization
    transaction_cost_weight: float = 1.0  # Weight for transaction cost penalty
    risk_penalty_weight: float = 0.1  # Weight for risk (e.g., VaR) penalty

    # Reward shaping
    normalize_rewards: bool = True
    reward_scale: float = 1.0

    # Risk measures
    use_var_penalty: bool = False
    var_alpha: float = 0.05  # VaR confidence level

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.hedging_error_weight >= 0, "Hedging error weight must be non-negative"
        assert self.transaction_cost_weight >= 0, "Transaction cost weight must be non-negative"
        assert self.risk_penalty_weight >= 0, "Risk penalty weight must be non-negative"
        assert 0 < self.var_alpha < 1, "VaR alpha must be in (0, 1)"


@dataclass
class MetricsConfig:
    """Performance metrics and evaluation configuration."""

    # Metrics to compute
    compute_sharpe: bool = True
    compute_sortino: bool = True
    compute_max_drawdown: bool = True
    compute_var: bool = True
    compute_cvar: bool = True

    # Risk-free rate for Sharpe/Sortino
    risk_free_rate: float = 0.05

    # VaR/CVaR parameters
    var_alpha: float = 0.05

    # Rolling window metrics
    compute_rolling_metrics: bool = True
    rolling_window_size: int = 50  # Number of paths for rolling metrics


@dataclass
class VisualizationConfig:
    """Visualization and plotting configuration."""

    # General settings
    style: Literal['seaborn', 'ggplot', 'bmh', 'default'] = 'seaborn'
    figure_size: tuple = (12, 8)
    dpi: int = 100
    save_format: Literal['png', 'pdf', 'svg'] = 'png'

    # Specific plots
    plot_pnl_distribution: bool = True
    plot_hedging_error: bool = True
    plot_transaction_costs: bool = True
    plot_hedge_positions: bool = True
    plot_convergence: bool = True  # For RL training

    # Interactive plots
    use_plotly: bool = False  # Use interactive Plotly instead of Matplotlib


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration combining all sub-configs."""

    # Experiment metadata
    name: str = "deep_hedging_experiment"
    description: str = ""
    random_seed: int = 42

    # Sub-configurations
    market: MarketConfig = field(default_factory=MarketConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Directories
    output_dir: Path = field(default_factory=lambda: Path("results"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    # Logging
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = 'INFO'
    verbose: bool = True

    def __post_init__(self):
        """Create directories and setup paths."""
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)

        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Load configuration from dictionary."""
        # Convert nested dicts to dataclasses
        if 'market' in config_dict and isinstance(config_dict['market'], dict):
            config_dict['market'] = MarketConfig(**config_dict['market'])
        if 'trading' in config_dict and isinstance(config_dict['trading'], dict):
            config_dict['trading'] = TradingConfig(**config_dict['trading'])
        if 'backtest' in config_dict and isinstance(config_dict['backtest'], dict):
            config_dict['backtest'] = BacktestConfig(**config_dict['backtest'])
        if 'agent' in config_dict and isinstance(config_dict['agent'], dict):
            config_dict['agent'] = AgentConfig(**config_dict['agent'])
        if 'reward' in config_dict and isinstance(config_dict['reward'], dict):
            config_dict['reward'] = RewardConfig(**config_dict['reward'])
        if 'metrics' in config_dict and isinstance(config_dict['metrics'], dict):
            config_dict['metrics'] = MetricsConfig(**config_dict['metrics'])
        if 'visualization' in config_dict and isinstance(config_dict['visualization'], dict):
            config_dict['visualization'] = VisualizationConfig(**config_dict['visualization'])

        return cls(**config_dict)

    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from YAML file."""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)


# Preset configurations for common use cases
def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig(name="default_experiment")


def get_low_transaction_cost_config() -> ExperimentConfig:
    """Configuration for low transaction cost environment."""
    config = ExperimentConfig(name="low_transaction_cost")
    config.trading.transaction_cost_pct = 0.0001  # 1 bp
    return config


def get_high_transaction_cost_config() -> ExperimentConfig:
    """Configuration for high transaction cost environment."""
    config = ExperimentConfig(name="high_transaction_cost")
    config.trading.transaction_cost_pct = 0.005  # 50 bps
    return config


def get_quick_test_config() -> ExperimentConfig:
    """Configuration for quick testing with fewer paths and timesteps."""
    config = ExperimentConfig(name="quick_test")
    config.backtest.num_paths = 100
    config.backtest.time_steps = 50
    config.agent.total_timesteps = 10_000
    return config


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    print("Default Configuration:")
    print(f"  Market: S0={config.market.spot_price}, σ={config.market.volatility}")
    print(f"  Trading: TC={config.trading.transaction_cost_pct*100:.2f}%")
    print(f"  Backtest: {config.backtest.num_paths} paths, {config.backtest.time_steps} steps")
    print(f"  Agent: {config.agent.algorithm.upper()}, LR={config.agent.learning_rate}")

    # Save configuration
    config.save(Path("config_example.yaml"))
    print(f"\nConfiguration saved to config_example.yaml")
