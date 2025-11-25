"""
Train a deep RL agent for option hedging using PPO.

This script trains a PPO agent to learn optimal hedging strategies
under transaction costs, using the HedgingEnv Gymnasium environment.
"""

import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from deep_hedging.config import MarketConfig, TradingConfig
from deep_hedging.agents.hedging_env import HedgingEnv
from deep_hedging.logger import get_logger

logger = get_logger(__name__)


def make_env(rank: int = 0, seed: int = 0):
    """
    Create a single hedging environment.

    Args:
        rank: Environment rank (for parallel envs)
        seed: Random seed

    Returns:
        Function that creates environment
    """
    def _init():
        market_config = MarketConfig(
            spot_price=100.0,
            volatility=0.2,
            risk_free_rate=0.05,
            strike_price=100.0,
            time_to_maturity=1.0,
            option_type='call'
        )

        trading_config = TradingConfig(
            transaction_cost_pct=0.001,  # 10 bps
            rebalance_frequency='daily'
        )

        env = HedgingEnv(
            market_config=market_config,
            trading_config=trading_config,
            num_steps=252,
            random_seed=seed + rank
        )

        env = Monitor(env)  # Wrap to log episode statistics
        return env

    return _init


def train_ppo_agent(
    total_timesteps: int = 100_000,
    save_dir: Path = Path("models"),
    log_dir: Path = Path("logs/tensorboard"),
    eval_freq: int = 10_000,
    n_eval_episodes: int = 10,
    seed: int = 42
):
    """
    Train PPO agent for hedging.

    Args:
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        eval_freq: Evaluate every N steps
        n_eval_episodes: Number of episodes for evaluation
        seed: Random seed

    Returns:
        Trained PPO model
    """
    logger.info("="*60)
    logger.info("TRAINING PPO AGENT FOR DEEP HEDGING")
    logger.info("="*60)

    # Create directories
    save_dir = Path(save_dir)
    log_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving models to: {save_dir}")
    logger.info(f"Tensorboard logs: {log_dir}")
    logger.info(f"Training for {total_timesteps:,} timesteps")

    # Create training environment
    logger.info("Creating training environment...")
    env = DummyVecEnv([make_env(rank=0, seed=seed)])

    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(rank=1000, seed=seed + 1000)])

    # Create PPO agent
    logger.info("Initializing PPO agent...")
    logger.info("  Policy: MlpPolicy")
    logger.info("  Network: [256, 256]")
    logger.info("  Learning rate: 3e-4")
    logger.info("  Batch size: 64")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,  # Steps per update
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Entropy coefficient for exploration
        policy_kwargs=dict(
            net_arch=[256, 256],  # Two hidden layers
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        tensorboard_log=str(log_dir),
        seed=seed
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_freq // 4, 1),
        save_path=str(save_dir / "checkpoints"),
        name_prefix="ppo_hedging"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir / "best"),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Train
    logger.info("Starting training...")
    logger.info("-"*60)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name="ppo_hedging",
            progress_bar=True
        )

        # Save final model
        final_model_path = save_dir / "ppo_hedging_final.zip"
        model.save(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        # Save current model
        interrupted_path = save_dir / "ppo_hedging_interrupted.zip"
        model.save(interrupted_path)
        logger.info(f"Interrupted model saved to: {interrupted_path}")

    logger.info("-"*60)
    logger.info("Training completed!")
    logger.info("="*60)

    return model


if __name__ == "__main__":
    import torch

    # Configuration
    TOTAL_TIMESTEPS = 100_000
    SAVE_DIR = Path("models")
    LOG_DIR = Path("logs/tensorboard")
    SEED = 42

    # Train agent
    model = train_ppo_agent(
        total_timesteps=TOTAL_TIMESTEPS,
        save_dir=SAVE_DIR,
        log_dir=LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=10,
        seed=SEED
    )

    logger.info("Training script completed successfully!")

    # Quick test
    logger.info("\nTesting trained agent on one episode...")
    env = make_env(seed=999)()
    obs, _ = env.reset()
    total_reward = 0

    for step in range(252):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated:
            logger.info(f"Test episode finished")
            logger.info(f"  Total reward: {total_reward:.4f}")
            logger.info(f"  Terminal P&L: {info['terminal_pnl']:.4f}")
            logger.info(f"  Cumulative costs: {info['cumulative_costs']:.4f}")
            break
