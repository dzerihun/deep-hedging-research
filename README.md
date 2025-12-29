# Deep Hedging Research

A research project comparing neural network-based hedging strategies against traditional Black-Scholes delta hedging under realistic market conditions with transaction costs.

## Overview

This project implements and evaluates deep reinforcement learning approaches to options hedging, comparing them against classical hedging strategies. The framework is designed for research-grade analysis with emphasis on realistic market conditions, transaction costs, and comprehensive performance evaluation.

## Motivation

Traditional hedging strategies like Black-Scholes delta hedging assume continuous, frictionless trading. In practice, transaction costs and discrete rebalancing significantly impact hedging performance. Deep reinforcement learning offers the potential to learn optimal hedging policies that explicitly account for these real-world constraints.

## Key Features

- **Modular Architecture**: Clean, extensible design following software engineering best practices
- **Traditional Strategies**: Implementation of Black-Scholes delta and delta-gamma hedging
- **Deep RL Hedging**: Reinforcement learning agents trained to optimize hedging under transaction costs
- **Realistic Backtesting**: Comprehensive backtesting engine with configurable transaction cost models
- **Performance Analytics**: Extensive metrics for comparing hedging effectiveness
- **Visualization Suite**: Professional charts and analysis tools for research presentation

## Project Structure

```
deep-hedging-research/
├── deep_hedging/               # Main package
│   ├── config.py              # Dataclass-based configuration
│   ├── logger.py              # Logging setup
│   ├── pricing/               # Options pricing (Black-Scholes, Monte Carlo)
│   ├── strategies/            # Hedging strategies (delta, delta-gamma, deep RL)
│   ├── agents/                # RL agents and environments
│   ├── backtesting/           # Backtesting engine with transaction costs
│   ├── metrics/               # Performance evaluation metrics
│   └── visualization/         # Plotting and analysis tools
├── experiments/               # Experiment scripts and configurations
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Unit tests
├── data/                      # Data directory (gitignored)
├── results/                   # Results and outputs (gitignored)
└── logs/                      # Log files (gitignored)
```

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-hedging-research.git
cd deep-hedging-research
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

```python
from deep_hedging.config import ExperimentConfig, MarketConfig, TradingConfig
from deep_hedging.strategies import BlackScholesDeltaHedge
from deep_hedging.backtesting import BacktestEngine

# Configure experiment
market_config = MarketConfig(
    spot_price=100.0,
    volatility=0.2,
    risk_free_rate=0.05
)

trading_config = TradingConfig(
    transaction_cost_pct=0.001,
    rebalance_frequency='daily'
)

# Run backtest
strategy = BlackScholesDeltaHedge(market_config)
engine = BacktestEngine(trading_config)
results = engine.run(strategy)

# Analyze results
print(f"Total P&L: {results.total_pnl:.2f}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
```

## Research Questions

1. **Performance Under Transaction Costs**: How do deep hedging strategies perform compared to Black-Scholes delta hedging when transaction costs are significant?

2. **Rebalancing Frequency**: What is the optimal rebalancing frequency for different hedging strategies?

3. **Market Conditions**: How do hedging strategies perform under different volatility regimes and market conditions?

4. **Risk-Adjusted Returns**: Which strategies provide the best risk-adjusted hedging performance?

## Methodology

### Traditional Strategies
- **Black-Scholes Delta Hedging**: Continuous rebalancing to maintain delta-neutral position
- **Delta-Gamma Hedging**: Higher-order hedging incorporating gamma exposure

### Deep RL Approach
- **Environment**: Custom Gymnasium environment modeling options market dynamics
- **Agent**: PPO/A2C agents from stable-baselines3
- **Reward Function**: Optimized for minimizing hedging error while penalizing transaction costs
- **Training**: Historical market data with realistic transaction cost simulation

### Evaluation Metrics
- Hedging error (mean, std, max)
- Total transaction costs
- Profit & Loss distribution
- Sharpe ratio
- Maximum drawdown
- Win/loss ratio

## Development

### Running Tests
```bash
pytest tests/ -v --cov=deep_hedging
```

### Code Quality
```bash
# Format code
black deep_hedging/ tests/ experiments/

# Lint
flake8 deep_hedging/ tests/ experiments/

# Type checking
mypy deep_hedging/
```

## Key Findings

This research systematically compares deep reinforcement learning (PPO) against Black-Scholes delta hedging for options hedging under transaction costs.

### Main Results

| Strategy | Transaction Cost | Hedging Error | Performance |
|----------|------------------|---------------|-------------|
| **Delta Hedging** | 10 bps | **$1.33** | ✅ Excellent |
| **Delta Hedging** | 0 bps | **$1.30** | ✅ Excellent |
| **PPO-RL (Naive)** | 10 bps | $17.29 | ❌ 13x worse |
| **PPO-RL (Tuned)** | 10 bps | $15.10 | ❌ 11x worse |
| **PPO-RL (Zero Cost)** | 0 bps | $15.83 | ❌ 12x worse |

### Critical Observations

1. **Delta hedging is highly robust**: Achieves $1.30-$1.33 hedging error across all transaction cost levels
2. **Vanilla PPO fails to learn hedging**: All RL configurations perform 11-13x worse than delta
3. **Failure is fundamental, not about costs**: Even at zero transaction costs, RL achieves $15.83 vs delta's $1.30
4. **Root causes identified**: Sparse terminal rewards, insufficient state representation, inadequate exploration

### Detailed Analysis

See [paper/research_findings.md](paper/research_findings.md) for comprehensive analysis including:
- Experimental methodology and setup
- Detailed results for all experiments
- Root cause analysis of RL failures
- Lessons learned and future directions
- Complete reproducibility information

### Value of Negative Results

While the RL agents failed to beat delta hedging, this research contributes:
- **Rigorous benchmarks** for future RL hedging research
- **Documentation of failure modes** to help others avoid similar pitfalls
- **Professional infrastructure** (backtesting, logging, configuration)
- **Systematic methodology** (multiple reward functions, zero-cost validation)
- **Open-source implementation** for community building

**Conclusion**: Vanilla PPO with standard reward shaping is insufficient for options hedging. Future work should explore dense rewards, curriculum learning, and alternative algorithms (SAC/TD3).

## Contributing

This is a research project. Suggestions and improvements are welcome through issues or pull requests.

## License

See LICENSE file for details.

## References

- Buehler, H., Gonon, L., Teichmann, J., & Wood, B. (2019). Deep hedging. *Quantitative Finance*, 19(8), 1271-1291.
- Kolm, P. N., & Ritter, G. (2019). Dynamic replication and hedging: A reinforcement learning approach. *The Journal of Financial Data Science*, 1(1), 159-171.
- Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.

## Contact

For questions or collaboration opportunities, please open an issue or reach out via email.

---

*This project is part of graduate school application portfolio demonstrating quantitative research and software engineering skills.*
