"""
Logging configuration for the deep hedging project.

Provides structured logging with file and console handlers, formatted for both
development and production use.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    Useful for log aggregation and analysis.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)

        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for console output.
    Makes logs easier to read during development.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
            )

        # Format the message
        formatted = super().format(record)

        # Reset levelname to original (for other handlers)
        record.levelname = levelname

        return formatted


def setup_logger(
    name: str = "deep_hedging",
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    console_output: bool = True,
    file_output: bool = True,
    json_logs: bool = False,
    colored_console: bool = True,
) -> logging.Logger:
    """
    Setup and configure logger for the project.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/)
        console_output: Enable console logging
        file_output: Enable file logging
        json_logs: Use JSON format for file logs
        colored_console: Use colored output for console

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))

        if colored_console and sys.stdout.isatty():
            console_format = ColoredFormatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:
            console_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler
    if file_output:
        if log_dir is None:
            log_dir = Path("logs")
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)  # File gets all logs

        if json_logs:
            file_format = JsonFormatter()
        else:
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance by name.

    If the logger doesn't exist, it will be created with default settings.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        return setup_logger(
            name=name.split('.')[0],  # Use base package name
            log_level="INFO",
            console_output=True,
            file_output=True,
            colored_console=True,
        )

    return logger


class ExperimentLogger:
    """
    Specialized logger for tracking experiments.

    Provides structured logging of experiment parameters, metrics, and results.
    """

    def __init__(self, experiment_name: str, output_dir: Path):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            output_dir: Directory to save experiment logs
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger(
            name=f"experiment_{experiment_name}",
            log_dir=self.output_dir,
            file_output=True,
            console_output=True,
        )

        # Metrics log file
        self.metrics_file = self.output_dir / f"{experiment_name}_metrics.jsonl"

    def log_config(self, config: dict) -> None:
        """Log experiment configuration."""
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info("=" * 80)
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 80)

        # Save config to JSON
        config_file = self.output_dir / f"{self.experiment_name}_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> None:
        """
        Log metrics during experiment.

        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number (e.g., training iteration)
        """
        # Log to console
        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items()])
        if step is not None:
            self.logger.info(f"Step {step} | {metric_str}")
        else:
            self.logger.info(f"Metrics | {metric_str}")

        # Append to metrics file (JSONL format)
        with open(self.metrics_file, 'a') as f:
            metric_data = {'timestamp': datetime.now().isoformat()}
            if step is not None:
                metric_data['step'] = step
            metric_data.update(metrics)
            f.write(json.dumps(metric_data) + '\n')

    def log_result(self, result: dict) -> None:
        """Log final experiment results."""
        self.logger.info("=" * 80)
        self.logger.info("Experiment Results:")
        self.logger.info("=" * 80)
        for key, value in result.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 80)

        # Save results to JSON
        result_file = self.output_dir / f"{self.experiment_name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    def log_error(self, error: Exception) -> None:
        """Log error during experiment."""
        self.logger.error(f"Experiment failed with error: {str(error)}", exc_info=True)


# Module-level convenience functions
_default_logger: Optional[logging.Logger] = None


def get_default_logger() -> logging.Logger:
    """Get or create the default project logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger(
            name="deep_hedging",
            log_level="INFO",
            console_output=True,
            file_output=True,
            colored_console=True,
        )
    return _default_logger


if __name__ == "__main__":
    # Example usage
    logger = get_default_logger()

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    # Experiment logger example
    exp_logger = ExperimentLogger("test_experiment", Path("logs/experiments"))
    exp_logger.log_config({
        "spot_price": 100.0,
        "volatility": 0.2,
        "num_paths": 1000
    })
    exp_logger.log_metrics({"pnl": 1234.56, "sharpe": 1.5}, step=100)
    exp_logger.log_result({"final_pnl": 5678.90, "success": True})
