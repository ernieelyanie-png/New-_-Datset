"""
Logging utilities for the ML Risk Prediction application.
Provides structured logging with file rotation and console output.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with color coding."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


def setup_logger(
    name: str = 'ml_risk_prediction',
    log_dir: str = 'logs',
    log_file: str = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console_output: bool = True,
    colored_console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to store log files
        log_file: Log file name (default: {name}_{timestamp}.log)
        level: Logging level
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
        colored_console: Whether to use colored console output
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Generate log file name with timestamp if not provided
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{name}_{timestamp}.log"
    
    log_path = os.path.join(log_dir, log_file)
    
    # Create file handler with rotation
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if colored_console:
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    logger.info(f"Logger '{name}' initialized. Log file: {log_path}")
    
    return logger


def get_logger(name: str = 'ml_risk_prediction') -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up with defaults
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


class LogContext:
    """Context manager for logging with additional context information."""
    
    def __init__(self, logger: logging.Logger, context: dict):
        """
        Initialize log context.
        
        Args:
            logger: Logger instance
            context: Dictionary of context information
        """
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Enter context and add context to log records."""
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original log record factory."""
        logging.setLogRecordFactory(self.old_factory)


def log_function_call(logger: logging.Logger = None):
    """
    Decorator to log function calls with parameters and execution time.
    
    Args:
        logger: Logger instance (default: get default logger)
    
    Returns:
        Decorated function
    """
    if logger is None:
        logger = get_logger()
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.info(f"Calling function: {func_name}")
            logger.debug(f"Arguments: args={args}, kwargs={kwargs}")
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Function {func_name} completed in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Function {func_name} failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_exception(logger: logging.Logger, message: str = "An error occurred"):
    """
    Log exception with full traceback.
    
    Args:
        logger: Logger instance
        message: Custom error message
    """
    logger.exception(message)


def log_model_metrics(logger: logging.Logger, metrics: dict, prefix: str = "Model"):
    """
    Log model performance metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        prefix: Prefix for log message
    """
    logger.info(f"{prefix} Metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"  {metric_name}: {metric_value}")


def log_data_info(logger: logging.Logger, data, name: str = "Dataset"):
    """
    Log information about a dataset.
    
    Args:
        logger: Logger instance
        data: DataFrame or array-like data
        name: Name of the dataset
    """
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            logger.info(f"{name} Info:")
            logger.info(f"  Shape: {data.shape}")
            logger.info(f"  Columns: {list(data.columns)}")
            logger.info(f"  Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            logger.info(f"  Missing values: {data.isnull().sum().sum()}")
        else:
            logger.info(f"{name} Info:")
            logger.info(f"  Shape: {data.shape if hasattr(data, 'shape') else len(data)}")
    except ImportError:
        logger.info(f"{name} Shape: {data.shape if hasattr(data, 'shape') else len(data)}")


if __name__ == "__main__":
    # Example usage
    logger = setup_logger(
        name='example',
        log_dir='logs',
        level=logging.DEBUG,
        colored_console=True
    )
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Example with context
    with LogContext(logger, {'user_id': '12345', 'session': 'abc'}):
        logger.info("Log message with context")
    
    # Example with metrics
    log_model_metrics(logger, {
        'accuracy': 0.9523,
        'precision': 0.8765,
        'recall': 0.9012,
        'f1_score': 0.8886
    })
