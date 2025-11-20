"""
Production-grade logging utilities with structured logging support.

Supports both console and file logging with JSON formatting for production environments.
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import uuid


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
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
        
        # Add correlation ID if present
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


class StandardFormatter(logging.Formatter):
    """Standard formatter for console output."""
    
    def __init__(self):
        fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[Path] = None,
    use_json: bool = False
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Logger name (typically __name__)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ('json' or 'standard')
        log_file: Optional path to log file
        use_json: Use JSON formatting (overrides log_format)
        
    Returns:
        Configured logger instance
    """
    # Get logger
    logger = logging.getLogger(name)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Determine log level from environment or default
    import os
    if level is None:
        level = os.getenv('LOG_LEVEL', 'INFO')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Determine format
    if use_json or log_format == 'json':
        formatter = JSONFormatter()
    else:
        formatter = StandardFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler (10MB max, 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_correlation_id() -> str:
    """Generate a unique correlation ID for request tracing."""
    return str(uuid.uuid4())


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds correlation ID to all log messages."""
    
    def __init__(self, logger: logging.Logger, correlation_id: Optional[str] = None):
        super().__init__(logger, {})
        self.correlation_id = correlation_id or get_correlation_id()
    
    def process(self, msg, kwargs):
        """Add correlation ID to log record."""
        kwargs.setdefault('extra', {})
        kwargs['extra']['correlation_id'] = self.correlation_id
        return msg, kwargs


# Default logger for the package
default_logger = setup_logger('telco_churn_mlops')


if __name__ == "__main__":
    # Test logging
    logger = setup_logger(__name__, level='DEBUG')
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Test JSON logging
    json_logger = setup_logger('json_test', level='INFO', use_json=True)
    json_logger.info("JSON formatted message")
    
    # Test correlation ID
    corr_logger = LoggerAdapter(logger, correlation_id='test-123')
    corr_logger.info("Message with correlation ID")