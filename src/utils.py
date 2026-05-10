"""
Utilities module for the Fact-Checking Web App.

This module provides shared functionality for logging, caching, configuration,
and helper functions used across the application.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Tuple, List, Optional
import re

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


def get_api_key(key_name: str) -> str:
    """
    Securely retrieve API key from Streamlit secrets or environment variables.
    
    Checks Streamlit secrets first (for deployment), then falls back to
    environment variables (for local development).
    
    Args:
        key_name: Name of the API key (e.g., "NVIDIA_API_KEY", "TAVILY_API_KEY")
        
    Returns:
        str: API key value
        
    Raises:
        ValueError: If API key is not found in either location
        
    Example:
        >>> nvidia_key = get_api_key("NVIDIA_API_KEY")
        >>> tavily_key = get_api_key("TAVILY_API_KEY")
    """
    # Try Streamlit secrets first (for deployment)
    if STREAMLIT_AVAILABLE:
        try:
            api_key = st.secrets.get(key_name)
            if api_key:
                return api_key
        except (AttributeError, FileNotFoundError):
            pass  # Secrets not available, fall back to env vars
    
    # Fall back to environment variables (for local development)
    api_key = os.getenv(key_name)
    
    if not api_key:
        raise ValueError(
            f"{key_name} not found. Please set it in:\n"
            f"  - Streamlit secrets (for deployment): .streamlit/secrets.toml\n"
            f"  - Environment variables (for local): .env file"
        )
    
    return api_key


def setup_logging() -> logging.Logger:
    """
    Configure logging with both file and console output.
    
    Creates a logger with:
    - File handler: Rotating log file (10MB max, 5 backups) at INFO level
    - Console handler: Console output at WARNING level
    - Format: [TIMESTAMP] [LEVEL] [MODULE] MESSAGE
    
    Returns:
        logging.Logger: Configured logger instance for the fact_checker application
        
    Example:
        >>> logger = setup_logging()
        >>> logger.info("Processing started")
        >>> logger.warning("API rate limit approaching")
    """
    logger = logging.getLogger("fact_checker")
    logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger
    
    # File handler with rotation (10MB max, 5 backups)
    file_handler = RotatingFileHandler(
        "fact_checker.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    )
    
    # Console handler (WARNING level and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config() -> dict:
    """
    Load configuration from environment variables using python-dotenv.
    
    Reads environment variables from .env file if present, then returns
    a dictionary containing all configuration values needed by the application.
    Uses secure API key retrieval via get_api_key().
    
    Returns:
        dict: Configuration dictionary with API keys and settings
        
    Example:
        >>> config = load_config()
        >>> nvidia_key = config.get("NVIDIA_API_KEY")
    """
    from dotenv import load_dotenv
    
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Return configuration dictionary with all required settings
    # Note: API keys should be retrieved via get_api_key() when needed
    config = {
        "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "nvidia"),
    }
    
    return config


def validate_api_keys() -> Tuple[bool, List[str]]:
    """
    Validate required API keys are present in environment variables or Streamlit secrets.
    
    Checks for the presence of NVIDIA_API_KEY and TAVILY_API_KEY using the secure
    get_api_key() function. Returns validation status and a list of any missing keys.
    
    Returns:
        Tuple[bool, List[str]]: (validation_status, list_of_missing_keys)
            - validation_status: True if all required keys are present, False otherwise
            - list_of_missing_keys: List of missing API key names (empty if all present)
        
    Example:
        >>> valid, missing = validate_api_keys()
        >>> if not valid:
        ...     print(f"Missing keys: {missing}")
    """
    missing_keys = []
    
    # Check for NVIDIA_API_KEY
    try:
        get_api_key("NVIDIA_API_KEY")
    except ValueError:
        missing_keys.append("NVIDIA_API_KEY")
    
    # Check for TAVILY_API_KEY
    try:
        get_api_key("TAVILY_API_KEY")
    except ValueError:
        missing_keys.append("TAVILY_API_KEY")
    
    # Return validation status and list of missing keys
    validation_status = len(missing_keys) == 0
    return (validation_status, missing_keys)


def sanitize_log_message(message: str) -> str:
    """
    Remove API keys and sensitive data from log messages.
    
    Uses regex patterns to identify and redact:
    - API keys (patterns like api_key, api-key, apikey)
    - Bearer tokens
    - Other sensitive authentication tokens
    
    Args:
        message: Log message that may contain sensitive data
        
    Returns:
        str: Sanitized message with sensitive data redacted
        
    Example:
        >>> sanitize_log_message("API key: sk-abc123")
        'API key: ***REDACTED***'
        >>> sanitize_log_message('{"api_key": "secret123"}')
        '{"api_key": "***REDACTED***"}'
    """
    # Redact API keys (various formats: api_key, api-key, apikey, etc.)
    # Matches patterns like: api_key=value, api-key: value, "apikey":"value", "API key: value"
    message = re.sub(
        r'(api[_\s-]?key["\s:=]+)[\w-]+',
        r'\1***REDACTED***',
        message,
        flags=re.IGNORECASE
    )
    
    # Redact bearer tokens
    message = re.sub(
        r'(bearer\s+)[\w-]+',
        r'\1***REDACTED***',
        message,
        flags=re.IGNORECASE
    )
    
    # Redact authorization headers
    message = re.sub(
        r'(authorization["\s:=]+["\']?(?:bearer\s+)?)["\']?[\w-]+["\']?',
        r'\1***REDACTED***',
        message,
        flags=re.IGNORECASE
    )
    
    # Redact tokens in various formats
    message = re.sub(
        r'(token["\s:=]+)[\w-]+',
        r'\1***REDACTED***',
        message,
        flags=re.IGNORECASE
    )
    
    return message


def format_date(date_str: str) -> str:
    """
    Parse and format ISO date strings.
    
    Parses ISO 8601 format date strings and returns them in YYYY-MM-DD format.
    Handles various ISO formats including timezone information.
    
    Args:
        date_str: ISO format date string (e.g., "2024-01-15T10:30:00Z" or "2024-01-15")
        
    Returns:
        str: Formatted date string in YYYY-MM-DD format
        
    Raises:
        ValueError: If date_str cannot be parsed as a valid date
        
    Example:
        >>> format_date("2024-01-15T10:30:00Z")
        '2024-01-15'
        >>> format_date("2024-01-15T10:30:00+00:00")
        '2024-01-15'
        >>> format_date("2024-01-15")
        '2024-01-15'
    """
    from datetime import datetime
    
    if not date_str:
        raise ValueError("Date string cannot be empty")
    
    # Try parsing with various ISO formats
    formats_to_try = [
        "%Y-%m-%dT%H:%M:%SZ",           # ISO with Z timezone
        "%Y-%m-%dT%H:%M:%S%z",          # ISO with timezone offset
        "%Y-%m-%dT%H:%M:%S.%fZ",        # ISO with microseconds and Z
        "%Y-%m-%dT%H:%M:%S.%f%z",       # ISO with microseconds and timezone
        "%Y-%m-%dT%H:%M:%S",            # ISO without timezone
        "%Y-%m-%d",                      # Date only
    ]
    
    # Try parsing with each format
    for fmt in formats_to_try:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    # If none of the formats worked, try fromisoformat (Python 3.7+)
    try:
        # Remove 'Z' suffix if present and replace with '+00:00' for fromisoformat
        normalized_str = date_str.replace('Z', '+00:00')
        parsed_date = datetime.fromisoformat(normalized_str)
        return parsed_date.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        pass
    
    # If all parsing attempts failed, raise an error
    raise ValueError(f"Unable to parse date string: {date_str}")


def truncate_text(text: str, max_length: int) -> str:
    """
    Truncate text with ellipsis if it exceeds max_length.
    
    If the text is longer than max_length, it will be truncated and '...'
    will be appended. The total length including '...' will not exceed max_length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length before truncation (must be >= 3 for ellipsis)
        
    Returns:
        str: Truncated text with ellipsis if needed, original text if within limit
        
    Raises:
        ValueError: If max_length is less than 3
        
    Example:
        >>> truncate_text("This is a long text", 10)
        'This is...'
        >>> truncate_text("Short", 10)
        'Short'
        >>> truncate_text("Exactly ten!", 12)
        'Exactly ten!'
    """
    if max_length < 3:
        raise ValueError("max_length must be at least 3 to accommodate ellipsis")
    
    if len(text) <= max_length:
        return text
    
    # Truncate to max_length - 3 to make room for '...'
    return text[:max_length - 3] + "..."
