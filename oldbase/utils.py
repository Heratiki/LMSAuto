"""Utility functions for logging and configuration validation."""

import json
import logging
from pathlib import Path
from typing import Any, Dict

# Configure logging


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration.

    Args:
        log_level: The logging level to use. Defaults to "INFO".
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("lmsauto.log")
        ]
    )


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.

    Args:
        config: Dictionary containing configuration parameters.

    Returns:
        bool: True if configuration is valid, False otherwise.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    required_fields = ["model_name", "context_length", "temperature"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(config["context_length"], int):
        raise ValueError("context_length must be an integer")

    if not isinstance(config["temperature"], (int, float)):
        raise ValueError("temperature must be a number")

    if not 0 <= config["temperature"] <= 1:
        raise ValueError("temperature must be between 0 and 1")

    return True


def load_json(file_path: str) -> Dict[str, Any]:
    """Load and parse a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Dict containing the parsed JSON data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with path.open() as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to a JSON file.

    Args:
        data: Dictionary to save.
        file_path: Path where to save the JSON file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w') as f:
        json.dump(data, f, indent=2)
