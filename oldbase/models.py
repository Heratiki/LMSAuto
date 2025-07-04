"""Classes for interacting with the LM Studio API."""

import logging
import requests
from typing import Dict, List, Optional, Any

from .utils import load_json, save_json

logger = logging.getLogger(__name__)


class LMStudioAPI:
    """Interface for interacting with LM Studio API."""
    def __init__(self, api_url: str = "http://localhost:1234"):
        """Initialize LM Studio API client.

        Args:
            api_url: Base URL for the LM Studio API.
        """
        self.api_url = api_url.rstrip('/')
        self.logger = logging.getLogger(__name__)

    def discover_models(self) -> List[Dict[str, str]]:
        """Discover available models through LM Studio API.

        Returns:
            List of dictionaries containing model information.
            Each dictionary contains:
                - name: Model name
                - path: Local path to model
                - type: Model type (if available)
        """
        self.logger.info("Discovering models via LM Studio API")
        
        try:
            response = requests.get(f"{self.api_url}/v1/models", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get("data", []):
                model_info = {
                    "name": model_data.get("id", "unknown"),
                    "path": model_data.get("id", "unknown"),  # LM Studio uses ID as path
                    "type": model_data.get("type", "unknown"),
                    "state": model_data.get("state", "unknown"),
                    "max_context_length": model_data.get("max_context_length", 0),
                    "quantization": model_data.get("quantization", "unknown"),
                    "arch": model_data.get("arch", "unknown"),
                    "publisher": model_data.get("publisher", "unknown")
                }
                models.append(model_info)
            
            self.logger.info(f"Discovered {len(models)} models")
            return models
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect to LM Studio API at {self.api_url}: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Error discovering models: {e}")
            return []

    def get_model_settings(self, model_name: str) -> Dict[str, Any]:
        """Get current settings for a specific model.

        Args:
            model_name: Name of the model to get settings for.

        Returns:
            Dictionary containing current model settings.
        """
        # TODO: Implement actual API call once API details are known
        self.logger.info(f"Getting settings for model: {model_name}")
        return {}

    def apply_settings(self, model_name: str,
                       settings: Dict[str, Any]) -> bool:
        """Apply settings to a specific model.

        Args:
            model_name: Name of the model to apply settings to.
            settings: Dictionary containing settings to apply.

        Returns:
            bool: True if settings were applied successfully,
            False otherwise.
        """
        # TODO: Implement actual API call once API details are known
        self.logger.info(f"Applying settings to model: {model_name}")
        return False


class ModelProfileManager:
    """Manages model configuration profiles."""
    def __init__(self, profiles_dir: str = "profiles"):
        """Initialize profile manager.

        Args:
            profiles_dir: Directory where profile JSON files are stored.
        """
        self.profiles_dir = profiles_dir
        self.logger = logging.getLogger(__name__)

    def save_profile(self, model_name: str, settings: Dict[str, Any]) -> None:
        """Save model settings profile to JSON file.

        Args:
            model_name: Name of the model.
            settings: Dictionary containing model settings.
        """
        profile_path = f"{self.profiles_dir}/{model_name}.json"
        self.logger.info(
            f"Saving profile for model {model_name} to {profile_path}"
        )
        save_json(settings, profile_path)

    def load_profile(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load model settings profile from JSON file.

        Args:
            model_name: Name of the model.

        Returns:
            Dictionary containing model settings if profile exists,
            None otherwise.
        """
        profile_path = f"{self.profiles_dir}/{model_name}.json"
        try:
            return load_json(profile_path)
        except FileNotFoundError:
            self.logger.warning(
                f"No profile found for model {model_name}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Error loading profile for model {model_name}: {e}"
            )
            return None
