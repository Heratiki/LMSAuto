# lmsauto/shared/context.py

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ModelInfo:
    """Represents information about a discovered language model."""
    name: str
    path: str
    platform: str
    # Add other relevant fields as needed, e.g., size, quantization, last_modified
    metadata: Dict[str, str] = field(default_factory=dict[str, str])

    def get_key(self) -> str:
        """Generates a unique key for the model based on platform and name."""
        # Using a simple combination, ensure this is robust enough for uniqueness
        return f"{self.platform.lower().replace(' ', '_')}_{self.name.lower().replace('/', '_')}"

@dataclass
class SharedContext:
    """
    A thread-safe data class to store and manage shared application state.

    This class holds information discovered about models, system hardware,
    configuration settings, and platform-specific data required by different
    components of the LMSAuto application. Access is managed via a lock
    to ensure thread safety.
    """
    models: Dict[str, ModelInfo] = field(
        default_factory=dict[str, ModelInfo],
        metadata={'description': 'Stores discovered model information, keyed by ModelInfo.get_key().'}
    )
    hardware_specs: Dict[str, str] = field(
        default_factory=dict[str, str],
        metadata={'description': 'Tracks system hardware specifications.'}
    )
    config_settings: Dict[str, str] = field(
        default_factory=dict[str, str],
        metadata={'description': 'Maintains application configuration settings.'}
    )
    platform_data: Dict[str, Dict[str, Any]] = field(
        default_factory=dict[str, dict[str, Any]],
        metadata={'description': 'Handles platform-specific data (LM Studio, Ollama, vLLM).'}
    )

    # Private lock for ensuring thread safety during attribute access/modification
    _lock: threading.Lock = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Initialize the thread lock after the dataclass is created."""
        self._lock = threading.Lock()

    def add_model(self, model_info: ModelInfo):
        """Adds or updates a model's information in the context thread-safely."""
        # Removed unnecessary isinstance check
        key = model_info.get_key()
        with self._lock:
            self.models[key] = model_info

    def get_model(self, key: str) -> Optional[ModelInfo]:
        """Gets model information by key in a thread-safe manner."""
        with self._lock:
            return self.models.get(key)

    def get_all_models(self) -> Dict[str, ModelInfo]:
        """Gets a copy of all model information in a thread-safe manner."""
        with self._lock:
            # Return a copy to prevent modification outside the lock
            return self.models.copy()

    def get_config_setting(self, key: str, default: Any = None) -> Any:
        """Gets a configuration setting in a thread-safe manner."""
        with self._lock:
            return self.config_settings.get(key, default)

    def set_config_setting(self, key: str, value: Any):
        """Sets a configuration setting in a thread-safe manner."""
        with self._lock:
            self.config_settings[key] = value

    def set_hardware_specs(self, specs: Dict[str, str]):
        """Sets the hardware specifications dictionary thread-safely."""
        with self._lock:
            # Overwrite the entire dictionary
            self.hardware_specs = specs.copy()

    def get_hardware_specs(self) -> Dict[str, str]:
        """Gets a copy of the hardware specifications dictionary thread-safely."""
        with self._lock:
            return self.hardware_specs.copy()