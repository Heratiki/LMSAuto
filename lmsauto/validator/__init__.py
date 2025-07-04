"""
Validation Engine module for LMSAuto.

This module provides configuration validation capabilities for different LLM platforms,
ensuring configuration consistency and system requirements are met.
"""

import abc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from lmsauto.shared.context import SharedContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
@dataclass
class ValidationResult:
    """Stores the results of a validation check."""
    is_valid: bool
    messages: List[str]
    platform: str
    validation_type: str
    details: Dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

class PlatformValidator(abc.ABC):
    """Abstract base class for platform-specific validators."""
    
    def __init__(self, context: SharedContext):
        """Initialize the validator with shared context."""
        self.context = context
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abc.abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate platform-specific configuration."""
        pass

    @abc.abstractmethod
    def validate_system_requirements(self) -> ValidationResult:
        """Validate system meets platform requirements."""
        pass

    def validate_resource_allocation(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate resource allocation settings.
        Override in platform-specific implementations.
        """
        return ValidationResult(
            is_valid=True,
            messages=[],
            platform="base",
            validation_type="resource_allocation"
        )

    @abc.abstractmethod
    def validate_model_compatibility(self, model_name: str) -> ValidationResult:
        """Validate model compatibility with platform."""
        pass

class LMStudioValidator(PlatformValidator):
    """LMStudio-specific validator implementation."""

    PLATFORM_NAME = "lmstudio"
    MIN_RAM_GB = 8
    MIN_VRAM_MB = 4096

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate LMStudio configuration parameters."""
        messages: List[str] = []
        is_valid = True

        # Validate context window
        if "context_length" in config:
            context_length = config["context_length"]
            if not isinstance(context_length, int) or context_length < 512 or context_length > 32768:
                messages.append(f"Invalid context length: {context_length}. Must be between 512 and 32768.")
                is_valid = False

        # Validate temperature
        if "temperature" in config:
            temp = config["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                messages.append(f"Invalid temperature: {temp}. Must be between 0 and 2.")
                is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            messages=messages,
            platform=self.PLATFORM_NAME,
            validation_type="config"
        )

    def validate_system_requirements(self) -> ValidationResult:
        """Validate system meets LMStudio requirements."""
        messages: List[str] = []
        is_valid = True

        specs = self.context.get_hardware_specs()
        
        # Check RAM
        if "ram_gb" in specs:
            ram_gb = float(specs["ram_gb"])
            if ram_gb < self.MIN_RAM_GB:
                messages.append(f"Insufficient RAM: {ram_gb}GB. Minimum required: {self.MIN_RAM_GB}GB")
                is_valid = False

        # Check VRAM
        if "gpu_vram_mb" in specs:
            vram_mb = float(specs["gpu_vram_mb"])
            if vram_mb < self.MIN_VRAM_MB:
                messages.append(f"Insufficient VRAM: {vram_mb}MB. Minimum required: {self.MIN_VRAM_MB}MB")
                is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            messages=messages,
            platform=self.PLATFORM_NAME,
            validation_type="system_requirements"
        )

    def validate_resource_allocation(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate resource allocation for LMStudio."""
        messages: List[str] = []
        is_valid = True

        specs = self.context.get_hardware_specs()
        
        # Check GPU memory allocation
        if "gpu_layers" in config and "gpu_vram_mb" in specs:
            vram_mb = float(specs["gpu_vram_mb"])
            gpu_layers = int(config["gpu_layers"])
            estimated_vram = gpu_layers * 2048  # Rough estimate: 2GB per layer
            
            if estimated_vram > vram_mb:
                messages.append(
                    f"GPU layer allocation exceeds available VRAM. "
                    f"Estimated requirement: {estimated_vram}MB, Available: {vram_mb}MB"
                )
                is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            messages=messages,
            platform=self.PLATFORM_NAME,
            validation_type="resource_allocation"
        )

    def validate_model_compatibility(self, model_name: str) -> ValidationResult:
        """Validate model compatibility with LMStudio."""
        messages: List[str] = []
        is_valid = True

        model = self.context.get_model(f"{self.PLATFORM_NAME}_{model_name}")
        if not model:
            return ValidationResult(
                is_valid=False,
                messages=[f"Model {model_name} not found in context"],
                platform=self.PLATFORM_NAME,
                validation_type="model_compatibility"
            )

        # Check file extension
        if not model.path.lower().endswith((".gguf", ".bin")):
            messages.append(f"Unsupported model format: {model.path}. Must be .gguf or .bin")
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            messages=messages,
            platform=self.PLATFORM_NAME,
            validation_type="model_compatibility"
        )

class ConfigValidator:
    """Main configuration validator class."""

    def __init__(self, context: SharedContext):
        """Initialize ConfigValidator with shared context."""
        self.context = context
        self.validators: Dict[str, PlatformValidator] = {
            "lmstudio": LMStudioValidator(context)
        }
        self.logger = logging.getLogger(__name__)
        self._validation_results: List[ValidationResult] = []

    def validate(self, platform: str, config: Dict[str, Any], model_name: Optional[str] = None) -> bool:
        """
        Perform comprehensive validation for a platform configuration.
        
        Args:
            platform: Platform identifier (e.g., "lmstudio")
            config: Configuration dictionary to validate
            model_name: Optional model name to validate compatibility
            
        Returns:
            bool: True if all validations pass, False otherwise
        """
        validator = self.validators.get(platform.lower())
        if not validator:
            self.logger.error(f"No validator available for platform: {platform}")
            return False

        self._validation_results = []
        
        # Validate configuration parameters
        config_result = validator.validate_config(config)
        self._validation_results.append(config_result)
        
        # Validate system requirements
        sys_result = validator.validate_system_requirements()
        self._validation_results.append(sys_result)
        
        # Validate resource allocation
        resource_result = validator.validate_resource_allocation(config)
        self._validation_results.append(resource_result)
        
        # Validate model compatibility if model specified
        if model_name:
            model_result = validator.validate_model_compatibility(model_name)
            self._validation_results.append(model_result)

        # Store results in shared context
        self._store_results()

        # Log validation messages
        for result in self._validation_results:
            for message in result.messages:
                if result.is_valid:
                    self.logger.info(f"[{platform}] {message}")
                else:
                    self.logger.error(f"[{platform}] {message}")

        # Return True only if all validations passed
        return all(result.is_valid for result in self._validation_results)

    def get_validation_results(self) -> List[ValidationResult]:
        """Get all validation results from the last validation run."""
        return self._validation_results.copy()

    def _store_results(self):
        """Store validation results in shared context."""
        results_dict: Dict[str, List[Dict[str, Any]]] = {
            "validation_results": [
                {
                    "platform": r.platform,
                    "type": r.validation_type,
                    "valid": r.is_valid,
                    "messages": r.messages,
                    "details": r.details
                }
                for r in self._validation_results
            ]
        }
        self.context.set_config_setting("validation_results", results_dict)