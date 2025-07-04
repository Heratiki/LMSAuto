"""
Configuration Generator Module

Provides functionality to generate platform-specific configurations for various LLM platforms
based on system specifications and model information.
"""

import abc
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from ..shared.context import SharedContext, ModelInfo

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class BaseConfig:
    """Base configuration dataclass with common settings."""
    model_path: str
    context_size: int = 4096
    threads: int = 4
    batch_size: int = 512
    gpu_layers: int = 0
    compute_type: str = "default"
    metadata: Dict[str, Any] = field(default_factory=lambda: dict())

@dataclass
class LMStudioConfig(BaseConfig):
    """LM Studio specific configuration settings."""
    prompt_template: str = "default"
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    stopping_strings: list[str] = field(default_factory=lambda: list())

class ConfigGeneratorInterface(abc.ABC):
    """Abstract base class defining the interface for platform-specific config generators."""
    
    def __init__(self, shared_context: SharedContext):
        """Initialize with shared context for access to model and system information."""
        self.shared_context = shared_context
        
    @abc.abstractmethod
    def generate_config(self, model_info: ModelInfo) -> Any:
        """Generate platform-specific configuration for the given model."""
        pass
    
    @abc.abstractmethod
    def validate_config(self, config: Any) -> bool:
        """Validate the generated configuration."""
        pass

class LMStudioConfigGenerator(ConfigGeneratorInterface):
    """Configuration generator for LM Studio platform."""

    def _determine_gpu_layers(self, gpu_memory_gb: Optional[int]) -> int:
        """Determine optimal number of GPU layers based on available GPU memory."""
        if not gpu_memory_gb:
            return 0
        # Simple heuristic: 1 layer per GB of VRAM, max 32 layers
        return min(gpu_memory_gb, 32)

    def _determine_compute_type(self, gpu_memory_gb: Optional[int]) -> str:
        """Determine optimal compute type based on system specifications."""
        if not gpu_memory_gb:
            return "float32"  # CPU only
        elif gpu_memory_gb >= 8:
            return "float16"  # Sufficient VRAM for FP16
        else:
            return "int8"  # Limited VRAM, use quantization

    def generate_config(self, model_info: ModelInfo) -> LMStudioConfig:
        """Generate LM Studio configuration based on model and system information."""
        logger.info(f"Generating LM Studio config for model: {model_info.name}")
        
        # Get system specifications
        hw_specs = self.shared_context.get_hardware_specs()
        gpu_memory = int(hw_specs.get("gpu_memory_gb", "0"))
        cpu_threads = int(hw_specs.get("cpu_threads", "4"))
        
        # Create configuration
        config = LMStudioConfig(
            model_path=model_info.path,
            threads=min(cpu_threads, 8),  # Cap at 8 threads
            gpu_layers=self._determine_gpu_layers(gpu_memory),
            compute_type=self._determine_compute_type(gpu_memory),
            metadata={
                "platform": "lmstudio",
                "model_name": model_info.name,
                "generated_at": "timestamp",  # TODO: Add actual timestamp
            }
        )
        
        # Store in shared context
        self.shared_context.platform_data.setdefault("lmstudio", {})[model_info.name] = config
        
        return config

    def validate_config(self, config: LMStudioConfig) -> bool:
        """Validate LM Studio configuration."""
        try:
            assert isinstance(config, LMStudioConfig), "Invalid config type"
            assert config.model_path, "Model path cannot be empty"
            assert 0 <= config.temperature <= 1, "Temperature must be between 0 and 1"
            assert 0 <= config.top_p <= 1, "Top-p must be between 0 and 1"
            assert config.threads > 0, "Thread count must be positive"
            assert config.batch_size > 0, "Batch size must be positive"
            assert config.gpu_layers >= 0, "GPU layers cannot be negative"
            return True
        except AssertionError as e:
            logger.error(f"Config validation failed: {str(e)}")
            return False

class ConfigGenerator:
    """Main configuration generator class that handles platform-specific generators."""
    
    def __init__(self, shared_context: SharedContext):
        """Initialize with shared context and platform-specific generators."""
        self.shared_context = shared_context
        self.generators: Dict[str, ConfigGeneratorInterface] = {
            "lmstudio": LMStudioConfigGenerator(shared_context)
        }
        
    def generate_config(self, model_info: ModelInfo) -> Optional[Any]:
        """Generate configuration for the specified model using appropriate generator."""
        generator = self.generators.get(model_info.platform.lower())
        if not generator:
            logger.error(f"No config generator available for platform: {model_info.platform}")
            return None
            
        try:
            config = generator.generate_config(model_info)
            if generator.validate_config(config):
                logger.info(f"Successfully generated config for {model_info.name}")
                return config
            else:
                logger.error(f"Generated config validation failed for {model_info.name}")
                return None
        except Exception as e:
            logger.error(f"Error generating config for {model_info.name}: {str(e)}")
            return None