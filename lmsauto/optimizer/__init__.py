"""
Optimizer module for LMSAuto.

This module provides optimization capabilities for language model configurations
based on validation results, system profiling data, and platform-specific strategies.
"""

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..shared.context import SharedContext

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Represents the result of an optimization attempt."""
    timestamp: datetime
    platform: str
    model_key: str
    changes: Dict[str, Any]
    metrics: Dict[str, float]
    success: bool
    message: str

@dataclass
class OptimizationHistory:
    """Tracks optimization attempts and their results."""
    model_key: str
    entries: List[OptimizationResult] = field(default_factory=lambda: [])

    def add_entry(self, result: OptimizationResult):
        """Adds a new optimization result to the history."""
        self.entries.append(result)

    def get_latest(self) -> Optional[OptimizationResult]:
        """Returns the most recent optimization result."""
        return self.entries[-1] if self.entries else None

class BaseOptimizer(abc.ABC):
    """Abstract base class for platform-specific optimizers."""
    
    @abc.abstractmethod
    def optimize_gpu_layers(self, model_key: str) -> Dict[str, Any]:
        """Optimizes GPU layer allocation for the model."""
        pass

    @abc.abstractmethod
    def optimize_memory_usage(self, model_key: str) -> Dict[str, Any]:
        """Optimizes memory usage and balancing."""
        pass

    @abc.abstractmethod
    def optimize_context_length(self, model_key: str) -> Dict[str, Any]:
        """Optimizes context length based on available resources."""
        pass

    @abc.abstractmethod
    def optimize_batch_size(self, model_key: str) -> Dict[str, Any]:
        """Optimizes batch size for inference."""
        pass

class LMStudioOptimizer(BaseOptimizer):
    """LM Studio specific optimization implementation."""
    
    def __init__(self, context: SharedContext):
        self.context = context

    def optimize_gpu_layers(self, model_key: str) -> Dict[str, Any]:
        """Optimizes GPU layer allocation for LM Studio models."""
        logger.info(f"Optimizing GPU layers for model: {model_key}")
        specs = self.context.get_hardware_specs()
        
        if "gpu_memory" not in specs:
            logger.warning("No GPU memory information available")
            return {}
            
        gpu_mem = int(specs["gpu_memory"])
        model = self.context.get_model(model_key)
        
        if not model:
            raise ValueError(f"Model {model_key} not found")
            
        # Basic layer allocation strategy based on available GPU memory
        if gpu_mem >= 24000:  # 24GB or more
            return {"gpu_layers": "all"}
        elif gpu_mem >= 12000:  # 12GB
            return {"gpu_layers": "auto"}
        else:
            return {"gpu_layers": "balanced"}

    def optimize_memory_usage(self, model_key: str) -> Dict[str, Any]:
        """Optimizes memory usage for LM Studio models."""
        logger.info(f"Optimizing memory usage for model: {model_key}")
        specs = self.context.get_hardware_specs()
        
        total_mem = int(specs.get("total_memory", 0))
        if total_mem < 16000:  # Less than 16GB
            return {
                "low_memory": True,
                "mmap": True,
                "mlock": False
            }
        return {
            "low_memory": False,
            "mmap": False,
            "mlock": True
        }

    def optimize_context_length(self, model_key: str) -> Dict[str, Any]:
        """Optimizes context length for LM Studio models."""
        logger.info(f"Optimizing context length for model: {model_key}")
        specs = self.context.get_hardware_specs()
        
        gpu_mem = int(specs.get("gpu_memory", 0))
        total_mem = int(specs.get("total_memory", 0))
        
        # Conservative context length based on available memory
        if gpu_mem >= 24000 or total_mem >= 32000:
            return {"context_length": 8192}
        elif gpu_mem >= 12000 or total_mem >= 16000:
            return {"context_length": 4096}
        else:
            return {"context_length": 2048}

    def optimize_batch_size(self, model_key: str) -> Dict[str, Any]:
        """Optimizes batch size for LM Studio models."""
        logger.info(f"Optimizing batch size for model: {model_key}")
        specs = self.context.get_hardware_specs()
        
        gpu_mem = int(specs.get("gpu_memory", 0))
        
        if gpu_mem >= 24000:
            return {"batch_size": 512}
        elif gpu_mem >= 12000:
            return {"batch_size": 256}
        else:
            return {"batch_size": 128}

class ConfigOptimizer:
    """Main configuration optimizer class."""
    
    def __init__(self, context: SharedContext):
        self.context = context
        self.history: Dict[str, OptimizationHistory] = {}
        self.optimizers: Dict[str, BaseOptimizer] = {
            "lm_studio": LMStudioOptimizer(context)
        }

    def _get_optimizer(self, platform: str) -> BaseOptimizer:
        """Gets the appropriate optimizer for the platform."""
        optimizer = self.optimizers.get(platform.lower())
        if not optimizer:
            raise ValueError(f"No optimizer available for platform: {platform}")
        return optimizer

    def optimize(self, model_key: str) -> OptimizationResult:
        """
        Optimizes configuration for a specific model using platform-specific strategies.
        
        Args:
            model_key: The unique identifier for the model
            
        Returns:
            OptimizationResult containing the optimization changes and metrics
        """
        logger.info(f"Starting optimization for model: {model_key}")
        
        model = self.context.get_model(model_key)
        if not model:
            raise ValueError(f"Model {model_key} not found")
            
        platform = model.platform.lower()
        optimizer = self._get_optimizer(platform)
        
        # Collect all optimizations
        changes: Dict[str, Any] = {}
        changes.update(optimizer.optimize_gpu_layers(model_key))
        changes.update(optimizer.optimize_memory_usage(model_key))
        changes.update(optimizer.optimize_context_length(model_key))
        changes.update(optimizer.optimize_batch_size(model_key))
        
        # Apply changes to shared context
        for key, value in changes.items():
            self.context.set_config_setting(f"{model_key}.{key}", value)
        
        # Create optimization result
        result = OptimizationResult(
            timestamp=datetime.now(),
            platform=platform,
            model_key=model_key,
            changes=changes,
            metrics={},  # To be populated with validation metrics
            success=True,
            message="Optimization completed successfully"
        )
        
        # Update history
        if model_key not in self.history:
            self.history[model_key] = OptimizationHistory(model_key)
        self.history[model_key].add_entry(result)
        
        logger.info(f"Optimization completed for model: {model_key}")
        return result

    def get_optimization_history(self, model_key: str) -> Optional[OptimizationHistory]:
        """Retrieves optimization history for a specific model."""
        return self.history.get(model_key)