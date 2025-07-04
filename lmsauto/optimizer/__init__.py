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



class OllamaOptimizer(BaseOptimizer):
    """Ollama specific optimization implementation (Placeholder)."""

    def __init__(self, context: SharedContext):
        self.context = context

    def optimize_gpu_layers(self, model_key: str) -> Dict[str, Any]:
        """Suggests num_gpu based on available VRAM."""
        logger.info(f"Optimizing GPU layers (num_gpu) for Ollama model: {model_key}")
        specs = self.context.get_hardware_specs()
        gpu_mem = int(specs.get("gpu_memory", 0))

        # Simple heuristic for num_gpu (adjust based on model size/needs later)
        if gpu_mem >= 24000:
            return {"num_gpu": 999} # Use as many layers as possible
        elif gpu_mem >= 12000:
            return {"num_gpu": 40} # Suggest a moderate number
        elif gpu_mem >= 8000:
            return {"num_gpu": 20} # Suggest fewer layers
        else:
            logger.warning("Low GPU memory, suggesting minimal GPU offload for Ollama.")
            return {"num_gpu": 10}

    def optimize_memory_usage(self, model_key: str) -> Dict[str, Any]:
        """Suggests low_vram based on total system memory."""
        logger.info(f"Optimizing memory usage (low_vram) for Ollama model: {model_key}")
        specs = self.context.get_hardware_specs()
        total_mem = int(specs.get("total_memory", 0))

        if total_mem < 16000: # Less than 16GB RAM
            logger.warning("Low system memory, enabling low_vram for Ollama.")
            return {"low_vram": True}
        else:
            return {"low_vram": False}

    def optimize_context_length(self, model_key: str) -> Dict[str, Any]:
        """Suggests num_ctx based on available memory."""
        logger.info(f"Optimizing context length (num_ctx) for Ollama model: {model_key}")
        specs = self.context.get_hardware_specs()
        gpu_mem = int(specs.get("gpu_memory", 0))
        total_mem = int(specs.get("total_memory", 0))

        # Similar logic to LMStudioOptimizer
        if gpu_mem >= 24000 or total_mem >= 32000:
            return {"num_ctx": 8192}
        elif gpu_mem >= 12000 or total_mem >= 16000:
            return {"num_ctx": 4096}
        else:
            logger.warning("Low memory, suggesting smaller context length for Ollama.")
            return {"num_ctx": 2048}

    def optimize_batch_size(self, model_key: str) -> Dict[str, Any]:
        """Suggests num_batch based on available VRAM."""
        logger.info(f"Optimizing batch size (num_batch) for Ollama model: {model_key}")
        specs = self.context.get_hardware_specs()
        gpu_mem = int(specs.get("gpu_memory", 0))

        # Similar logic to LMStudioOptimizer
        if gpu_mem >= 24000:
            return {"num_batch": 512}
        elif gpu_mem >= 12000:
            return {"num_batch": 256}
        else:
            logger.warning("Low GPU memory, suggesting smaller batch size for Ollama.")
            return {"num_batch": 128}


class VLLMOptimizer(BaseOptimizer):
    """vLLM specific optimization implementation (Placeholder)."""

    def __init__(self, context: SharedContext):
        self.context = context

    def optimize_gpu_layers(self, model_key: str) -> Dict[str, Any]:
        """Suggests tensor_parallel_size for vLLM based on GPU count."""
        logger.info(f"Optimizing GPU layers (tensor_parallel_size) for vLLM model: {model_key}")
        specs = self.context.get_hardware_specs()
        gpu_count = int(specs.get("gpu_count", 1))

        # vLLM handles layer distribution well, but we can suggest tensor parallelism
        if gpu_count > 1:
            logger.info(f"Multiple GPUs detected ({gpu_count}), suggesting tensor_parallel_size={gpu_count} for vLLM.")
            return {"tensor_parallel_size": gpu_count}
        else:
            # For single GPU, vLLM manages layers automatically
            return {}

    def optimize_memory_usage(self, model_key: str) -> Dict[str, Any]:
        """Suggests gpu_memory_utilization for vLLM."""
        logger.info(f"Optimizing memory usage (gpu_memory_utilization) for vLLM model: {model_key}")
        # Suggest using a high percentage of GPU memory, vLLM manages it well
        # Avoid 1.0 to leave some buffer for OS/other processes
        return {"gpu_memory_utilization": 0.90}

    def optimize_context_length(self, model_key: str) -> Dict[str, Any]:
        """Suggests max_model_len for vLLM based on available memory."""
        logger.info(f"Optimizing context length (max_model_len) for vLLM model: {model_key}")
        specs = self.context.get_hardware_specs()
        gpu_mem = int(specs.get("gpu_memory", 0))
        total_mem = int(specs.get("total_memory", 0))

        # vLLM can often handle larger contexts due to PagedAttention
        if gpu_mem >= 24000 or total_mem >= 64000: # Higher threshold for vLLM
             # Check model's max context if available, otherwise default
            model = self.context.get_model(model_key)
            if model and hasattr(model, 'metadata') and model.metadata:
                 model_max_ctx = model.metadata.get("max_context", 16384)
            else:
                 logger.warning(f"Could not retrieve metadata for model {model_key} to determine max context. Defaulting.")
                 model_max_ctx = 16384 # Default if model or metadata not found
            suggested_ctx = min(model_max_ctx, 16384) # Cap at 16k for now
            return {"max_model_len": suggested_ctx}
        elif gpu_mem >= 12000 or total_mem >= 32000:
            return {"max_model_len": 8192}
        else:
            logger.warning("Low memory, suggesting smaller context length for vLLM.")
            return {"max_model_len": 4096}

    def optimize_batch_size(self, model_key: str) -> Dict[str, Any]:
        """Suggests max_num_batched_tokens for vLLM based on VRAM."""
        logger.info(f"Optimizing batch size (max_num_batched_tokens) for vLLM model: {model_key}")
        specs = self.context.get_hardware_specs()
        gpu_mem = int(specs.get("gpu_memory", 0))
        # vLLM's optimal batching depends heavily on PagedAttention and KV cache.
        # max_num_batched_tokens is a key parameter. Start with context length as a baseline.
        context_length_suggestion = self.optimize_context_length(model_key)
        suggested_ctx = context_length_suggestion.get("max_model_len", 4096)

        # Heuristic: Allow batching up to context size, adjust based on VRAM
        if gpu_mem >= 24000:
             # Allow potentially larger batches on high-VRAM cards
            return {"max_num_batched_tokens": max(suggested_ctx, 4096)}
        elif gpu_mem >= 12000:
            return {"max_num_batched_tokens": suggested_ctx}
        else:
            logger.warning("Low GPU memory, suggesting smaller max_num_batched_tokens for vLLM.")
            # Reduce based on context length
            return {"max_num_batched_tokens": max(suggested_ctx // 2, 1024)}

class ConfigOptimizer:
    """Main configuration optimizer class."""
    
    def __init__(self, context: SharedContext):
        self.context = context
        self.history: Dict[str, OptimizationHistory] = {}
        self.optimizers: Dict[str, BaseOptimizer] = {
            "lm_studio": LMStudioOptimizer(context),
            "ollama": OllamaOptimizer(context),
            "vllm": VLLMOptimizer(context)
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

    def optimize_prompts(self, model_key: str, task_description: Optional[str] = None,
                       task_type: str = "completion", max_iterations: int = 3) -> Any:
        """
        Optimizes prompts for a specific model using zero-shot prompt optimization.
        
        This method uses the PromptWizardOptimizer to automatically generate, evaluate,
        and refine prompts for the specified model and task without requiring labeled examples.
        
        Args:
            model_key: The unique identifier for the model
            task_description: Description of the task the model will perform
            task_type: Type of prompt template to use (e.g., "completion", "chat", "instruction")
            max_iterations: Maximum number of optimization iterations
            
        Returns:
            PromptOptimizationResult containing the optimized prompt and metrics
        """
        logger.info(f"Starting prompt optimization for model: {model_key}")
        
        # Get model information
        model = self.context.get_model(model_key)
        if not model:
            raise ValueError(f"Model {model_key} not found")
        
        # If no task description is provided, generate one based on model metadata
        if not task_description:
            task_description = f"Generate text using the {model.name} model"
            # If model has metadata, use it to enhance the task description
            if hasattr(model, 'metadata') and model.metadata:
                model_type = model.metadata.get("type", "")
                model_use = model.metadata.get("primary_use", "")
                if model_type or model_use:
                    task_description += f", which is {('a ' + model_type) if model_type else ''} " \
                                      f"model{(' specialized for ' + model_use) if model_use else ''}"
        
        # Import the PromptWizardOptimizer
        try:
            from .prompt_optimizer import PromptWizardOptimizer # Removed unused PromptOptimizationResult
            # Create prompt optimizer
            prompt_optimizer = PromptWizardOptimizer(self.context)
            
            # Run optimization
            result = prompt_optimizer.optimize(
                model_key=model_key,
                task_description=task_description,
                task_type=task_type,
                max_iterations=max_iterations
            )
            
            logger.info(f"Prompt optimization completed for model: {model_key}")
            return result
        except ImportError as e:
            logger.error(f"Failed to import PromptWizardOptimizer: {e}")
            raise ImportError(f"PromptWizardOptimizer not available. Make sure prompt_optimizer.py is in the optimizer package.")
        except Exception as e:
            logger.error(f"Error during prompt optimization for model {model_key}: {e}")
            raise