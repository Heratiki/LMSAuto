from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class LMSAutoContext:
    """
    Shared context object for LMSAuto components.
    Holds data passed between Scanner, Profiler, Optimizer, Validator, and Config Generator.
    """
    # From Profiler
    hardware_specs: Dict[str, Any] = field(default_factory=dict)

    # From Scanner
    scanned_models: List[Dict[str, Any]] = field(default_factory=list) # List of dicts, each representing a model

    # From Optimizer
    optimized_prompts: Dict[str, str] = field(default_factory=dict) # Model ID -> Optimized Prompt

    # From Validator
    validation_results: Dict[str, bool] = field(default_factory=dict) # Model ID -> Validation Status (True/False)

    # From Config Generator
    generated_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict) # Platform -> Config Dict

    # User Overrides / Settings
    user_settings: Dict[str, Any] = field(default_factory=dict)

    # General status or error tracking (optional)
    status: str = "Initialized"
    errors: List[str] = field(default_factory=list)