"""Tests for the validator module."""

from typing import Dict, Any

import pytest
from lmsauto.shared.context import SharedContext, ModelInfo
from lmsauto.validator import ConfigValidator


@pytest.fixture
def shared_context():
    """Fixture providing a SharedContext instance."""
    context = SharedContext()
    # Add mock hardware specs
    context.set_hardware_specs({
        "ram_gb": "16",
        "gpu_vram_mb": "8192",
        "gpu_name": "NVIDIA RTX 3070"
    })
    # Add mock model
    model = ModelInfo(
        name="test_model",
        path="/path/to/model.gguf",
        platform="lmstudio"
    )
    context.add_model(model)
    return context

@pytest.fixture
def config_validator(shared_context: SharedContext) -> ConfigValidator:
    """Fixture providing a ConfigValidator instance."""
    return ConfigValidator(shared_context)

def test_validate_valid_config(config_validator: ConfigValidator) -> None:
    """Test validation of a valid configuration."""
    config: Dict[str, Any] = {
        "context_length": 2048,
        "temperature": 0.7,
        "gpu_layers": 2
    }
    
    assert config_validator.validate("lmstudio", config, "test_model")
    results = config_validator.get_validation_results()
    assert all(r.is_valid for r in results)
    assert len(results) > 0

def test_validate_invalid_config(config_validator: ConfigValidator) -> None:
    """Test validation of an invalid configuration."""
    config: Dict[str, Any] = {
        "context_length": 100,  # Invalid - too small
        "temperature": 3.0,     # Invalid - too high
        "gpu_layers": 10        # Invalid - too many for VRAM
    }
    
    assert not config_validator.validate("lmstudio", config, "test_model")
    results = config_validator.get_validation_results()
    assert not all(r.is_valid for r in results)
    assert any("context length" in msg.lower() for r in results for msg in r.messages)
    assert any("temperature" in msg.lower() for r in results for msg in r.messages)

def test_validate_invalid_platform(config_validator: ConfigValidator) -> None:
    """Test validation with non-existent platform."""
    config = {"test": "value"}
    assert not config_validator.validate("nonexistent", config)

def test_validate_system_requirements(config_validator: ConfigValidator, shared_context: SharedContext) -> None:
    """Test system requirements validation."""
    # Test with insufficient RAM
    shared_context.set_hardware_specs({
        "ram_gb": "4",  # Below minimum
        "gpu_vram_mb": "8192",
        "gpu_name": "NVIDIA RTX 3070"
    })
    
    assert not config_validator.validate("lmstudio", {})
    results = config_validator.get_validation_results()
    assert any("Insufficient RAM" in msg for r in results for msg in r.messages)

def test_validate_model_compatibility(config_validator: ConfigValidator, shared_context: SharedContext) -> None:
    """Test model compatibility validation."""
    # Test with unsupported model format
    invalid_model = ModelInfo(
        name="invalid_model",
        path="/path/to/model.invalid",
        platform="lmstudio"
    )
    shared_context.add_model(invalid_model)
    
    assert not config_validator.validate("lmstudio", {}, "invalid_model")
    results = config_validator.get_validation_results()
    assert any("Unsupported model format" in msg for r in results for msg in r.messages)

def test_validation_results_storage(config_validator: ConfigValidator, shared_context: SharedContext) -> None:
    """Test that validation results are properly stored in shared context."""
    config = {"context_length": 2048}
    config_validator.validate("lmstudio", config)
    
    stored_results = shared_context.get_config_setting("validation_results")
    assert stored_results is not None
    assert "validation_results" in stored_results
    assert len(stored_results["validation_results"]) > 0

def test_resource_allocation_validation(config_validator: ConfigValidator) -> None:
    """Test resource allocation validation."""
    config: Dict[str, Any] = {
        "gpu_layers": 8  # Will exceed available VRAM
    }
    
    assert not config_validator.validate("lmstudio", config)
    results = config_validator.get_validation_results()
    assert any("GPU layer allocation exceeds available VRAM" in msg 
              for r in results for msg in r.messages)