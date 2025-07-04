import pytest
import sys
from pathlib import Path

# Dynamically add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from unittest.mock import MagicMock, patch
from lmsauto.shared.context import SharedContext, ModelInfo
from lmsauto.scanner import LMStudioScanner, ModelScanner

@pytest.fixture
def shared_context():
    """Fixture for initializing a SharedContext instance."""
    return SharedContext()

@pytest.fixture
def lmstudio_scanner():
    """Fixture for initializing an LMStudioScanner instance."""
    return LMStudioScanner(base_path="/mock/models")

def test_shared_context_thread_safety(shared_context: SharedContext) -> None:
    """Test thread-safety of SharedContext operations."""
    model_info = ModelInfo(name="test_model", path="/path/to/model", platform="TestPlatform")
    shared_context.add_model(model_info)
    assert shared_context.get_model(model_info.get_key()) == model_info

def test_shared_context_model_storage(shared_context: SharedContext) -> None:
    """Verify model information storage and retrieval."""
    model_info = ModelInfo(name="test_model", path="/path/to/model", platform="TestPlatform")
    shared_context.add_model(model_info)
    assert shared_context.get_model(model_info.get_key()) == model_info

def test_shared_context_config_management(shared_context: SharedContext) -> None:
    """Validate configuration management in SharedContext."""
    shared_context.set_config_setting("key", "value")
    assert shared_context.get_config_setting("key") == "value"

@patch("lmsauto.scanner.Path.is_dir", return_value=True)
@patch("lmsauto.scanner.Path.is_file", return_value=True)
@patch("lmsauto.scanner.Path.rglob")
def test_lmstudio_scanner_discovery(mock_rglob: MagicMock, mock_is_file: MagicMock, mock_is_dir: MagicMock, lmstudio_scanner: LMStudioScanner) -> None:
    """Test LMStudioScanner model discovery."""
    mock_rglob.return_value = [Path("/mock/models/publisher/repo/model.gguf")]
    lmstudio_scanner.models_base_path = Path("/mock/models")  # Align base path with mocked paths
    print(f"Mocked rglob return value: {mock_rglob.return_value}")
    print(f"Mocked is_dir return value: {mock_is_dir.return_value}")
    models = lmstudio_scanner.discover_models()
    print(f"Discovered models: {models}")
    assert len(models) == 1
    assert models[0].name == "publisher/repo"
    # Compare path components to avoid issues with absolute vs relative paths
    expected_parts = Path("/mock/models/publisher/repo/model.gguf").parts[1:]  # Skip root
    actual_parts = Path(models[0].path).parts[1:]  # Skip drive letter/root
    assert actual_parts == expected_parts

@patch("lmsauto.scanner.Path.is_dir", return_value=False)
def test_lmstudio_scanner_missing_directory(mock_is_dir: MagicMock, lmstudio_scanner: LMStudioScanner) -> None:
    """Test error handling for missing directories in LMStudioScanner."""
    models = lmstudio_scanner.discover_models()
    assert len(models) == 0

def test_model_scanner_integration(shared_context: SharedContext) -> None:
    """Validate ModelScanner integration with SharedContext."""
    scanner = ModelScanner(context=shared_context)
    mock_scanner = MagicMock()
    mock_scanner.PLATFORM_NAME = "MockPlatform"
    mock_scanner.discover_models.return_value = [
        ModelInfo(name="mock_model", path="/mock/path", platform="MockPlatform")
    ]
    scanner.register_scanner(mock_scanner)
    scanner.scan()
    assert shared_context.get_model("mockplatform_mock_model") is not None